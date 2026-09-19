# Ship detector: `ship_centrenet.pt`

A small U-Net (0.39 M parameters, 1.6 MB) that predicts a heatmap of ship centres in a
Sentinel-1 scene. It satisfies the pipeline's `Detector` protocol, so it's used by setting
`detector: cnn` and `detector_weights: models/ship_centrenet.pt` in a run configuration. Code:
[`detect/cnn.py`](../src/darkvessel/detect/cnn.py) (model and detector),
[`detect/train.py`](../src/darkvessel/detect/train.py) (training),
[`detect/benchmark.py`](../src/darkvessel/detect/benchmark.py) (evaluation).

## Data

LS-SSDD-v1.0 (Zhang et al., *Remote Sensing* 2020): 15 Sentinel-1 IW scenes cut into 9,000
labelled 800×800 sub-images, 6,015 ships. It comes from the authors' release,
https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN (Apache-2.0). It isn't in this repository;
see `requirements.md` for where to put it.

| Split | Scenes | Sub-images | Ships | Used for |
|---|---|---|---|---|
| train | 01–04, 06–09 | 4,800 | 2,930 | fitting the weights |
| validation | 05, 10 | 1,200 | 707 | choosing the epoch and the threshold |
| test | 11–15 (official) | 3,000 | 2,377 | the numbers below, and nothing else |

Validation is held out **by scene**. Neighbouring sub-images share sea state and often the
same ships, so a random split would leak.

`darkvessel audit-data` checks every image and label before use. It found:
- **The splits are clean.** No sub-image or scene is on both sides, and inshore plus offshore
  equals test.
- **One-off checks run alongside it also passed.** No labelled ship sits on no-data pixels,
  and the sub-image labels agree with the full-scene labels scene by scene, totalling 6,015 as
  published.
- **One ship is boxed twice**, in test image `12_16_16` (IoU 0.8). It's merged; left in, it
  would count as a miss against every detector.
- **273 sub-images are mostly outside the radar swath.** They're kept, and the no-data pixels
  are masked.
- **Heavy class imbalance.** 82% of training sub-images contain no ship.
- **Test ships are larger than training ships** (median 22 px vs 15 px).

## Results

Each detector's threshold is chosen on the validation scenes, then applied unchanged to the
test scenes. A prediction counts as a hit when it falls inside a ship's box (3 px margin),
paired one-to-one by optimal assignment. AP is threshold-free.

| LS-SSDD test | Detector | Precision | Recall | F1 | AP |
|---|---|---|---|---|---|
| **Offshore** (2,234 images) | CFAR | 0.596 | 0.817 | 0.689 | 0.790 |
| | **CNN** | **0.824** | **0.912** | **0.866** | **0.951** |
| **Inshore** (766 images) | **CFAR** | **0.265** | 0.295 | **0.279** | **0.176** |
| | CNN | 0.138 | **0.321** | 0.193 | 0.117 |
| **All** (3,000 images) | CFAR | **0.488** | 0.623 | **0.547** | 0.563 |
| | CNN | 0.444 | **0.692** | 0.541 | **0.626** |

The baseline is two-parameter CA-CFAR ([`detect/cfar.py`](../src/darkvessel/detect/cfar.py)),
the classical radar detector, with its windows and threshold tuned on the same validation
scenes. Speed per 800×800 image, including JPEG decoding: CNN 46 ms on an Apple M5 GPU, CFAR
111 ms on its CPU.

**What this means.**
- **At sea, the CNN is clearly better.** It finds 91% of ships with less than half of CFAR's
  false alarms, and open sea is where a dark vessel is searched for.
- **Inshore, both detectors are poor, and the CNN is worse.** Its false alarms concentrate in a
  few scenes: dense urban land, whose canal banks read as hulls against a bright median, and a
  band of noisy low-backscatter sea at a swath edge.
- **Overall F1 is a tie; overall AP favours the CNN.** Milestone 4 in `Final_goal.md` ("beats
  the CFAR baseline") holds offshore and on AP, but not on inshore F1.
- **Mask land before detection.** A coastline mask with a buffer takes away most of what the
  inshore numbers measure, and the planned SAR preprocessing includes one.
- **LS-SSDD is 8-bit imagery.** Input normalisation is scale-invariant (tested), but the model
  still needs checking on one calibrated scene before its results are trusted.

## Training, and the anomalies found in it

The final model trained for 30 epochs, then stopped after 8 epochs without improvement, in
about an hour on an M5 GPU. It uses half ship-centred 256 px crops, penalty-reduced focal
loss, AdamW with cosine decay, and an exponential moving average of the weights. Each run's
log is in [`logs/`](logs/); the full history is in `ship_centrenet.json`.

1. **Validation AP swung between epochs** (0.27 ↔ 0.70) while the training loss fell smoothly
   ([`1_batchnorm.log`](logs/1_batchnorm.log)). Runs are seeded, so three explanations could
   be tested against identical weights:

   | Explanation | Test | Result |
   |---|---|---|
   | BatchNorm statistics learnt on ship-biased crops | Recompute them on mostly-sea images | **Rejected:** AP 0.18 vs 0.61 ([log](logs/3_batchnorm_recalibrated_on_sea.log)) |
   | BatchNorm statistics lagging the weights | Recompute them on fresh training batches | **Rejected:** same swings ([log](logs/4_batchnorm_precise.log)) |
   | GroupNorm instead | Retrain | Stable, but half the speed and a lower plateau ([log](logs/2_groupnorm.log)) |
   | The weights themselves oscillating | Evaluate and ship an EMA of the weights | **Fixed:** AP rises monotonically ([log](logs/5_batchnorm_ema_final.log)) |

2. **Validation didn't predict inshore performance.** Validation scenes 05 and 10 contain
   little land, so validation F1 0.69 turned into 0.19 on inshore test. Hard-negative
   fine-tuning mined the model's 12,751 false alarms in the *training* scenes and learnt them,
   but inshore test AP **fell** (0.117 → 0.100,
   [`6_hard_negatives.log`](logs/6_hard_negatives.log), [benchmark](benchmark_cnn_hard_negatives.json)).
   The training scenes' land doesn't look like the test city. That model isn't shipped.

3. **Counted throughout, and benign.**
   - Gradient clipping fired on 24% of steps, most often early, typical of focal loss at the
     start of training.
   - 1,185 crops that fell outside the swath were re-drawn.
   - No non-finite losses occurred.

## Reproduce

```bash
pip install -e ".[detector]"
gdown --fuzzy "https://drive.google.com/file/d/1-fSKYsKr5wuYuXJE2CDX_Yfr4Ylwjfqm/view" -O data/lsssdd/LS-SSDD-v1.0-OPEN.zip
unzip data/lsssdd/LS-SSDD-v1.0-OPEN.zip -d data/lsssdd "LS-SSDD-v1.0-OPEN/JPEGImages_sub/*" \
  "LS-SSDD-v1.0-OPEN/Annotations_sub/*" "LS-SSDD-v1.0-OPEN/Annotations/*" "LS-SSDD-v1.0-OPEN/ImageSets/*"
darkvessel audit-data
darkvessel train --out models/ship_centrenet.pt           # seed 0 reproduces this model
darkvessel evaluate --detector cnn --out models/benchmark_cnn.json
darkvessel evaluate --detector cfar --out models/benchmark_cfar.json
```
