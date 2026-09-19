"""Training `CentreNet` on LS-SSDD, with every anomaly counted rather than silently absorbed.

What can go wrong in training, and what this loop does about each:

- **Class imbalance.** 82% of LS-SSDD training sub-images contain no ship, and ships cover a
  tiny fraction of the pixels in the rest. Half of every batch is therefore cropped around a
  ship, and the loss is CenterNet's penalty-reduced focal loss, normalised by ship count.
- **Crops outside the swath.** A random crop can land entirely in no-data; those are counted
  and re-drawn instead of teaching the network that the swath edge is the sea.
- **Non-finite loss.** The step is skipped and counted; three in an epoch halve the learning
  rate, because the usual cause is a step size the loss surface cannot take.
- **Exploding gradients.** Gradients are clipped, and every clipped step is counted.
- **Validation AP swinging between epochs.** With the training loss falling smoothly,
  validation AP jumped between 0.27 and 0.70 from one epoch to the next
  (`models/logs/1_batchnorm.log`). Three explanations were tested against the *same* weights
  (the runs are seeded, so their losses agree to five decimals):
  BatchNorm statistics learnt on ship-centred crops — recomputing them on whole, mostly-sea
  images made AP far worse, 0.18 against 0.61 (`logs/3_batchnorm_recalibrated_on_sea.log`);
  BatchNorm statistics lagging the weights — recomputing them on fresh training batches left
  the swings as they were (`logs/4_batchnorm_precise.log`); GroupNorm, which removes both,
  trained half as fast and plateaued lower (`logs/2_groupnorm.log`). What remained was the
  weights themselves oscillating: at a high learning rate each epoch lands on a slightly
  different fit to the training scenes, and the held-out scenes see the difference. So what
  is evaluated and shipped is an exponential moving average of the weights, with BatchNorm
  statistics recomputed for those averaged weights (`logs/5_batchnorm_ema_final.log`).
- **False alarms the validation scenes never showed.** The first model found 91% of offshore
  test ships but raised 1773 false alarms inshore, 80% of them in 66 images: dense urban land
  whose canal banks read as hulls against a bright median, and a band of noisy low-backscatter
  sea at a swath edge. Random crops rarely land on either. `hard_negative_fraction` fine-tunes
  on the current model's false alarms in the *training* scenes (never test). Tried, and it did
  not help: 12,751 false alarms were mined and learnt, yet inshore test AP fell from 0.117 to
  0.100 (`models/logs/6_hard_negatives.log`) — the training scenes' land does not look like
  the test scenes' city. Kept off by default; the shipped model does not use it. The remedy
  for inshore clutter in this pipeline is a land mask, not more of the same data.
- **Overfitting.** Validation AP is measured every epoch on whole held-out scenes; the best
  epoch's weights are kept, and training stops once it has not improved for `patience` epochs.

Everything above lands in the returned history and in `<out>.json` next to the weights.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from darkvessel.detect import evaluate
from darkvessel.detect import lsssdd as L
from darkvessel.detect.cnn import CentreNet, default_device, peaks, prepare


@dataclass
class TrainConfig:
    epochs: int = 40
    steps_per_epoch: int = 150
    batch_size: int = 16
    crop_px: int = 256
    ship_fraction: float = 0.5
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    patience: int = 8
    width: int = 16
    norm: str = "batch"
    recalibration_batches: int = 40
    ema_decay: float = 0.998
    finetune_from: str | None = None
    hard_negative_fraction: float = 0.0
    hard_negative_floor: float = 0.2
    seed: int = 0


def train(root: Path, out: Path, config: TrainConfig | None = None, log=print) -> dict:
    config = config or TrainConfig()
    _seed(config.seed)
    device = torch.device(default_device())

    audit = L.Audit()
    train_set = L.load(root, L.split_names(root, "train"), audit)
    val_set = L.load(root, L.split_names(root, "val"), audit)
    log(f"loaded {len(train_set)} train and {len(val_set)} validation sub-images")
    for line in audit.lines():
        log("  audit " + line)

    images = {s.name: _read_u8(s.image_path) for s in train_set}
    with_ships = [s for s in train_set if len(s.boxes)]
    sampler = _Sampler(train_set, with_ships, images, config)
    anomalies: Counter = Counter()

    model = CentreNet(width=config.width, norm=config.norm).to(device)
    if config.finetune_from:
        model.load_state_dict(torch.load(config.finetune_from, map_location="cpu")["state_dict"])
        log(f"fine-tuning from {config.finetune_from}")
    if config.hard_negative_fraction > 0:
        sampler.false_alarms = mine_false_alarms(
            model, train_set, images, config.hard_negative_floor, device
        )
        mined = sum(len(v) for v in sampler.false_alarms.values())
        anomalies["false alarms mined from training scenes"] = mined
        log(f"mined {mined} false alarms in {len(sampler.false_alarms)} training sub-images")
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    total_steps = config.epochs * config.steps_per_epoch
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))
    )
    lr_scale = 1.0
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config.ema_decay)
    )

    history: list[dict] = []
    best = {"ap": -1.0, "epoch": -1, "state": None, "threshold": 0.3}
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses, epoch_anomalies = [], Counter()
        for _ in range(config.steps_per_epoch):
            x, heat, n_ships = sampler.batch(epoch_anomalies)
            x, heat = x.to(device), heat.to(device)
            loss = focal_loss(model(x), heat, n_ships)
            if not torch.isfinite(loss):
                epoch_anomalies["non-finite loss (step skipped)"] += 1
                optimiser.zero_grad(set_to_none=True)
                continue
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if not torch.isfinite(norm):
                epoch_anomalies["non-finite gradient (step skipped)"] += 1
                optimiser.zero_grad(set_to_none=True)
                continue
            if norm > config.grad_clip:
                epoch_anomalies["gradient clipped"] += 1
            optimiser.step()
            schedule.step()
            ema.update_parameters(model)
            losses.append(float(loss.detach()))

        if epoch_anomalies["non-finite loss (step skipped)"] >= 3:
            lr_scale *= 0.5
            for group in optimiser.param_groups:
                group["initial_lr"] = config.lr * lr_scale
            schedule.base_lrs = [config.lr * lr_scale for _ in schedule.base_lrs]
            epoch_anomalies["learning rate halved"] += 1

        # Averaged weights have no running statistics of their own: recompute them.
        averaged = ema.module
        if config.norm == "batch":
            recalibrate_batchnorm(averaged, sampler, config.recalibration_batches, device)
        val = validate(averaged, val_set, device)
        anomalies.update(epoch_anomalies)
        record = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 5) if losses else None,
            "val_ap": round(val["ap"], 4),
            "val_best_f1": val["best"].as_dict(),
            "lr": optimiser.param_groups[0]["lr"],
            "anomalies": dict(epoch_anomalies),
            "minutes": round((time.time() - started) / 60, 1),
        }
        history.append(record)
        log(
            f"epoch {epoch:3d}  loss {record['train_loss']}  val AP {val['ap']:.3f}  "
            f"F1 {val['best'].f1:.3f} @ {val['best'].threshold:.2f}  "
            + (f"anomalies {dict(epoch_anomalies)}" if epoch_anomalies else "")
        )

        if val["ap"] > best["ap"]:
            best = {
                "ap": val["ap"],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in averaged.state_dict().items()},
                "threshold": val["best"].threshold,
                "val_best": val["best"].as_dict(),
            }
        elif epoch - best["epoch"] >= config.patience:
            log(f"stopping: validation AP has not improved for {config.patience} epochs")
            anomalies["early stop (validation plateau)"] += 1
            break

    summary = {
        "model": "CentreNet",
        "width": config.width,
        "norm": config.norm,
        "trained_on": "LS-SSDD-v1.0 train scenes " + ",".join(
            s for s in L.TRAIN_SCENES if s not in L.VAL_SCENES
        ),
        "validated_on": "LS-SSDD-v1.0 scenes " + ",".join(L.VAL_SCENES),
        "best_epoch": best["epoch"],
        "val_ap": round(best["ap"], 4),
        "val_best_f1": best.get("val_best"),
        "threshold": best["threshold"],
        "config": asdict(config),
        "data_audit": dict(audit.counts),
        "training_anomalies": dict(anomalies),
        "history": history,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], **{k: summary[k] for k in (
        "width", "norm", "threshold", "best_epoch", "val_ap", "trained_on", "validated_on"
    )}}, out)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


def focal_loss(logits: torch.Tensor, target: torch.Tensor, n_ships: int) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss; near-miss pixels are penalised less."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1).float()
    negative_weight = torch.pow(1 - target, 4)
    pos_loss = torch.log(p) * torch.pow(1 - p, 2) * positive
    neg_loss = torch.log(1 - p) * torch.pow(p, 2) * negative_weight * (1 - positive)
    return -(pos_loss.sum() + neg_loss.sum()) / max(n_ships, 1)


@torch.no_grad()
def recalibrate_batchnorm(model, sampler: _Sampler, batches: int, device) -> None:
    """Recompute every BatchNorm's statistics with the current weights, over fresh batches from
    the training sampler, equally weighted — the distribution the layers were trained against.

    The sampler's random state is saved and restored, so recalibrating does not change which
    crops training sees next.
    """
    state = random.getstate(), np.random.get_state()
    norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # a cumulative average over the pass, not an exponential one
    model.train()
    for _ in range(batches):
        x, _, _ = sampler.batch(Counter())
        model(x.to(device))
    for m, momentum in zip(norms, momenta):
        m.momentum = momentum
    model.eval()
    random.setstate(state[0])
    np.random.set_state(state[1])


@torch.no_grad()
def mine_false_alarms(model, samples, images, floor: float, device) -> dict[str, np.ndarray]:
    """Where `model` fires on a training sub-image and no labelled ship is: (row, col) each."""
    model.eval()
    found = {}
    for sample in samples:
        x = torch.from_numpy(prepare(images[sample.name].astype(np.float32) / 255.0))[None]
        x = x.to(device)
        points = peaks(torch.sigmoid(model(x))[0, 0], x[0, 1], floor=floor)
        false = points[~evaluate.match_image(points, sample.boxes, margin=8.0)]
        if len(false):
            found[sample.name] = false[:, :2]
    return found


@torch.no_grad()
def validate(model, samples: list[L.Sample], device) -> dict:
    model.eval()
    results = []
    for sample in samples:
        x = torch.from_numpy(prepare(L.read_image(sample.image_path)))[None].to(device)
        heat = torch.sigmoid(model(x))[0, 0]
        results.append((peaks(heat, x[0, 1], floor=0.02), sample.boxes))
    scores, hits, ships = evaluate.pooled(results)
    return {
        "ap": evaluate.average_precision(scores, hits, ships),
        "best": evaluate.best_f1(scores, hits, ships),
    }


def render_heatmap(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Gaussian bumps at ship centres, peak exactly 1 at the centre pixel."""
    heat = np.zeros(shape, dtype=np.float32)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    for x0, y0, x1, y1 in boxes:
        r, c = int((y0 + y1) / 2), int((x0 + x1) / 2)
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            continue
        sigma = float(np.clip(min(x1 - x0, y1 - y0) / 6, 1.0, 4.0))
        reach = int(3 * sigma) + 1
        r0, r1 = max(r - reach, 0), min(r + reach + 1, shape[0])
        c0, c1 = max(c - reach, 0), min(c + reach + 1, shape[1])
        distance2 = (rows[r0:r1, c0:c1] - r) ** 2 + (cols[r0:r1, c0:c1] - c) ** 2
        bump = np.exp(-distance2 / (2 * sigma**2))
        heat[r0:r1, c0:c1] = np.maximum(heat[r0:r1, c0:c1], bump)
        heat[r, c] = 1.0
    return heat


class _Sampler:
    def __init__(self, samples, with_ships, images, config: TrainConfig):
        self.samples, self.with_ships = samples, with_ships
        self.images, self.config = images, config
        self.false_alarms: dict[str, np.ndarray] = {}
        self.by_name = {s.name: s for s in samples}

    def batch(self, anomalies: Counter):
        xs, heats, ships = [], [], 0
        size = self.config.crop_px
        while len(xs) < self.config.batch_size:
            draw = random.random()
            hard = self.config.hard_negative_fraction
            if draw < hard and self.false_alarms:
                name = random.choice(list(self.false_alarms))
                sample = self.by_name[name]
                r, c = self.false_alarms[name][random.randrange(len(self.false_alarms[name]))]
                cr = int(np.clip(r - random.uniform(16, size - 16), 0, L.SIZE - size))
                cc = int(np.clip(c - random.uniform(16, size - 16), 0, L.SIZE - size))
            elif draw < hard + self.config.ship_fraction and self.with_ships:
                sample = random.choice(self.with_ships)
                x0, y0, x1, y1 = sample.boxes[random.randrange(len(sample.boxes))]
                cr = int(np.clip((y0 + y1) / 2 - random.uniform(16, size - 16), 0, L.SIZE - size))
                cc = int(np.clip((x0 + x1) / 2 - random.uniform(16, size - 16), 0, L.SIZE - size))
            else:
                sample = random.choice(self.samples)
                cr, cc = random.randrange(L.SIZE - size + 1), random.randrange(L.SIZE - size + 1)
            crop = self.images[sample.name][cr : cr + size, cc : cc + size]
            if (crop == 0).mean() > 0.98:
                anomalies["crop outside the swath (re-drawn)"] += 1
                continue
            boxes = sample.boxes - np.array([cc, cr, cc, cr], dtype=np.float32)
            centre_r = (boxes[:, 1] + boxes[:, 3]) / 2
            centre_c = (boxes[:, 0] + boxes[:, 2]) / 2
            boxes = boxes[(centre_r >= 0) & (centre_r < size) & (centre_c >= 0) & (centre_c < size)]
            x = prepare(crop.astype(np.float32) / 255.0)
            heat = render_heatmap(boxes, (size, size))
            # Ships have no preferred orientation in a radar scene: flips and quarter turns are
            # label-preserving.
            k, flip = random.randrange(4), random.random() < 0.5
            x, heat = np.rot90(x, k, axes=(1, 2)), np.rot90(heat, k)
            if flip:
                x, heat = x[:, :, ::-1], heat[:, ::-1]
            xs.append(np.ascontiguousarray(x))
            heats.append(np.ascontiguousarray(heat))
            ships += len(boxes)
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(heats))[:, None], ships


def _read_u8(path: Path) -> np.ndarray:
    return np.round(L.read_image(path) * 255).astype(np.uint8)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
