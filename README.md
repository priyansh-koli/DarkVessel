# darkvessel

Detecting undeclared vessels by fusing Sentinel-1 SAR with AIS.

Built to avoid three silent errors that a straightforward implementation makes: greedy
matching that reports explainable vessels as dark, structure clustering whose answer depends
on row order, and treating an AIS archive's silence as uniform evidence when its reach is not
(see [Three errors this design avoids](#three-errors-this-design-avoids)).

## The problem

Ships are legally required to broadcast their position over AIS. Some do not: the
transponder is off, spoofed, or absent. These are *dark vessels*, and they matter — illegal
fishing, sanctions evasion, unreported transfers at sea.

Radar sees them anyway. Sentinel-1 acquires C-band SAR through cloud and darkness, and a
metal hull is a strong scatterer against near-black water. Detect every vessel in the radar
scene, match those detections against what AIS declared *at the exact instant of
acquisition*, and what is left over did not announce itself.

## Quick start

No credentials, no weights, no GPU, no network:

```bash
pip install -e ".[dev]"
darkvessel synthesise --out data/synthetic
darkvessel run --config configs/pipeline.yaml
```

```
5 detections in EPSG:25832 -> outputs/detections.gpkg
  4 matched, 1 dark, 0 in an AIS reception shadow, 0 at a fixed structure, at a tolerance of 200 m
8 declarations searched against the radar
  2 undetected, 1 below the detector's floor, 1 outside the scene
  radar and AIS agree on 4 of 6 declared, detectable, in-scene vessels (apparent recall 0.67); 1 detection no declaration explains
  40 report intervals in, 9 dropped (not_a_pair: 9, ...), 31 attributed to 53 cells at a max gap of 10 min
```

The synthetic scene exercises every branch on purpose. One of the five is a vessel under way
whose *nearest single AIS report* sits ~900 m from the target — outside any sane tolerance —
but whose interpolated position lands on it. Matched against a report taken as it stands, it
comes back as a dark vessel that was never there. That is the failure this pipeline is built
to avoid.

## The viewer

```bash
pip install -e ".[web]"
darkvessel serve --config configs/pipeline.yaml    # http://127.0.0.1:8000
```

A scene viewer with the detections overlaid, and a control panel that re-runs the chain as you
move it. It exists to answer one question per detection — *why was this classified the way it
was* — so every panel is an audit trail rather than a summary:

- **The overlay** draws each detection coloured by status, each AIS declaration, and a dashed
  line from where a moving vessel *declared* to where the radar actually *drew* it. That line
  is the azimuth-shift correction, made visible.
- **Turn the correction off** and watch the fast vessel fall out of tolerance and be reported
  dark — 4 matched / 1 dark becomes 3 matched / 2 dark. The step is not cosmetic.
- **The inspector** states the verdict in words: which declaration explained a detection, at
  what distance, whether its position was interpolated or taken from the nearest report, and
  how much the correction moved it.
- **The AIS panel** shows the cleaning report — a dark claim rests on what the search actually
  searched, so the per-rule removed-row counts are on screen rather than buried in a log.
- **The structure register** lets you register known fixed positions by clicking the scene and
  watch dark detections that stand on them reclassify.
- **The reception floor** asks how well the AIS archive reaches each place before its silence
  there counts. Turn on **Show reception shadows** and the water the archive barely hears is
  shaded; raise the floor and a dark detection standing in it is reported `shadowed` instead.
- **The other side** counts the declarations the *radar* failed to explain, and lists the
  `undetected` ones — the mirror of a dark vessel, and a recall estimate needing no labels.

**Share** copies a link that restores the whole view (every control, the register and the
selected detection), and **CSV** / **JSON** download the current run. `?select=3` alone
deep-links a detection. The viewer also ships a [Guide](src/darkvessel/web/static/guide.html)
with one-click scenarios and a [How it works](src/darkvessel/web/static/about.html) page.

To publish it without a Python host, pre-render a static bundle. It has no server, and every
control still works because each position they can take is run at build time:

```bash
darkvessel render --config configs/pipeline.yaml --out site
python -m http.server -d site 8080
```

See [docs/deployment.md](docs/deployment.md) for what each hosting shape needs and how to deploy it.

## How it works

![The darkvessel pipeline: a SAR path and an AIS path prepared independently, fused by an optimal one-to-one assignment within a tolerance, then filtered through a structure register and a reception model into five statuses — matched, structure, dark, shadowed and unsearched.](src/darkvessel/web/static/pipeline.svg)

The two paths never touch until the fusion step, and that is the point: the SAR side answers
*what is on the water*, the AIS side answers *what was declared*, and neither is allowed to
influence the other's preparation. Everything downstream of the fusion is about being honest
concerning what the answer rests on — which declarations were searched, how far it looked, and
whether a position was measured or inferred.

Read left to right, the five statuses mean five different things:

- **matched** — a declaration explains this detection, within the stated tolerance.
- **dark** — the AIS search ran and nothing explains it. *This is a claim about evidence
  searched, not a verdict.*
- **shadowed** — the search ran but could not have reached here: the archive hears this water
  too rarely to have placed a transmitting vessel at the acquisition instant, so the silence
  is not evidence.
- **structure** — it recurs at a fixed position across acquisitions, so it is not a ship.
- **unsearched** — no AIS was supplied at all. Distinct from dark, and deliberately so: a
  pipeline that reported these as dark would be manufacturing findings out of missing data.

And a run answers a second question, because the assignment already settles it: **which
declarations did no detection explain?** Those come back `explained`, `undetected`,
`below_detectable` or `outside_scene`, in their own layer. See
[The other side of the fusion](#the-other-side-of-the-fusion).

Five things carry the design:

**Interpolation to the acquisition instant.** A vessel at 12 knots covers ~370 m a minute,
more than the match tolerance. Every declaration is placed at the acquisition timestamp
before anything is compared. Nothing is ever extrapolated past the end of a track — that
would manufacture a position with no measurement behind it. Where no bracket exists the
nearest report is used and the row says so, in `position_basis`.

**Azimuth-shift correction.** SAR reads along-track position from Doppler, so a moving
target is *drawn* displaced along the satellite's ground track. Uncorrected, this alone puts
fast declared vessels hundreds of metres from their detections and reports them dark.

**Ownership-based tile dedup.** Overlapping tiles see edge targets twice. Rather than merge
detections afterwards (needing a radius to tune, and risking merging two genuinely close
hulls), each tile owns a non-overlapping core and reports only what falls inside it. Every
pixel is in exactly one core, so a target is claimed exactly once, by construction.

**Recurrence-based structure exclusion.** A position carrying a detection acquisition after
acquisition is not a ship. This needs no labels and no appearance model — only provenance.

**Reception-weighted dark claims.** An archive's reach is not uniform: terrestrial AIS
receivers hear a transponder for some tens of kilometres and no further, so reception falls
away with distance from shore — along the very gradient a dark-vessel study cares about. Every
vessel the archive holds is used as a probe for how often it would have placed a *transmitting*
vessel at the acquisition instant nearby, and that number goes on every row.

## Three errors this design avoids

**1. Optimal assignment instead of greedy matching** (`fusion/match.py`). Sorting every
(detection, declaration) pair by distance and claiming the closest first is *not* the
maximum-cardinality matching. Taking the globally-closest pair first can consume a
declaration another detection needed, leaving that detection reported **dark** despite a
valid assignment existing that explains it. Worked example, at a 100 m tolerance:

| | A1 | A2 |
|---|---|---|
| **D1** | 90 m | 95 m |
| **D2** | 98 m | 283 m ✗ |

Greedy takes (D1,A1) at 90 m, stranding D2 — even though (D1,A2) + (D2,A1) matches both.
This gets *more* likely in dense shipping lanes, exactly where such studies are sited.
Solved with `scipy.optimize.linear_sum_assignment`. Held by `tests/test_match.py`.

**2. Order-independent structure clustering** (`embed/structures.py`). Greedy
seed-and-claim grouping is row-order dependent: for A–B = 90 m, B–C = 90 m, A–C = 180 m at a
100 m tolerance, order A,B,C gives two groups (max 2 acquisitions) and B,A,C gives one group
(3 acquisitions). Whether a structure clears an exclusion floor could depend on archive row
order. Solved with union-find connected components. Across the six orderings of A, B, C,
greedy grouping returns `{(1,3), (2,2)}`; union-find returns `{(1,3)}` for all of them. Held
by `tests/test_structures.py`.

**3. An archive's silence is not uniform evidence** (`fusion/reception.py`). A map of
unexplained detections built without accounting for where the AIS archive can actually hear is
partly a map of where the receivers are, and the bias runs the wrong way: reception is worst
offshore, which is exactly where a dark vessel matters. Every vessel the archive already holds
is its own probe. For a vessel's consecutive reports a gap of `g` seconds apart, an instant
drawn from that gap is within `max_gap` `w` of a report over `min(g, 2w)` of it, so

    reception = sum(min(g, 2w)) / sum(g)

over the gaps observed nearby — the same `max_gap` the matcher uses, so one control moves both
halves of the claim. Below a stated floor a `dark` detection becomes `shadowed`. A place with
*no* estimate stays `dark` with the reason on the row: reclassifying on an absence of evidence
would be the same mistake pointed the other way. Held by `tests/test_reception.py`.

The caveat is on the tin: a vessel switching its transponder off looks exactly like a receiver
that cannot hear it, so where many vessels go dark, reception is underestimated and genuine
findings are downgraded. That is the wrong direction for enforcement and the right one for a
published claim, and the estimate is on the row either way.

A fourth, smaller bug was found while building: the bright-pixel stand-in reported one
detection per pixel across a flat plateau of equal values. Fixed with connected-component
collapsing.

## The other side of the fusion

A dark vessel is a detection no declaration explains. The same assignment answers the mirror
question for free — *which declarations did no detection explain?* — and the pipeline used to
throw that half away. `fusion/declarations.py` keeps it, in a second layer of the output
GeoPackage:

- **explained** — a detection stands where this vessel declared.
- **undetected** — inside the scene, long enough to expect, and the radar drew nothing. Either
  a position that is not true, or a hull the detector missed. Both are findings.
- **below_detectable** — shorter than the smallest vessel this detector is trusted to find, so
  a miss was expected. A declaration whose length the archive never gave is *not* put here: an
  unknown length is not a small one.
- **outside_scene** — the declared position is not in the image. The declaration side's
  `unsearched`.

Together the two sides are a confusion matrix over two sensors, and the share of declared,
detectable, in-scene vessels the radar also drew is an estimate of **detector recall on this
scene, from data nobody annotated**. It is reported with its biases stated: vessels that
declare themselves are larger and more cooperative than those that do not, and a spoofed
declaration counts against the detector although nothing was ever there to find.

## The detector

The pipeline takes its detector as a parameter, and ships three:

- **`stub`** is a deterministic stand-in for the synthetic fixture, used by the quick start.
- **`cfar`** is two-parameter CA-CFAR, the classical radar ship detector and the baseline.
- **`cnn`** is a small centre-heatmap U-Net trained on LS-SSDD-v1.0, 6,015 labelled ships in
  15 Sentinel-1 IW scenes.

Both real detectors were benchmarked the same way: threshold chosen on held-out training
scenes, then reported on the official test scenes.

| LS-SSDD test | CFAR F1 / AP | CNN F1 / AP |
|---|---|---|
| Offshore | 0.689 / 0.790 | **0.866 / 0.951** |
| Inshore | **0.279 / 0.176** | 0.193 / 0.117 |
| All | 0.547 / 0.563 | 0.541 / **0.626** |

At sea, where a dark vessel is searched for, the CNN finds 91% of ships with less than half
of CFAR's false alarms. Inshore, both are poor and the CNN is worse: urban land clutter reads
as hulls. The remedy there is a land mask, not more training. [`models/README.md`](models/README.md)
has the model card: data audit, protocol, the training anomalies found and how each was
diagnosed, and how to reproduce.

```yaml
detector: cnn                          # in a run configuration
detector_weights: models/ship_centrenet.pt
tile_px: 512
```

## Repository layout

```
src/darkvessel/
  pipeline.py   the single seam: scene + AIS + injected detector -> classified detections
  cli.py        the one command
  config.py     run configuration
  data/         study area, scene reading, tiling, provenance, AIS ingestion,
                the DMA archive, synthetic fixtures
  detect/       detector contract, stand-in, CFAR, the CNN and its training, LS-SSDD reading
                and auditing, scoring, pixel->ground, whole-scene inference
  fusion/       AIS interpolation, azimuth correction, matching, the structure register,
                AIS reception, and the declaration side of the fusion
  embed/        embedder contract, detection crops, recurrence-based structure finding
  context/      contextual variable schema
  render.py     SAR pixels -> PNG, for the viewer and the static build
  web/          the viewer: payload builder, FastAPI app, frontend
configs/        run configuration
models/         the trained detector, its model card, benchmarks and training logs
tests/          unit tests for every geometry-critical path
docs/           deployment, requirements, the goal and the progress log
Dockerfile      the live app, for any container platform
```

The working documents live in [`docs/`](docs/): [deployment.md](docs/deployment.md) (how to
host it), [requirements.md](docs/requirements.md) (what it needs to run),
[final-goal.md](docs/final-goal.md) (where it is going) and
[progress.md](docs/progress.md) (the session-by-session record).

The detector arrives as a *parameter*, never an import. That is what lets the whole chain run
and be tested with a deterministic stand-in; a trained CNN satisfying the same `Detector`
protocol is a drop-in replacement and nothing else changes — which is how the trained one
went in.

## Status

**Built and tested (241 tests passing):** the full Tier-1 core — tiling and dedup,
pixel→ground, AIS interpolation, azimuth correction, optimal matching, structure register,
recurrence clustering, the pipeline seam, the CLI, and raw-archive AIS ingestion with every
cleaning rule counted and auditable — plus reception-weighted dark claims, the declaration side
of the fusion, the viewer in both its live and static forms, and a CNN ship detector trained on
LS-SSDD and benchmarked against CFAR.

**Not yet built:** Earth Engine export and contextual sampling (distance to shore, depth,
fishing effort, EEZ join); a land mask for inshore scenes; contrastive embeddings; the
archive-wide concentration analysis and the static map.

Where results are modest they should be reported as modest. A detector that usefully *ranks*
candidates for inspection is a different and more honest claim than one that maps them.

## Licence

MIT — see [LICENSE](LICENSE). © 2026 Priyansh Koli.
