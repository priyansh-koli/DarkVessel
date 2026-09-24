# darkvessel — progress log

The running record of what has been done on this project, one entry per work session, newest
first. Read this, [requirements.md](requirements.md) and [final-goal.md](final-goal.md) at the
start of any new session to pick up where the last one stopped.

## How to use this file

**At the start of a session:** read "Current state" and the latest entry, then run the
health check below. If the numbers don't match what is written here, trust the code and fix
this file first.

**At the end of a session:** add a new entry at the top of the log (template at the bottom),
and update "Current state" and "Next steps" to match. Keep the entries factual: what changed,
what was verified and how, what broke, and what was left undone.

### Health check

Run from the project root:

```bash
source .venv/bin/activate          # if missing: python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
pytest -q                           # expect: 262 passed
ruff check src/ tests/              # expect: All checks passed!
darkvessel synthesise --out data/synthetic
darkvessel run --config configs/pipeline.yaml
# expect: 5 detections in EPSG:25832 -> outputs/detections.gpkg
#           4 matched, 1 dark, 0 in an AIS reception shadow, 0 at a fixed structure, ...
#         8 declarations searched against the radar
#           2 undetected, 1 below the detector's floor, 1 outside the scene
#           radar and AIS agree on 4 of 6 declared, detectable, in-scene vessels ...
```

Without the `detector` extra, 5 of those tests (the ones needing torch) are skipped, and pytest
reports `257 passed, 5 skipped` — that is what CI shows, since it installs `.[dev]` only.

The detector, if the LS-SSDD data is in `data/lsssdd/` (see [models/README.md](../models/README.md)
for the download):

```bash
pip install -e ".[detector]"
darkvessel audit-data              # expect: splits clean; test 3000 sub-images, 2377 ships
darkvessel evaluate --detector cnn # expect: test_offshore F1 0.866, test F1 0.541 (~3 min on M5)
```

---

## Current state (as of 2026-09-24)

| Area | Status |
|---|---|
| Tier 1 — core pipeline (tiling, pixel→ground, AIS interpolation, azimuth correction, optimal matching, structure register, recurrence clustering, CLI) | **Done**, tested |
| AIS reception — per-place estimate of whether the archive could have placed a transmitting vessel, and the `shadowed` status | **Done**, tested. Validated only against the synthetic fixture |
| The declaration side — declarations no detection explains, and the label-free recall estimate | **Done**, tested. Written as a second GeoPackage layer |
| AIS ingestion — DMA CSV reader and cleaning rules with per-rule counts | **Done**, tested against the format; not yet run on a real DMA day file |
| Viewer — FastAPI live app + static bundle (every control baked), radar display, share panel, Guide and How it works pages | **Done**, every control checked in Chrome in both modes, at desktop and phone widths. The radar display and share panel are **not yet published** (see Git below) |
| Deployment — CI, GitHub Pages | **Live.** CI green on 3.9 and 3.12; static viewer at https://priyansh-koli.github.io/DarkVessel/ |
| Deployment — Dockerfile (live app) | **Written, never built** (no docker on the dev machine) |
| Detector | `stub` (synthetic, the default), `cfar` baseline, `cnn` trained on LS-SSDD — see `models/README.md`. CNN beats CFAR offshore (F1 0.87 vs 0.69) and on AP, loses inshore (0.19 vs 0.28) until land is masked. Not yet run on a real georeferenced scene |
| Tier 2 — Earth Engine export and contextual variables | **Not started** (`context/gee_layers.py` is a no-op) |
| Tier 3 — CNN detector on LS-SSDD | **Done** (benchmarked); contrastive embeddings **not started** |
| Analysis — archive-wide concentration analysis, static map | **Not started** |
| Real data | **Training data only.** LS-SSDD-v1.0 (real Sentinel-1 chips, labelled) is in `data/lsssdd/` (7.8 GB zip + 2.7 GB extracted, git-ignored). The pipeline itself still runs only on the synthetic fixture: no real scene or DMA AIS file yet |

- Tests: 262 passing locally (257 + 5 skipped in CI, which has no torch). Lint: clean.
- Git: `main` tracks `origin` = https://github.com/priyansh-koli/DarkVessel (public). Every push
  to `main` runs CI and republishes the viewer. For the current commit and tree state, run
  `git log --oneline -5` and `git status -sb`. This file deliberately records no hash, since
  committing it would make the hash stale.
- Python: the dev machine has only system Python 3.9.6, so the project targets `>=3.9`.

## Next steps

In rough priority order — see [final-goal.md](final-goal.md) for why.

0. Decide the bracket-width question in `fusion/interpolate.py` (see Open questions): the
   reception work made it concrete rather than theoretical.
1. Run the pipeline on one real Sentinel-1 scene and the matching DMA AIS day file, with
   `detector: cnn` — and check the CNN on calibrated backscatter (it was trained on 8-bit chips).
2. Choose the land polygons (OSM land polygons or GSHHG) and the buffer width for the study
   area, and re-benchmark the CNN inshore with the mask on. The mask itself is built
   (`data/land.py`, `land_path` / `land_buffer_m`) but has only been run on constructed land.
3. Build the Earth Engine contextual layers (distance to shore, depth, fishing effort, EEZ).
4. Archive-wide run and the concentration analysis and map.

## Open questions

- The azimuth-shift model in `fusion/azimuth.py` and the cleaning thresholds in `data/ais.py`
  are only validated against the synthetic fixture. Check both against real Sentinel-1 / DMA
  data and the SAR literature before trusting real-world results.
- The study area (EPSG:25832 → Danish waters) and the time range for the real archive have
  not been fixed yet.
- **Two interpolation behaviours in `fusion/interpolate.py` need a decision** (found
  2026-09-18, deliberately not changed because they alter results): a pair of reports
  bracketing the acquisition is interpolated however far apart they are (`max_gap` only limits
  a lone report); and when the acquisition precedes a track, velocity comes from the track's
  *last* two reports, not its first two. The first of these got sharper on 2026-09-23: the
  reception work added a fixture vessel heard once an hour, and it is *declared* at the default
  10-minute max gap, with `position_age_s` of 0, because two of its hourly reports happen to
  bracket the pass. A straight line drawn across an hour is a fabrication at any speed. The
  bracket's width is now recorded as `position_span_s` on every row so the problem is visible,
  but the behaviour is unchanged and still needs a decision — most likely bounding a bracket by
  some multiple of `max_gap`, which would change match results.
- **How the reception floor should be set on real data.** 0.5 is a placeholder chosen so the
  fixture tells its story; on a real archive it should come from what the reception estimate
  actually looks like over the study area, and the choice needs stating in the write-up.
- **The CNN was trained on 8-bit JPEG chips.** Its input normalisation is scale-invariant (and
  tested), but it has not yet seen calibrated σ⁰ backscatter. Check it on one real scene
  before trusting its detections there.
- **How to mask land** before detection (coastline source, buffer width) — the fix for the
  CNN's inshore false alarms.

## Things worth knowing

- **The synthetic fixture has 5 targets, one per code path:** `stationary_match`,
  `simple_match`, `interpolated_match` (its nearest single AIS report is 900 m away, but its
  two bracketing reports interpolate exactly onto it), `azimuth_corrected_match` (~511 m off
  until the correction is applied; its raw AIS position is computed by inverting
  `Geometry.displacement()`, so it stays correct if the formula changes), and `dark_vessel`.
- **The fixture's dark vessel has a declaration 170 m away — inside the 200 m tolerance.** It
  stays dark only because optimal assignment gives that declaration to `simple_match` (30 m
  away). A matcher letting one declaration explain two detections would hide it. Pinned in
  `test_synthetic.py` and `test_pipeline.py`.
- **Matches land ~7 m off, not 0 m.** Each target is painted as a 2×2 pixel plateau whose
  centre is half a pixel from its ground coordinate. Tests assert against that residual.
- **Viewer behaviours to expect:** turning off the azimuth correction changes 4 matched /
  1 dark into 3 matched / 2 dark; registering a fixed position under the dark vessel
  reclassifies it to `structure` without touching any match. `?select=3` deep-links a detection.
- **The fixture has four AIS-only vessels (`2191000xx`)** carrying the two newer features: a
  lane reporting every two minutes 245 m east of the dark vessel (its dense reporting is what
  makes that darkness mean something, and nothing is detected where it declares, so it is
  `undetected`); a drifter heard once an hour beside `faint_dark` (which puts that
  neighbourhood's reception at 1/3, so lowering the threshold to 0.3 reports `faint_dark`
  `shadowed`); a 12 m vessel (`below_detectable`); and one declared outside the image
  (`outside_scene`). None of them sits within the 200 m tolerance of any painted target —
  pinned by `test_synthetic.py`, because one that did would quietly become a fifth story.
- **`faint_trawler` and `faint_dark` are deliberately different kinds of unknown.** At
  threshold 0.3 the first is `dark` with `insufficient_evidence` (nothing measures the
  archive's reach there) and the second is `shadowed` (measured, and low). They look alike and
  are not: one search came back empty, the other never reached.
- **The fixture uses `reception_cell_m: 200`**, not the library default of 1000, because the
  synthetic scene is only 1.6 km across. A real Sentinel-1 scene wants the default or more.
- **The static bundle roughly tripled** (11 MB to ~33 MB) because reception moves with the max
  gap, so the gap equivalence classes fragmented from about 5 to all 11. Page weight per
  visitor is unchanged — runs are fetched lazily — and `site/` is git-ignored and CI-built.
- **Screenshots:** headless Chrome on macOS clamps windows to 500 px minimum width, so a
  414 px screenshot is a cropped 500 px render and looks broken when it isn't. Render inside a
  414 px iframe to see the real mobile layout.
- **Python 3.9 rules:** use `from __future__ import annotations` for `X | None` hints; no
  `zip(strict=True)` or other 3.10+ features.
- **Each detector reads `detector_threshold` in its own units:** brightness (stub), clutter
  standard deviations (cfar; default 6.0, calibrated on validation), heatmap score (cnn; 0.30,
  shipped in the weights). Leave it out for cfar/cnn. The viewer always runs the stub,
  whatever the config says.
- **Leave `tile_px`/`overlap_px` out for `cnn` and `cfar`:** each then uses its
  `preferred_tiling` (1024/64). The CNN gives the same detections at any tile size (checked
  again 2026-09-24 at 512 and 1024 on a real LS-SSDD mosaic); CFAR needs an overlap of at least
  its `context_px` (40) or tile edges truncate its clutter window, and the CLI warns.
- **Training is seeded and deterministic enough to compare runs:** two runs with the same seed
  agree on training loss to five decimals, which is how the BatchNorm hypotheses were tested
  on identical weights. A full run is ~1 hour on the M5 (MPS); GroupNorm doubles that.
- **Validation scenes 05 and 10 contain little land,** so validation scores predict offshore
  test performance well and inshore poorly. Don't read validation F1 as an inshore number.
- **Playwright + SVG:** `inner_text()` returns `None` for SVG `<text>`; use `text_content()`.
  To click a detection, target `.mark-halo`, not the `.mark` group — the group's bounding box
  includes the speed label, and its centre falls in empty space.
- The test suite was mutation-checked: breaking tile stepping, the max-gap drop, the dark-only
  register filter, the incidence term, plateau collapsing, tile ownership, context absent
  values, crop zero-padding or the duplicate-AIS rule each turns it red.

---

## Session log

### 2026-09-24 — real-scene reading, land mask, faster CFAR and AIS interpolation

- **`data/scene.py`.** `image.npy` is now opened as a memory map, and a scene can instead be a
  GeoTIFF named by `"image"` in `scene.json` (`GeoTiffImage`), read one window at a time, with
  its transform and CRS from the file, no-data and NaN read as 0. GDAL's block cache is capped
  at 64 MB per read unless `GDAL_CACHEMAX` is set: uncapped, it grew to hold the whole file
  (825 MB for a 768 MB scene, against 53 MB capped). CFAR over that scene now peaks at 564 MB
  from a GeoTIFF against 1274 MB loaded whole; the rest is CFAR's per-tile working arrays.
  `embed/crops.py` reads one window per crop instead of padding the whole scene.
- **`data/land.py` (new) and the `masked` verdict.** `LandMask` reads land polygons near the
  scene from any vector file, in any CRS, and buffers them. `detect_scene` skips tiles whose
  core is all land without reading them, zeroes masked pixels, and drops any detection on the
  mask. A declaration standing on the mask is now `masked`, not `undetected`, so the mask is
  never counted against the detector; `Agreement.masked` counts them. CLI, live viewer, static
  bundle and the viewer's panel all carry it.
- **Tiling.** `tile_px`/`overlap_px` are now optional; left out, a detector's
  `preferred_tiling` is used (CFAR and CNN: 1024/64; stub: 128/32). The CLI warns when a stated
  overlap is under CFAR's `context_px`. On a 4800 px LS-SSDD mosaic, 1024 px tiles gave the CNN
  the same 14 ships as 512 px, 17% faster; batching several windows per forward pass was tried
  and removed, as it gained nothing on the M5 GPU.
- **`detect/cfar.py`.** Blob sizes, peaks and centroids are measured over the labelled pixels
  only; `ndimage.maximum` was sorting every pixel of every tile. CFAR on the mosaic: 3.72 s to
  1.65 s, and across 20,251 blobs the output moved by at most 3e-5 px. Each box count is
  filtered once instead of twice. The old 128/32 tiling changed 1 of 24 detections on the
  mosaic against a whole-scene pass (a truncated clutter window); the new default matches it,
  and a test pins that.
- **`fusion/interpolate.py`.** `positions_at` is vectorised (one sort, then counting per
  vessel) instead of a Python loop per MMSI: 0.78 s to 0.31 s on 600k reports / 1,500
  vessels. Checked equal to the old code on 400 random archives; the open velocity question is
  kept exactly as it was. Fixed on the way: codes are numbered after dropping unusable rows, so a
  vessel with no usable report cannot shift the others.
- **Also fixed.** `affine` 3 deprecates `*` for transforms, which produced 474,742 warnings per
  test run; all uses are now `@` and `affine>=3.0` is pinned. `darkvessel` prints a one-line
  error for a missing file, extra or setting instead of a traceback (`DARKVESSEL_DEBUG=1`
  restores it). A GeoTIFF that failed to open raised again in its destructor.
- **Verified:** 261 tests pass, ruff clean, the CI quick-start greps pass, and every new setting
  was run through `darkvessel run`, the live API (`TestClient`) and `darkvessel render`.
- **Left undone:** no real coastline file has been tried; the viewer still re-runs the stub
  whatever the configured detector; the viewer's scene PNG still reads the whole image.


### 2026-09-22 — viewer layout rebalanced, document pages rebuilt, three accessibility bugs fixed

- **Filled the dead space right of the scene.** The inspector column was 345 px tall against
  the viewer column's 1007, leaving a 340×660 void. The key moved out from under the scene and
  the AIS search-space readout moved out of the controls column; both now sit under the
  inspector, which puts the evidence for a verdict beside the verdict and leaves the left
  column for controls only. The scene took the freed height (`max-height` 62vh → 72vh, so the
  radar is 648 px rather than 558). Columns now measure 988 against 999.
- **`position: sticky` had never worked anywhere on the site.** `overflow-x: hidden` on
  `html, body` makes them scroll containers, and a sticky descendant then has no scrollport to
  stick to, so it just scrolls away. That silently disabled the control column, the inspector
  column and the new contents rail. Changed to `overflow-x: clip`, which suppresses the same
  overflow without creating a scroll container. Verified: the rail now pins at 88 px through a
  3,200 px scroll, and no page overflows horizontally at 1600/1280/1024/820/390 px.
- **The overlay hid its own focusable detections.** `#overlay` carried `aria-hidden="true"`
  while containing the five detection marks, each a `role="button"` with `tabindex="0"` and a
  label — so they were in the tab order but absent from the accessibility tree, which is the
  worst of both. The blanket attribute is gone; `group()` now hides the decorative layers
  (grid, trails, lines, decls) individually and leaves `marks` exposed. All five are now in the
  tree with their labels.
- **Contrast.** `--faint` (`#64748b`) sat at 3.5–4.1:1 against the three panel backgrounds and
  failed WCAG AA everywhere it was used — hint text under every control, figure captions,
  table headers. Raised to `#7d8da3`, which measures 4.96:1 at worst and is still visibly
  dimmer than `--muted`. The counts bar also became a `<section>` so it sits in a landmark.
  axe-core now reports **0 violations** on all three pages, down from 3 types on the viewer
  and 1 each on the document pages.
- **Both document pages rebuilt around a sticky contents rail** with scroll-spy
  (`aria-current="location"`), which collapses back to the pill list below 1080 px. Added
  self-linking headings with real anchors that are keyboard-reachable and carry an accessible
  name, copy buttons on code blocks that fall back to selecting the text where the clipboard
  API is refused, and a back-to-top control that moves focus to the `h1` rather than leaving
  it on a button that has just vanished. All of it is progressive enhancement in a new
  `static/docs.js`; with the script off both pages still read.
- **Guide:** added a twelve-term **glossary** (AIS, SAR, dark vessel, detection, declaration,
  match tolerance, azimuth shift, interpolated/nearest, MMSI, structure, tile/overlap, CFAR).
- **How it works:** added **The detector**, which the site did not mention at all — the CNN
  against the CFAR baseline on LS-SSDD, with the inshore column reported alongside the
  offshore one and the hard-negative result that went the wrong way. Corrected "What it can't
  yet claim", which still listed a trained detector as a future step; the honest remaining gap
  is that the trained detector and a real scene have not yet met.
- **Tests:** 189 → 192. The three new ones pin the fixes that would otherwise regress in
  silence: that every contents entry points at a heading that exists and both pages load
  `docs.js`, that `#overlay` carries no `aria-hidden`, and that the stylesheet never goes back
  to `overflow-x: hidden`.
- **Committed as nine steps**, each one green on its own, with each bug fix carrying its own
  regression test: the sticky fix (`7792144`), the overlay (`8e1b674`), contrast (`bac0032`),
  the counts landmark (`3f325fd`), the layout rebalance (`a2f6d52`), the glossary (`ffea8a8`),
  the detector page (`c41b354`), the document-page rebuild (`c498549`), and this entry.
- **Verified:** `ruff check` clean, `pytest -q` → 192 passed, `darkvessel render` → every file
  the publish workflow asserts plus `docs.js`, and axe-core clean on all three pages.

### 2026-09-21 — repository tidied and the working documents moved into `docs/`

- **Moved four documents into `docs/`** as tracked renames: `DEPLOYMENT.md` →
  `docs/deployment.md`, `Final_goal.md` → `docs/final-goal.md`, `requirements.md` and
  `progress.md` (this file) likewise. `README.md`, `LICENSE` and `models/README.md` stayed
  where they were. Every cross-reference was updated — the links inside these four, the
  pointer in `README.md`, two mentions in `models/README.md` and one comment in
  `detect/cfar.py`. `grep` for the old names now returns nothing.
- **Added a `docs/` line and a document index to the README's repository layout**, so the
  moved files are still findable from the entry point.
- **Corrected the README's test count**: it claimed 185, the suite is 189. The health check
  above already had the right number.
- **Deleted `data/lsssdd/LS-SSDD-v1.0-OPEN.zip` (8.3 GB).** It was fully extracted alongside
  itself — 9,000 sub-images and 9,000 annotations verified present before removing it. The
  download command is in `models/README.md` if it is ever needed again.
- **Deleted regenerable clutter:** `.pytest_cache/`, `.ruff_cache/`, `src/darkvessel.egg-info/`,
  `outputs/`, three `.DS_Store` files, and `site/` — which was a *stale* build from before
  `b2e8a20`, missing `guide.html`, `about.html` and `data/manifest.json`. All of it is
  git-ignored. `site/` and `outputs/` were rebuilt from scratch afterwards.
- **Nothing in `src/` changed** beyond the one comment. Checked first: every module under
  `src/darkvessel/` is imported or tested somewhere, and every log in `models/logs/` is cited
  by `models/README.md`, so there was no dead code or orphaned artifact to remove.
- **Verified:** `pip install -e ".[dev]"`, `ruff check src/ tests/` → clean, `pytest -q` →
  189 passed. `darkvessel synthesise` then `darkvessel run` → the exact line CI greps for
  (`4 matched, 1 dark, 0 at a fixed structure`). `darkvessel render --out site` → all six
  files the publish workflow asserts, `run.json` total 5, and every run named by the manifest
  present (17,600 control positions, 1,686 distinct runs).
- **Committed as five steps** so the move stays reviewable on its own: the renames
  (`1de6382`), the reference updates (`92c6550`), the README index (`7a54acd`), the test
  count (`b763a86`), and this entry. Every one leaves the suite green — the only change
  under `src/` is a comment.
- **Left undone:** nothing from this session; the next step is still the real-scene run.

### 2026-09-20 — progress log reconciled with the code

- Re-ran the health check: `pytest -q` → 189 passed, `ruff check src/ tests/` → clean.
- Corrected the recorded test counts, which had drifted: 185 → 189 locally, and 180 → 184 for
  the no-torch run CI does (5 torch tests still skip there).
- Corrected the Git note. It still said the 2026-09-18 detector / radar / share session was
  uncommitted at `b2e8a20`; that work was published in `97abc44` and the tree is clean.
- Added `.DS_Store` to `.gitignore` — Finder had left them in the root and in `src/`.
- Nothing in `src/` changed; this entry is bookkeeping only.

### 2026-09-18 — trained detector, radar scene, share fix

- **Share button** did nothing in embedded browsers (editor previews, iframes): the clipboard
  API is refused there and the `window.prompt` fallback is suppressed. Share now opens a panel
  with the link selected, a Copy button (clipboard API, then `execCommand`), and a list of what
  the link restores. Verified in a normal tab and in an iframe.
- **Radar display** for the scene (default; `SAR image` switches back): phosphor-tinted
  imagery, range rings, bearings, a CSS-driven sweep with per-blip afterglow timed to the
  sweep, course-oriented hull glyphs for matched vessels, 20 s speed vectors, pulsing diamonds
  for dark contacts. Payload declarations now carry `course_deg` / `speed_kn` (absent, not
  zero, for a lone report). Reduced-motion disables the animation.
- **Data:** LS-SSDD-v1.0 from the authors' Google Drive link (Apache-2.0), 8.3 GB zip in
  `data/lsssdd/` (git-ignored). `detect/lsssdd.py` audits every label and split, counting each
  rule; found one ship boxed twice in test (merged), 273 mostly-no-data sub-images, 82%
  ship-free training images, a train/test ship-size shift. Splits clean, totals match paper.
- **Detectors:** `detect/cfar.py` (CA-CFAR baseline), `detect/cnn.py` + `detect/train.py`
  (centre-heatmap U-Net), `detect/evaluate.py` + `detect/benchmark.py` (threshold on val scenes
  05/10, report on test). `detector: stub|cfar|cnn` in the run config; new CLI commands
  `audit-data`, `train`, `evaluate`. Weights in `models/ship_centrenet.pt` (1.6 MB), meant to be
  committed (the `models/*.pt` ignore rule was removed) but **not committed yet**.
- **Results (test):** offshore F1 CNN 0.866 vs CFAR 0.689; inshore 0.193 vs 0.279; overall AP
  0.626 vs 0.563. CNN 46 ms / 800 px image on the M5 GPU, CFAR 111 ms.
- **Anomalies in training, diagnosed with seeded runs on identical weights** (logs in
  `models/logs/`): validation AP swinging 0.27–0.70 was *not* BatchNorm statistics (two
  recalibration experiments rejected it) but weight oscillation — fixed by evaluating and
  shipping an EMA of the weights. Hard-negative fine-tuning on training-scene false alarms made
  inshore test *worse* (AP 0.117 -> 0.100); not shipped.
- **CFAR default threshold** was 5.0σ while its calibrated operating point is 6.0σ, so a config
  leaving `detector_threshold` out did not get the calibrated baseline as documented. Default
  is now 6.0 (found while updating this file on 2026-09-19).
- **Phone layout:** neighbouring radar labels ("2 · 5.8 kn", "3 · 15.6 kn") collided when the
  stage is under 520 px wide; the speed is now hidden there (still in the tooltip and inspector).
- Verified: 185 tests (shipped-weights smoke test included), ruff clean, quick start numbers
  unchanged, CNN consistent under pipeline tiling (512 px tiles ~2x faster than 128), viewer
  browser checks pass in static and live modes, radar checked at phone width and with reduced
  motion.
- **Left undone:** nothing committed or pushed — the live site still has the old Share button.
- **Next:** land/coastline mask before detection (the inshore gap); check the CNN on one
  calibrated real Sentinel-1 scene (it was trained on 8-bit LS-SSDD chips).

### 2026-09-18 — viewer fixes, interactive static site, new pages, caching

- **Bugs fixed (all reproduced in Chrome first):** scene markers could not be clicked (unfilled
  boxes were only hit-testable on their 2 px outline; the halo is now the hit area); every
  control on the published static site was disabled; one failed live request left the
  viewer permanently unable to refresh; registering a structure by clicking a dark
  detection selected it instead; a filter on a status that dropped to zero stuck behind a
  disabled card; Reset fired duplicate requests and left placing mode and the URL behind.
- **Static site is now interactive:** `darkvessel render` bakes every control position
  (`web/bake.py`, 17,600 positions -> 80 distinct runs, ~1 s) behind `data/manifest.json`.
  The register is applied in the browser, which is exact (dark -> structure, after matching).
- **AIS is now cleaned before matching** in the viewer and in `darkvessel run`. Before this,
  cleaning was only *reported*: the viewer's "a dark claim rests on this" was not true.
  Synthetic numbers unchanged.
- **Efficiency:** `pipeline.run` split into `detect_scene` + `fuse`; the viewer holds a
  `Viewer` that caches detection and crops per threshold and declared positions per max gap;
  the WGS84 transformer is built once; the API gzips responses (4.7 KB -> 1.1 KB).
- **New:** Guide page (tour with five one-click scenario links, statuses, shortcuts, FAQ),
  How it works page, site nav, Share (URL restores controls, register and selection), CSV /
  JSON export, inspector previous / next / close.
- `docs/pipeline.svg` moved to `src/darkvessel/web/static/pipeline.svg` so the site can use it.
- Verified: 163 tests (new ones mutation-checked), ruff clean, a Playwright script clicking
  every control in static and live modes, the five guide links, error recovery, and
  phone-width layout with no horizontal scroll.
- **Found, not changed (science semantics; needs a decision):** `fusion/interpolate.py`
  interpolates between bracketing reports however far apart they are (max gap is only
  applied to lone reports); and when the acquisition precedes a track, velocity is taken
  from the *last* two reports of the track rather than the first two.

### 2026-09-18 — published to GitHub

- Rewrote history into a single commit authored by Priyansh Koli, then pushed to
  https://github.com/priyansh-koli/DarkVessel.
- First push was rejected: the saved Personal Access Token lacked the `workflow` scope needed
  to push `.github/workflows/`. Fixed with a new classic token (`repo` + `workflow`).
- Enabled Pages (Settings → Pages → Source: GitHub Actions) and re-ran the publish workflow.
- Verified: `ci` passed; `publish viewer` passed on its second attempt; the live site serves
  `index.html`, `assets/scene.png` and `data/run.json` (counts 5 total / 4 matched / 1 dark).
  `/api/health` returns 404 HTML there, so the viewer correctly stays in static mode.

### 2026-09-18 — continuity files, licence, pre-push cleanup

- Added `docs/progress.md` (this file), `docs/requirements.md` and `docs/final-goal.md` so the project can be
  resumed from a fresh session with no prior context.
- Added a `LICENSE` (MIT, © Priyansh Koli) and set the author in `pyproject.toml`.
- Reworded the README, docstrings, tests, CI comment and pipeline diagram to describe the
  design decisions directly. Removed `HANDOFF.md`; its useful notes now live in "Things worth
  knowing" above.
- Re-verified: 150 tests pass, ruff clean, quick start prints the expected numbers.

### 2026-09-17 — health probe fix

- The viewer now requires a JSON response from `/api/health` before switching to live mode,
  so a static host that answers every path with an HTML page no longer tricks the frontend
  into live mode.

### 2026-09-23 — AIS reception, the declaration side, and a faster matcher

- **`fusion/reception.py` (new).** Estimates, per place, whether the archive would have placed
  a transmitting vessel at the acquisition instant, from the gaps between the reports of the
  vessels it already holds: `sum(min(g, 2w)) / sum(g)` over the gaps observed nearby, where
  `w` is the pipeline's own `max_gap`. Evidence is pooled over a cell and its eight neighbours;
  a neighbourhood under `min_intervals` gets no estimate rather than a confident one. Below a
  configured floor, a `dark` detection becomes the new `shadowed` status. A *missing* estimate
  never shadows anything. `Coverage` deliberately mirrors `fusion.register.Register`
  (`from_archive` / `mark`, plus a `without_coverage` no-op), and `reception_p`,
  `reception_basis` and `reception_intervals` are on every row whether or not the stage ran.
- **`fusion/declarations.py` (new).** The other half of the assignment, which the pipeline used
  to discard: declarations no detection explains, as `explained` / `undetected` /
  `below_detectable` / `outside_scene`. `Agreement` turns both sides into a two-sensor
  confusion matrix and an *apparent recall* — a label-free estimate of the detector's recall on
  the scene, reported with its biases stated. `classify` now returns a two-sided `Match` and
  `pipeline.fuse` a `Fusion`; `darkvessel run` writes two GeoPackage layers.
- **`fusion/match.py` rewritten for scale.** The distance matrix was built with a Python double
  loop of shapely `.distance` calls and solved densely, which an archive-wide run cannot
  afford. Now: a k-d tree builds the within-tolerance feasibility graph, `connected_components`
  splits it, and each block is solved alone — provably the same answer, because no feasible edge
  crosses a component and the infeasible penalty is flat. `tests/test_match.py` checks that
  equivalence against the dense solution over 25 random clustered point clouds. Isolated
  detections and declarations never enter a cost matrix at all. Column writes in `classify` are
  one vectorised pass per column rather than a `.loc` per matched pair, and
  `Geometry.displacements` moves a whole archive in one call.
- **Fixture, config, CLI, viewer, docs.** Four AIS-only vessels added to the synthetic fixture,
  one per new branch (see Things worth knowing). New settings: `reception_cell_m`,
  `reception_floor`, `reception_min_intervals`, `smallest_detectable_m`. The viewer gained a
  reception-floor slider, a shadow overlay, a reception column and inspector section, an
  "other side" panel, an AIS CSV export, and the `shadowed` status throughout. The pipeline
  diagram, README, Guide and How it works pages were all updated.
- **Verified:** 241 tests pass, ruff clean, and both viewer modes were driven in headless
  Chromium with no console errors — counts, the floor slider, the overlay, the inspector and
  the exports all checked live and against the static bundle.
- **Two bugs found while building, both only on paths the unit tests did not cover.** First,
  `pd.DatetimeIndex(...).asi8` returns the column's *own* resolution, and a GeoPackage hands
  back milliseconds where the code assumed nanoseconds — so every gap came out a thousand times
  too short, read as an implausible speed, and the whole reception model was silently thrown
  away. It only showed through `darkvessel run`, never in-process. Fixed by pinning
  `as_unit("ns")` before reading the integers. Second, the static bundle looked up its reception
  model by a key Python wrote as `"10.0"` and JavaScript asked for as `"10"`, so static mode
  loaded no model at all; the browser check caught it, and the index is now positional like the
  run index. A third, smaller wrongness was fixed on the way: reception files were being shared
  between max gaps that shared a *run*, which is not the same thing, so a viewer could describe
  the wrong search.
- **Left undone:** the bracket-width question in `fusion/interpolate.py` (now item 0 in Next
  steps) — `position_span_s` makes it visible but nothing acts on it. Reception is validated
  only against the synthetic fixture, and the 0.5 floor is a placeholder.

---

### 2026-09-16 — core package, viewer, deployment

- Built the `src/darkvessel/` package: data (scene I/O, tiling, provenance, AIS cleaning,
  DMA reader, synthetic fixture), detect (protocol, stand-in, tile-wise inference,
  pixel → ground), fusion (interpolation, azimuth correction, optimal matching, structure
  register), embed (protocol, crops, union-find recurrence), context (schema, stub), config
  and CLI.
- Matching uses optimal (Hungarian) assignment rather than greedy; structure clustering uses
  union-find so it is independent of row order; the stand-in detector collapses flat bright
  plateaus into one detection.
- Set the Python floor to 3.9 (the dev machine has only 3.9.6).
- Wrote tests for every module (150 total) and mutation-checked them.
- Built the viewer (`darkvessel serve`, `darkvessel render`), the Dockerfile,
  `docs/deployment.md`, CI and the GitHub Pages workflow. Fixed three frontend bugs found in a real
  browser: the loading overlay never cleared (CSS `display` overriding `hidden`), marker
  labels frozen at one size, and a rendering artifact from the toggle switch's hidden input.

---

### Entry template

```markdown
### YYYY-MM-DD — short title

- What changed (files/modules, commit hashes).
- How it was verified (tests, commands run, outputs checked).
- What broke or surprised you, and what was done about it.
- What was left unfinished.
```
