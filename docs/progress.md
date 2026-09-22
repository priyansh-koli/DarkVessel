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
pytest -q                           # expect: 192 passed
ruff check src/ tests/              # expect: All checks passed!
darkvessel synthesise --out data/synthetic
darkvessel run --config configs/pipeline.yaml
# expect: 5 detections in EPSG:25832 -> outputs/detections.gpkg
#           4 matched, 1 dark, 0 at a fixed structure, at a tolerance of 200 m
```

Without the `detector` extra, 5 of those tests (the ones needing torch) are skipped, and pytest
reports `187 passed, 5 skipped` — that is what CI shows, since it installs `.[dev]` only.

The detector, if the LS-SSDD data is in `data/lsssdd/` (see [models/README.md](../models/README.md)
for the download):

```bash
pip install -e ".[detector]"
darkvessel audit-data              # expect: splits clean; test 3000 sub-images, 2377 ships
darkvessel evaluate --detector cnn # expect: test_offshore F1 0.866, test F1 0.541 (~3 min on M5)
```

---

## Current state (as of 2026-09-22)

| Area | Status |
|---|---|
| Tier 1 — core pipeline (tiling, pixel→ground, AIS interpolation, azimuth correction, optimal matching, structure register, recurrence clustering, CLI) | **Done**, tested |
| AIS ingestion — DMA CSV reader and cleaning rules with per-rule counts | **Done**, tested against the format; not yet run on a real DMA day file |
| Viewer — FastAPI live app + static bundle (every control baked), radar display, share panel, Guide and How it works pages | **Done**, every control checked in Chrome in both modes, at desktop and phone widths. The radar display and share panel are **not yet published** (see Git below) |
| Deployment — CI, GitHub Pages | **Live.** CI green on 3.9 and 3.12; static viewer at https://priyansh-koli.github.io/DarkVessel/ |
| Deployment — Dockerfile (live app) | **Written, never built** (no docker on the dev machine) |
| Detector | `stub` (synthetic, the default), `cfar` baseline, `cnn` trained on LS-SSDD — see `models/README.md`. CNN beats CFAR offshore (F1 0.87 vs 0.69) and on AP, loses inshore (0.19 vs 0.28) until land is masked. Not yet run on a real georeferenced scene |
| Tier 2 — Earth Engine export and contextual variables | **Not started** (`context/gee_layers.py` is a no-op) |
| Tier 3 — CNN detector on LS-SSDD | **Done** (benchmarked); contrastive embeddings **not started** |
| Analysis — archive-wide concentration analysis, static map | **Not started** |
| Real data | **Training data only.** LS-SSDD-v1.0 (real Sentinel-1 chips, labelled) is in `data/lsssdd/` (7.8 GB zip + 2.7 GB extracted, git-ignored). The pipeline itself still runs only on the synthetic fixture: no real scene or DMA AIS file yet |

- Tests: 189 passing locally (184 + 5 skipped in CI, which has no torch). Lint: clean.
- Git: `main` tracks `origin` = https://github.com/priyansh-koli/DarkVessel (public). Every push
  to `main` runs CI and republishes the viewer. Working tree clean and in sync with `origin/main`;
  last commit is `97abc44`, which published the detector, radar display and share fix.
- Python: the dev machine has only system Python 3.9.6, so the project targets `>=3.9`.

## Next steps

In rough priority order — see [final-goal.md](final-goal.md) for why.

1. Run the pipeline on one real Sentinel-1 scene and the matching DMA AIS day file, with
   `detector: cnn` — and check the CNN on calibrated backscatter (it was trained on 8-bit chips).
2. Mask land (coastline plus a buffer) before detection: the CNN's inshore false alarms are
   the main weakness in `models/README.md`, and more training did not fix them.
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
  *last* two reports, not its first two.
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
- **Screenshots:** headless Chrome on macOS clamps windows to 500 px minimum width, so a
  414 px screenshot is a cropped 500 px render and looks broken when it isn't. Render inside a
  414 px iframe to see the real mobile layout.
- **Python 3.9 rules:** use `from __future__ import annotations` for `X | None` hints; no
  `zip(strict=True)` or other 3.10+ features.
- **Each detector reads `detector_threshold` in its own units:** brightness (stub), clutter
  standard deviations (cfar; default 6.0, calibrated on validation), heatmap score (cnn; 0.30,
  shipped in the weights). Leave it out for cfar/cnn. The viewer always runs the stub,
  whatever the config says.
- **For the CNN use `tile_px: 512`:** it is fully convolutional and gives the same detections at
  any tile size, but 128 px tiles run ~4x slower than whole images.
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
