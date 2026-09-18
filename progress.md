# darkvessel — progress log

The running record of what has been done on this project, one entry per work session, newest
first. Read this, [requirements.md](requirements.md) and [Final_goal.md](Final_goal.md) at the
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
pytest -q                           # expect: 150 passed
ruff check src/ tests/              # expect: All checks passed!
darkvessel synthesise --out data/synthetic
darkvessel run --config configs/pipeline.yaml
# expect: 5 detections in EPSG:25832 -> outputs/detections.gpkg
#           4 matched, 1 dark, 0 at a fixed structure, at a tolerance of 200 m
```

---

## Current state (as of 2026-09-18)

| Area | Status |
|---|---|
| Tier 1 — core pipeline (tiling, pixel→ground, AIS interpolation, azimuth correction, optimal matching, structure register, recurrence clustering, CLI) | **Done**, tested |
| AIS ingestion — DMA CSV reader and cleaning rules with per-rule counts | **Done**, tested against the format; not yet run on a real DMA day file |
| Viewer — FastAPI live app + static bundle | **Done**, checked in a real browser at desktop and phone widths |
| Deployment — Dockerfile, CI, GitHub Pages workflow | **Written**; the Dockerfile has never been built (no docker on the dev machine); workflows have never run (no remote) |
| Detector | Deterministic stand-in (`BrightPixelDetector`) only — no trained model |
| Tier 2 — Earth Engine export and contextual variables | **Not started** (`context/gee_layers.py` is a no-op) |
| Tier 3 — CNN detector on LS-SSDD, contrastive embeddings | **Not started** |
| Analysis — archive-wide concentration analysis, static map | **Not started** |
| Real data | **None yet** — everything runs on the synthetic fixture |

- Tests: 150 passing. Lint: clean.
- Git: local repo on `main`, **no remote configured yet**.
- Python: the dev machine has only system Python 3.9.6, so the project targets `>=3.9`.

## Next steps

In rough priority order — see [Final_goal.md](Final_goal.md) for why.

1. Push to a GitHub remote so CI and the Pages workflow actually run for the first time.
2. Run the pipeline on one real Sentinel-1 scene and the matching DMA AIS day file.
3. Replace the stand-in detector with a real one (CFAR baseline first, then a CNN trained on
   LS-SSDD).
4. Build the Earth Engine contextual layers (distance to shore, depth, fishing effort, EEZ).
5. Archive-wide run and the concentration analysis and map.

## Open questions

- The azimuth-shift model in `fusion/azimuth.py` and the cleaning thresholds in `data/ais.py`
  are only validated against the synthetic fixture. Check both against real Sentinel-1 / DMA
  data and the SAR literature before trusting real-world results.
- The study area (EPSG:25832 → Danish waters) and the time range for the real archive have
  not been fixed yet.

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
- The test suite was mutation-checked: breaking tile stepping, the max-gap drop, the dark-only
  register filter, the incidence term, plateau collapsing, tile ownership, context absent
  values, crop zero-padding or the duplicate-AIS rule each turns it red.

---

## Session log

### 2026-09-18 — continuity files, licence, pre-push cleanup

- Added `progress.md` (this file), `requirements.md` and `Final_goal.md` so the project can be
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
  `DEPLOYMENT.md`, CI and the GitHub Pages workflow. Fixed three frontend bugs found in a real
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
