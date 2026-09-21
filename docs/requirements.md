# darkvessel — requirements

Everything the project needs: platforms, libraries, data, accounts, compute and architecture.
Items marked **(in use)** are already in `pyproject.toml` or the code. Items marked
**(planned)** are what the later stages in [final-goal.md](final-goal.md) are expected to need;
they are recommendations, not commitments, and should be moved to "in use" as they are added.

`pyproject.toml` is the source of truth for installed dependencies. If it disagrees with this
file, update this file.

---

## 1. Platform

| Requirement | Detail |
|---|---|
| Python | **3.9+** (in use). The dev machine has only macOS system Python 3.9.6. CI tests 3.9 and 3.12; servers should run 3.12. |
| OS | macOS (dev) and Linux x86-64 (CI, Docker, hosting). Apple Silicon is fine. |
| Not supported | Alpine/musl, 32-bit ARM — no geospatial wheels, forces source builds of GDAL. |
| Virtual env | `.venv/` in the project root, created with `python3 -m venv .venv`. |
| Git | Local repo on `main`. A GitHub remote is needed for CI and Pages. |
| Docker | Needed only to build/verify the live-app image. Not installed on the dev machine. |

### Python 3.9 compatibility rules

Because the floor is 3.9, code must:

- use `from __future__ import annotations` in any module with `X | None` annotations;
- not use `zip(..., strict=True)`, `match` statements, or other 3.10+ syntax;
- pass `ruff check` with `target-version = "py39"`.

## 2. Libraries

### Core (in use) — `pip install -e .`

| Library | Used for |
|---|---|
| numpy | scene arrays, geometry |
| pandas | AIS tables, cleaning |
| scipy | `linear_sum_assignment` (optimal matching), connected components |
| rasterio, affine | reading scenes, pixel ↔ ground transforms |
| geopandas, pyogrio | GeoPackage I/O for AIS and detections |
| shapely | geometries |
| pyproj | CRS handling (EPSG:25832 by default) |
| pyyaml | `configs/pipeline.yaml` |

### Extras (in use)

| Extra | Libraries | Needed for |
|---|---|---|
| `web` | fastapi, uvicorn[standard], pillow | the viewer (`darkvessel serve`, `darkvessel render`) |
| `dev` | pytest, ruff, httpx, + the `web` libs | tests and lint |
| `detector` | torch, pillow, gdown | training and running the CNN ship detector; downloading LS-SSDD |
| `gee` | earthengine-api | contextual layers (Tier 2) — declared, not yet used |

### Planned

| Library | For |
|---|---|
| `asf_search` or `sentinelsat`/Copernicus Data Space API client | downloading Sentinel-1 GRD scenes |
| `pyroSAR` + ESA SNAP, or `sarsen` | SAR preprocessing: orbit file, calibration to σ⁰, terrain correction, land mask |
| `xarray`, `rioxarray` | handling multi-scene stacks |
| `matplotlib`, `cartopy` or `contextily` | the static concentration map |
| `scikit-learn` | clustering/evaluation in the embedding and concentration analysis |
| `geemap` (optional) | exporting and inspecting Earth Engine layers |

## 3. Data

| Dataset | Source | Status |
|---|---|---|
| Synthetic fixture | `darkvessel synthesise` | **in use** — the only data the project runs on today |
| AIS | Danish Maritime Authority daily CSVs (`web.ais.dk/aisdata/`), read by `data/dma.py` | reader built; no real file run yet |
| SAR scenes | Sentinel-1 IW GRD (VV/VH), from Copernicus Data Space Ecosystem, ASF, or Earth Engine `COPERNICUS/S1_GRD` | planned |
| Detector training data | LS-SSDD-v1.0 (Zhang et al., *Remote Sensing* 2020), 15 Sentinel-1 IW scenes, 9000 labelled 800×800 sub-images, 6015 ships. From the authors' release, https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN (Apache-2.0); the zip is the Google Drive file linked there. Unzipped to `data/lsssdd/` (git-ignored, 8.3 GB zip). | **in use**: audited, trained on, benchmarked |
| Distance to shore / coastline | GSHHG or OSM coastline; Earth Engine | planned |
| Bathymetry | GEBCO grid | planned |
| Fishing effort | Global Fishing Watch (Earth Engine or API) | planned |
| EEZ boundaries | Marine Regions (VLIZ) EEZ shapefile | planned |

Raw data and outputs are git-ignored (`data/synthetic/`, `outputs/`). Keep real data out of the
repo; record where it lives and its licence here instead.

## 4. Accounts and credentials

None are needed for anything built so far. Later stages need:

- **Copernicus Data Space** or **NASA Earthdata (ASF)** account — Sentinel-1 downloads.
- **Google Earth Engine** service account — contextual layers (`gee` extra).
- **Global Fishing Watch** API token — fishing effort, if not taken via Earth Engine.
- **GitHub** — remote, Actions, and Pages (Settings → Pages → Source: GitHub Actions).

Credentials must go in environment variables or an untracked file, never in the repo.

## 5. Compute

| Stage | Needs |
|---|---|
| Core pipeline, tests, viewer | Any laptop. No GPU, no network. |
| One real Sentinel-1 scene | ~1–2 GB RAM per scene (the scene is held in memory as float32; see `docs/deployment.md`). Tiled reading will be needed for full scenes. |
| Detector training | A GPU (local CUDA, Colab, or cloud). Inference runs on CPU. |
| Archive-wide run | Disk for the scene archive (~1 GB per GRD scene), batch execution. |
| Live viewer hosting | Small container instance, ~400 MB image. Static bundle: any static host, ~80 KB. |

## 6. Architecture

```
              SAR path                                   AIS path
  Sentinel-1 scene ─► tiling (owned cores) ─►      DMA CSV ─► cleaning (per-rule counts)
  detector (injected) ─► pixel → ground            ─► interpolate to acquisition instant
                         │                          ─► azimuth-shift correction
                         └──────────────┬───────────────┘
                                        ▼
                    optimal one-to-one assignment within tolerance
                                        ▼
                    structure register / recurrence clustering
                                        ▼
               status per detection: matched · dark · structure · unsearched
                                        ▼
            GeoPackage output ─► viewer (live FastAPI or static bundle)
                              ─► (planned) context join ─► concentration analysis ─► map
```

Design rules that must hold as the project grows:

1. **The two paths don't touch until fusion.** SAR preparation never sees AIS and vice versa.
2. **The detector is a parameter, never an import.** Anything satisfying the `Detector`
   protocol (`detect/detector.py`) drops in; the stand-in keeps the chain testable.
3. **Never extrapolate AIS past a track's end.** Record how each position was obtained in
   `position_basis`.
4. **`dark` means "searched and unexplained", not "guilty".** `unsearched` stays distinct from
   `dark`.
5. **Matching is optimal, not greedy; clustering is order-independent.** Both are pinned by
   tests.
6. **Every run is reproducible from a config file** (`configs/pipeline.yaml`).

Package layout (`src/darkvessel/`):

| Module | Role |
|---|---|
| `data/` | study area, scene I/O, tiling, provenance, AIS cleaning, DMA reader, synthetic fixture |
| `detect/` | detector protocol, stand-in, tile-wise inference, pixel → ground |
| `fusion/` | interpolation, azimuth correction, matching, structure register |
| `embed/` | embedder protocol, crops, recurrence clustering |
| `context/` | contextual variable schema, Earth Engine layers (stub) |
| `web/` | payload builder, FastAPI app, static frontend |
| `pipeline.py` | the single seam that wires it all together |
| `cli.py` | `synthesise`, `run`, `serve`, `render` |

## 7. Quality bar

- `pytest` green and `ruff check src/ tests/` clean before every commit.
- CI must keep printing the README's quick-start numbers (checked in `.github/workflows/ci.yml`).
- New geometry or matching logic gets a unit test that fails if the logic is broken.
- Results are reported as what they are: a detector that ranks candidates is not a map of
  illegal vessels.
