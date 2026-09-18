# Deploying darkvessel

What this project needs to run, and how to host the viewer. There are two deployment shapes
and they have very different requirements — pick the one that matches what you need first,
because it decides everything else.

| | **Static bundle** | **Live app** |
|---|---|---|
| What the visitor gets | Every control position pre-computed, fully interactive | Controls that re-run the pipeline |
| Runtime on the server | None — plain files | Python 3.9+ |
| Where it can host | Any static host or CDN | Container platform or VM |
| Bundle / image size | ~800 KB (synthetic scene) | ~400 MB image |
| Cost | Free tier nearly everywhere | Small instance, from ~$0–7/mo |
| Build step needs Python | Yes (at build time only) | Yes |

If you only need to *show* the result — a demo, a report, a link in a paper — the static
bundle is the right answer and costs nothing to run. Its controls work too: every position
they can take was run at build time. Choose the live app for arbitrary parameter values (max
gap in single minutes, tolerances off the 25 m grid) or for a real scene too big to bake.

---

## 1. What the project needs to run at all

### Python

**Python 3.9 or newer.** This machine has only the macOS system Python 3.9.6, so the project
targets `>=3.9`; it runs on 3.11/3.12 unchanged and those are a better choice on a server.

### The geospatial stack

The dependencies that matter are `geopandas`, `pyogrio`, `shapely`, `pyproj` and `rasterio`.
All five ship **manylinux/macOS wheels with GDAL, GEOS and PROJ statically bundled**, so on
Linux x86-64 and Apple Silicon `pip install` needs no system GDAL and no compiler. This is the
single most important fact for hosting: it means a plain `python:3.12-slim` base image works,
and you do not need the heavyweight `osgeo/gdal` image or a conda environment.

Platforms where that is *not* true, and you will need system GDAL or conda instead:
Alpine (musl — no manylinux wheels), 32-bit ARM, and most "bring your own runtime" serverless
environments with a small deployment size cap.

### No GPU, no network, no credentials

The core chain is deliberately runnable with none of those. The optional extras change that:

| Extra | Pulls in | Needs |
|---|---|---|
| *(none)* | the full core chain | nothing — this is the default |
| `web` | fastapi, uvicorn, pillow | nothing |
| `detector` | torch, torchvision | a GPU for training; CPU is fine for inference |
| `gee` | earthengine-api | a Google Earth Engine service account |

`pip install -e ".[web]"` is all the viewer needs.

### Memory — the real constraint on a server

Memory is driven by the scene array, which is held in full:

| Scene | Pixels | As float32 |
|---|---|---|
| The synthetic fixture | 240 × 240 | 0.2 MB |
| A real Sentinel-1 IW GRD | ~16 700 × 25 000 | **~1.7 GB** |

So: the synthetic demo runs comfortably in a 256 MB container, but **a real Sentinel-1 scene
needs 4 GB of RAM or more** once you account for the tiling pass and GeoPandas overhead. Size
the instance for the scenes you actually intend to load, not for the demo.

---

## 2. Running it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

darkvessel synthesise --out data/synthetic        # write the no-network fixture
darkvessel run --config configs/pipeline.yaml     # the CLI run
darkvessel serve --config configs/pipeline.yaml   # the viewer on http://127.0.0.1:8000
```

`darkvessel serve` binds `127.0.0.1` by default, which is correct for a laptop. Use
`--host 0.0.0.0` only behind a reverse proxy or inside a container.

---

## 3. Hosting the static bundle

Build it once; the output is plain files with no Python at runtime.

```bash
darkvessel synthesise --out data/synthetic
darkvessel render --config configs/pipeline.yaml --out site
```

`render` is the "generate the images before deploying" step. It writes:

```
site/
  index.html  guide.html  about.html  styles.css  app.js  pipeline.svg
  assets/scene.png        # the SAR scene, rendered at native resolution
  data/run.json           # the default run, with every detection crop inlined as a PNG
  data/manifest.json      # every control position -> the id of its result
  data/runs/<id>.json     # each distinct result, once
```

`render` runs the pipeline for every position the tolerance, max-gap, threshold and azimuth
controls can take (17,600 on the synthetic scene). Detection depends only on the threshold
and declared positions only on the max gap, so equivalent inputs are run once, and identical
results are written once: the synthetic scene bakes in about a second into 80 result files.
Each file is the same payload the live API returns, minus the `config` block the browser
already knows. The structure register is applied in the browser, which is exact because it
only ever turns a dark row into a structure after matching. The whole bundle is ~800 KB.

A real scene with hundreds of detections makes each result file larger; if the bundle grows
too big, narrow the grids at the top of `web/bake.py`.

Check it locally before shipping:

```bash
python -m http.server -d site 8080
```

The viewer detects that no API is present, loads `data/manifest.json`, and shows a
**"Static build — every control pre-computed"** badge. Every control works; max gap steps
through its preset values instead of single minutes. A bundle without a manifest (built
before baking existed) falls back to `run.json` with the controls read-only.

### Where to put it

Any of these work with no configuration beyond pointing at `site/`:

- **GitHub Pages** — push `site/` to a `gh-pages` branch, or use the workflow below.
- **Cloudflare Pages / Netlify / Vercel** — set the output directory to `site`. Because these
  build in the cloud, set the build command to install Python deps and run `render` (see below).
- **S3 + CloudFront**, **Azure Static Web Apps**, **GCS** — sync the folder, enable static
  website hosting, set `index.html` as the index document.
- **Any nginx/Apache** — `root /path/to/site;`.

Nothing needs a server-side runtime, a database, or environment variables.

### GitHub Actions: build and publish

```yaml
name: publish viewer
on:
  push: { branches: [main] }
permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[web]"
      - run: darkvessel synthesise --out data/synthetic
      - run: darkvessel render --config configs/pipeline.yaml --out site
      - uses: actions/upload-pages-artifact@v3
        with: { path: site }
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: github-pages
    steps:
      - uses: actions/deploy-pages@v4
```

For Netlify or Cloudflare Pages the equivalent build command is:

```
pip install -e ".[web]" && darkvessel synthesise --out data/synthetic && darkvessel render --out site
```

with publish directory `site`.

---

## 4. Hosting the live app

The live app is a normal ASGI application, so anything that can run a Python web process will
host it. The entry point is:

```
darkvessel.web.app:app      # an app factory — pass --factory to uvicorn
```

and it reads its run configuration from `DARKVESSEL_CONFIG` (default `configs/pipeline.yaml`).

```bash
DARKVESSEL_CONFIG=configs/pipeline.yaml \
  uvicorn darkvessel.web.app:app --factory --host 0.0.0.0 --port 8000
```

### Container

A `Dockerfile` is included. Build and run:

```bash
docker build -t darkvessel .
docker run -p 8000:8000 darkvessel
```

The image bakes the synthetic fixture at build time so the container starts with something to
show. To serve your own scene, mount it and point the config at it:

```bash
docker run -p 8000:8000 \
  -v /data/scenes:/data/scenes:ro \
  -e DARKVESSEL_CONFIG=/data/scenes/pipeline.yaml \
  darkvessel
```

### Platform notes

| Platform | Fit | What to set |
|---|---|---|
| **Render** | Easiest. Docker or native Python | Start: `uvicorn darkvessel.web.app:app --factory --host 0.0.0.0 --port $PORT` |
| **Fly.io** | Good — `fly launch` reads the Dockerfile | Set `internal_port = 8000`; pick a 512 MB+ machine |
| **Railway** | Good | Same start command; `$PORT` is injected |
| **Google Cloud Run** | Good, scales to zero | Listen on `$PORT`; set memory to 1 GB+ (4 GB for real scenes); note cold starts reload the scene |
| **AWS App Runner / ECS Fargate** | Good | Health check path `/api/health` |
| **Azure Container Apps** | Good | Target port 8000 |
| **Plain VPS** | Most control | systemd unit below, nginx in front |
| **AWS Lambda / Vercel functions** | ✗ Avoid | The geospatial wheels blow past the unzipped size limit, and a per-request scene load is the wrong shape |

Every platform above needs the same two things: a port to listen on, and enough memory for the
scene. Nothing else — no database, no object store, no secrets for the core chain.

### systemd on a VPS

```ini
# /etc/systemd/system/darkvessel.service
[Unit]
Description=darkvessel viewer
After=network.target

[Service]
User=darkvessel
WorkingDirectory=/srv/darkvessel
Environment=DARKVESSEL_CONFIG=/srv/darkvessel/configs/pipeline.yaml
ExecStart=/srv/darkvessel/.venv/bin/uvicorn darkvessel.web.app:app \
          --factory --host 127.0.0.1 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```nginx
server {
    listen 443 ssl;
    server_name darkvessel.example.org;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### Things to know before exposing it publicly

- **`/api/run` does real work.** Every call re-runs detection over the whole scene. On the
  synthetic fixture that is milliseconds; on a real scene it is seconds of CPU per request,
  and an open endpoint is therefore a cheap way for someone to burn your CPU. Put it behind
  auth, a rate limit, or a cache keyed on the query string.
- **One worker is usually right.** Each worker loads its own copy of the scene, so `--workers 4`
  on a real scene means four times the memory. Scale with instances, not workers, unless you
  have the RAM.
- **The app serves the frontend itself** from `/`, so you do not need a separate static host.
- **There is no authentication and no write path.** Every endpoint is read-only, and no user
  input reaches the filesystem — `register` is parsed strictly as two floats. Still, treat the
  scene you load as the sensitive part: the config decides what is exposed.
- **CORS is not enabled.** The frontend is same-origin. If you split them, add
  `fastapi.middleware.cors.CORSMiddleware` explicitly rather than allowing `*`.

---

## 5. Which one should you use?

- Publishing a result, a demo, or a link for a paper → **static bundle**, on GitHub Pages.
  Free, nothing to operate, nothing to break.
- Analysts need arbitrary parameter values, not the baked grid → **live app**, on Render or
  Fly.io with 1 GB RAM for the synthetic fixture. The app caches detection per threshold and
  gzips responses, so a fusion control costs a match, not a scene read.
- Real Sentinel-1 scenes → **live app on a 4 GB+ instance**, with caching in front of
  `/api/run`, or pre-render one static bundle per scene if the parameters are fixed.
