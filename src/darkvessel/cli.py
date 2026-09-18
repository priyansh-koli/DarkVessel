"""The CLI: `synthesise` builds the no-network fixture, `run` runs the chain over one scene."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import geopandas as gpd

from darkvessel import config as config_module
from darkvessel.data.scene import read_scene
from darkvessel.data.synthetic import write_synthetic
from darkvessel.data.tiling import Tiling
from darkvessel.detect.stub import BrightPixelDetector
from darkvessel.fusion.match import DARK, MATCHED
from darkvessel.fusion.register import STRUCTURE
from darkvessel.pipeline import run as run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="darkvessel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthesise = subparsers.add_parser(
        "synthesise", help="write the no-network synthetic fixture"
    )
    synthesise.add_argument("--out", required=True, help="directory to write the fixture to")

    run_cmd = subparsers.add_parser("run", help="run the pipeline over one scene")
    run_cmd.add_argument("--config", required=True, help="path to a run configuration YAML file")

    serve = subparsers.add_parser("serve", help="serve the viewer, re-running the chain live")
    serve.add_argument("--config", default="configs/pipeline.yaml", help="run configuration YAML")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="restart on source changes")

    render = subparsers.add_parser(
        "render", help="pre-render a static viewer bundle: no Python needed to host it"
    )
    render.add_argument("--config", default="configs/pipeline.yaml", help="run configuration YAML")
    render.add_argument("--out", required=True, help="directory to write the bundle to")

    args = parser.parse_args(argv)

    if args.command == "synthesise":
        return _synthesise(Path(args.out))
    if args.command == "serve":
        return _serve(args)
    if args.command == "render":
        return _render(Path(args.config), Path(args.out))
    return _run(Path(args.config))


def _synthesise(out_dir: Path) -> int:
    write_synthetic(out_dir)
    print(f"synthetic fixture written to {out_dir}")
    return 0


def _run(config_path: Path) -> int:
    cfg = config_module.load(config_path)

    scene = read_scene(cfg.scene_dir)
    ais = gpd.read_file(cfg.ais_path) if cfg.ais_path else None
    tiling = Tiling(tile_px=cfg.tile_px, overlap_px=cfg.overlap_px)
    detector = BrightPixelDetector(threshold=cfg.detector_threshold)

    detections = run_pipeline(
        scene=scene,
        ais=ais,
        detector=detector,
        tiling=tiling,
        tolerance_m=cfg.tolerance_m,
        max_gap=cfg.max_gap,
        geometry=cfg.geometry,
    )

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    detections.to_file(cfg.output_path, driver="GPKG")

    counts = detections["status"].value_counts()
    matched = int(counts.get(MATCHED, 0))
    dark = int(counts.get(DARK, 0))
    structure = int(counts.get(STRUCTURE, 0))

    print(f"{len(detections)} detections in {detections.crs} -> {cfg.output_path}")
    print(
        f"  {matched} matched, {dark} dark, {structure} at a fixed structure, "
        f"at a tolerance of {cfg.tolerance_m:g} m"
    )
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("the viewer needs the web extra: pip install -e '.[web]'", file=sys.stderr)
        return 1

    os.environ["DARKVESSEL_CONFIG"] = str(args.config)
    print(f"darkvessel viewer on http://{args.host}:{args.port}  (config: {args.config})")
    uvicorn.run(
        "darkvessel.web.app:app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


def _render(config_path: Path, out_dir: Path) -> int:
    """Bake the viewer plus one run into a folder any static host can serve."""
    from darkvessel.render import scene_png
    from darkvessel.web.app import STATIC_DIR
    from darkvessel.web.payload import RunRequest, build

    cfg = config_module.load(config_path)
    scene = read_scene(cfg.scene_dir)
    ais = gpd.read_file(cfg.ais_path) if cfg.ais_path else None

    payload = build(
        scene,
        ais,
        RunRequest(
            tolerance_m=cfg.tolerance_m,
            max_gap_minutes=cfg.max_gap.total_seconds() / 60,
            detector_threshold=cfg.detector_threshold,
            tile_px=cfg.tile_px,
            overlap_px=cfg.overlap_px,
            apply_azimuth=cfg.geometry is not None,
        ),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "assets").mkdir(exist_ok=True)
    (out_dir / "data").mkdir(exist_ok=True)

    for name in ("index.html", "styles.css", "app.js"):
        shutil.copyfile(STATIC_DIR / name, out_dir / name)
    (out_dir / "assets" / "scene.png").write_bytes(scene_png(scene.image))
    (out_dir / "data" / "run.json").write_text(json.dumps(payload))

    counts = payload["counts"]
    print(f"static viewer written to {out_dir}")
    print(
        f"  {counts['total']} detections · {counts['matched']} matched, {counts['dark']} dark, "
        f"{counts['structure']} at a fixed structure"
    )
    print(f"  serve it with any static host, e.g. python -m http.server -d {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
