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
from darkvessel.data.ais import clean
from darkvessel.data.scene import read_scene
from darkvessel.data.synthetic import write_synthetic
from darkvessel.data.tiling import Tiling
from darkvessel.detect.factory import make_detector
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

    audit = subparsers.add_parser(
        "audit-data", help="audit LS-SSDD labels and splits, counting every anomaly found"
    )
    audit.add_argument("--root", default=_LSSSDD, help="the unzipped LS-SSDD-v1.0-OPEN folder")

    train = subparsers.add_parser("train", help="train the CNN ship detector on LS-SSDD")
    train.add_argument("--root", default=_LSSSDD, help="the unzipped LS-SSDD-v1.0-OPEN folder")
    train.add_argument("--out", default="models/ship_centrenet.pt", help="where to write weights")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--seed", type=int, default=0)

    bench = subparsers.add_parser(
        "evaluate", help="benchmark a detector on LS-SSDD (threshold from val, reported on test)"
    )
    bench.add_argument("--root", default=_LSSSDD, help="the unzipped LS-SSDD-v1.0-OPEN folder")
    bench.add_argument("--detector", choices=["cfar", "cnn"], required=True)
    bench.add_argument("--weights", default="models/ship_centrenet.pt", help="cnn weights")
    bench.add_argument("--out", help="write the report as JSON here")

    args = parser.parse_args(argv)

    if args.command == "audit-data":
        return _audit_data(Path(args.root))
    if args.command == "train":
        return _train(args)
    if args.command == "evaluate":
        return _evaluate(args)

    if args.command == "synthesise":
        return _synthesise(Path(args.out))
    if args.command == "serve":
        return _serve(args)
    if args.command == "render":
        return _render(Path(args.config), Path(args.out))
    return _run(Path(args.config))


_LSSSDD = "data/lsssdd/LS-SSDD-v1.0-OPEN"


def _audit_data(root: Path) -> int:
    from darkvessel.detect import lsssdd

    print("splits")
    for line in lsssdd.audit_splits(root).lines()[1:] or ["  no problems"]:
        print(line)
    for split in ("train", "val", "test"):
        audit = lsssdd.Audit()
        samples = lsssdd.load(root, lsssdd.split_names(root, split), audit)
        ships = sum(len(s.boxes) for s in samples)
        empty = sum(1 for s in samples if not len(s.boxes))
        print(f"{split}: {len(samples)} sub-images, {ships} ships, {empty} with no ships")
        for line in audit.lines()[1:]:
            print(line)
    return 0


def _train(args: argparse.Namespace) -> int:
    from darkvessel.detect.train import TrainConfig, train

    summary = train(
        Path(args.root), Path(args.out), TrainConfig(epochs=args.epochs, seed=args.seed)
    )
    print(f"best epoch {summary['best_epoch']}: validation AP {summary['val_ap']}")
    print(f"weights -> {args.out}, history and anomalies -> {Path(args.out).with_suffix('.json')}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from darkvessel.detect.benchmark import benchmark

    if args.detector == "cfar":
        from darkvessel.detect.cfar import CFARDetector

        detector = CFARDetector()
        report = benchmark(lambda image: detector.scored(image, floor=3.0), Path(args.root), "cfar")
    else:
        from darkvessel.detect.cnn import CNNDetector

        detector = CNNDetector(args.weights)
        report = benchmark(lambda image: detector.scored(image, floor=0.02), Path(args.root), "cnn")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


def _synthesise(out_dir: Path) -> int:
    write_synthetic(out_dir)
    print(f"synthetic fixture written to {out_dir}")
    return 0


def _run(config_path: Path) -> int:
    cfg = config_module.load(config_path)

    scene = read_scene(cfg.scene_dir)
    # Cleaned before matching, as in the viewer: a dark claim is only as good as the archive
    # searched, and the viewer's cleaning report must describe the same archive this run used.
    ais = clean(gpd.read_file(cfg.ais_path))[0] if cfg.ais_path else None
    tiling = Tiling(tile_px=cfg.tile_px, overlap_px=cfg.overlap_px)
    detector = make_detector(cfg.detector, cfg.detector_threshold, cfg.detector_weights)

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
    from darkvessel.web.bake import bake
    from darkvessel.web.payload import RunRequest, Viewer

    cfg = config_module.load(config_path)
    scene = read_scene(cfg.scene_dir)
    ais = gpd.read_file(cfg.ais_path) if cfg.ais_path else None

    viewer = Viewer(scene, ais)
    default = RunRequest(
        tolerance_m=cfg.tolerance_m,
        max_gap_minutes=cfg.max_gap.total_seconds() / 60,
        # The viewer re-runs the stand-in detector: its threshold slider is in brightness units.
        detector_threshold=cfg.detector_threshold if cfg.detector == "stub" else 0.5,
        tile_px=cfg.tile_px,
        overlap_px=cfg.overlap_px,
        apply_azimuth=cfg.geometry is not None,
    )
    payload = viewer.run(default)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "assets").mkdir(exist_ok=True)
    (out_dir / "data").mkdir(exist_ok=True)

    for source in STATIC_DIR.iterdir():
        if source.is_file():
            shutil.copyfile(source, out_dir / source.name)
    (out_dir / "assets" / "scene.png").write_bytes(scene_png(scene.image))
    (out_dir / "data" / "run.json").write_text(json.dumps(payload))
    manifest = bake(viewer, default, out_dir / "data")
    (out_dir / "data" / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))

    counts = payload["counts"]
    print(f"static viewer written to {out_dir}")
    print(
        f"  {counts['total']} detections · {counts['matched']} matched, {counts['dark']} dark, "
        f"{counts['structure']} at a fixed structure"
    )
    print(
        f"  {len(manifest['runs'])} control positions baked into "
        f"{manifest['distinct_runs']} distinct runs"
    )
    print(f"  serve it with any static host, e.g. python -m http.server -d {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
