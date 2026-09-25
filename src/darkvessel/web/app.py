"""The live viewer: FastAPI over `pipeline.run`, re-running the chain per request.

The scene and its AIS archive are read, and the archive cleaned, once at startup; detection is
cached per detector setting, so moving a fusion control re-runs only matching. Everything the
frontend needs arrives as one JSON payload — see `web.payload`.
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from darkvessel import config as config_module
from darkvessel.data.land import LandMask
from darkvessel.data.scene import read_scene
from darkvessel.embed.structures import SAME_POSITION_M
from darkvessel.render import scene_png
from darkvessel.web.payload import RunRequest, Viewer

STATIC_DIR = Path(__file__).parent / "static"


def create_app(config_path: str | Path) -> FastAPI:
    cfg = config_module.load(config_path)
    scene = read_scene(cfg.scene_dir)
    ais = gpd.read_file(cfg.ais_path) if cfg.ais_path else None
    viewer = Viewer(
        scene,
        ais,
        reception_cell_m=cfg.reception_cell_m,
        reception_min_intervals=cfg.reception_min_intervals,
        smallest_detectable_m=cfg.smallest_detectable_m,
        land=LandMask.read(cfg.land_path, scene, cfg.land_buffer_m) if cfg.land_path else None,
        geometry=cfg.geometry,
    )
    # The viewer re-runs the stand-in detector, so it uses the configured tiling or the
    # stand-in's default, never the configured detector's preference.
    tiling = cfg.tiling_for(None)
    png = scene_png(scene.image)

    app = FastAPI(title="darkvessel", docs_url="/api/docs", openapi_url="/api/openapi.json")
    # A payload is mostly repeated keys and base64 crops; it compresses several-fold.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    defaults = {
        "tolerance_m": cfg.tolerance_m,
        "max_gap_minutes": cfg.max_gap.total_seconds() / 60,
        # The viewer runs the stand-in detector, whose threshold is a brightness.
        "detector_threshold": cfg.detector_threshold if cfg.detector == "stub" else 0.5,
        "apply_azimuth": cfg.geometry is not None,
        "register_tolerance_m": SAME_POSITION_M,
        "reception_floor": cfg.reception_floor,
    }

    @app.get("/api/health")
    def health() -> dict:
        # `defaults` is what Reset returns the controls to: the run configuration, not
        # whatever the page's markup happens to say.
        return {"status": "ok", "scene": scene.id, "defaults": defaults}

    @app.get("/api/scene.png")
    def scene_image() -> Response:
        return Response(
            content=png,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/api/run")
    def run(
        tolerance_m: float = Query(cfg.tolerance_m, ge=1.0, le=5000.0),
        max_gap_minutes: float = Query(cfg.max_gap.total_seconds() / 60, ge=0.0, le=720.0),
        detector_threshold: float = Query(defaults["detector_threshold"], ge=0.0, le=1000.0),
        apply_azimuth: bool = Query(cfg.geometry is not None),
        register: list[str] = Query(default=[]),
        register_tolerance_m: float = Query(defaults["register_tolerance_m"], ge=1.0, le=5000.0),
        reception_floor: float = Query(cfg.reception_floor, ge=0.0, le=1.0),
    ) -> dict:
        return viewer.run(
            RunRequest(
                tolerance_m=tolerance_m,
                max_gap_minutes=max_gap_minutes,
                detector_threshold=detector_threshold,
                tile_px=tiling.tile_px,
                overlap_px=tiling.overlap_px,
                apply_azimuth=apply_azimuth,
                register_xy=_parse_register(register),
                register_tolerance_m=register_tolerance_m,
                reception_floor=reception_floor,
            ),
        )

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def _parse_register(values: list[str]) -> list[tuple[float, float]]:
    """Registered structure positions arrive as repeated `register=x,y` ground coordinates."""
    positions = []
    for value in values:
        parts = value.split(",")
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail=f"expected 'x,y', got {value!r}")
        try:
            positions.append((float(parts[0]), float(parts[1])))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"non-numeric position {value!r}") from None
    return positions


def app() -> FastAPI:
    """ASGI entry point for a process manager: `uvicorn darkvessel.web.app:app --factory`."""
    return create_app(os.environ.get("DARKVESSEL_CONFIG", "configs/pipeline.yaml"))
