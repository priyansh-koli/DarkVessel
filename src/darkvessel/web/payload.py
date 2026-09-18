"""One pipeline run, shaped for a viewer.

The frontend draws in pixel space, so every ground coordinate is projected back through the
scene transform here rather than duplicating affine maths in JavaScript. Declarations carry
*both* their raw position and where the azimuth correction moved them to, because the gap
between those two points is the thing an analyst most needs to see: it is the difference
between a matched vessel and a reported dark one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import Point

from darkvessel.data.ais import clean
from darkvessel.data.scene import Scene
from darkvessel.data.tiling import Tiling
from darkvessel.detect.stub import BrightPixelDetector
from darkvessel.embed.structures import SAME_POSITION_M
from darkvessel.fusion.azimuth import Geometry
from darkvessel.fusion.interpolate import positions_at
from darkvessel.fusion.register import Register
from darkvessel.pipeline import run as run_pipeline
from darkvessel.render import crop_data_uris

# Columns the inspector shows verbatim. Anything the pipeline adds later appears automatically.
_HIDDEN = {"geometry", "embedding"}


@dataclass(frozen=True)
class RunRequest:
    """Everything the viewer is allowed to vary between runs."""

    tolerance_m: float = 200.0
    max_gap_minutes: float = 10.0
    detector_threshold: float = 0.5
    tile_px: int = 128
    overlap_px: int = 32
    apply_azimuth: bool = True
    register_xy: list[tuple[float, float]] = field(default_factory=list)
    register_tolerance_m: float = SAME_POSITION_M


def build(scene: Scene, ais: gpd.GeoDataFrame | None, request: RunRequest) -> dict[str, Any]:
    """Run the chain under `request` and return everything the viewer needs to draw it."""
    max_gap = timedelta(minutes=request.max_gap_minutes)
    geometry = (
        Geometry(heading_deg=scene.heading_deg, incidence_deg=scene.incidence_deg)
        if request.apply_azimuth
        else None
    )
    register = _register(request)

    detections = run_pipeline(
        scene=scene,
        ais=ais,
        detector=BrightPixelDetector(threshold=request.detector_threshold),
        tiling=Tiling(tile_px=request.tile_px, overlap_px=request.overlap_px),
        tolerance_m=request.tolerance_m,
        max_gap=max_gap,
        geometry=geometry,
        structures=register,
    )

    crops = crop_data_uris(scene.image, detections[["row", "col"]])
    return {
        "scene": _scene_summary(scene),
        "config": _config_summary(request, register),
        "counts": _counts(detections),
        "detections": _detections(detections, crops),
        "declarations": _declarations(scene, ais, max_gap, geometry, detections),
        "ais": _ais_summary(ais),
        "register": [
            {"x": x, "y": y, **_to_pixel(scene, x, y)} for x, y in request.register_xy
        ],
    }


def _register(request: RunRequest) -> Register | None:
    if not request.register_xy:
        return None
    positions = pd.DataFrame(
        {
            "x": [x for x, _ in request.register_xy],
            "y": [y for _, y in request.register_xy],
            "acquisitions": [2] * len(request.register_xy),
            "crops": [2] * len(request.register_xy),
        }
    )
    return Register(positions=positions, tolerance_m=request.register_tolerance_m)


def _scene_summary(scene: Scene) -> dict[str, Any]:
    height, width = scene.image.shape
    return {
        "id": scene.id,
        "width": int(width),
        "height": int(height),
        "crs": str(scene.crs),
        "acquired_at": scene.acquired_at.isoformat(),
        "heading_deg": scene.heading_deg,
        "incidence_deg": scene.incidence_deg,
        "pixel_size_m": abs(scene.transform.a),
        # (a, b, c, d, e, f): ground x = a*col + b*row + c, ground y = d*col + e*row + f.
        # Shipped so the viewer can turn a click into a coordinate without its own affine.
        "transform": [float(v) for v in list(scene.transform)[:6]],
    }


def _config_summary(request: RunRequest, register: Register | None) -> dict[str, Any]:
    return {
        "tolerance_m": request.tolerance_m,
        "max_gap_minutes": request.max_gap_minutes,
        "detector_threshold": request.detector_threshold,
        "tile_px": request.tile_px,
        "overlap_px": request.overlap_px,
        "apply_azimuth": request.apply_azimuth,
        "register_positions": 0 if register is None else len(register.positions),
        "register_tolerance_m": request.register_tolerance_m,
    }


def _counts(detections: gpd.GeoDataFrame) -> dict[str, int]:
    counts = detections["status"].value_counts()
    return {
        "total": int(len(detections)),
        "matched": int(counts.get("matched", 0)),
        "dark": int(counts.get("dark", 0)),
        "structure": int(counts.get("structure", 0)),
        "unsearched": int(counts.get("unsearched", 0)),
    }


def _detections(detections: gpd.GeoDataFrame, crops: list[str]) -> list[dict[str, Any]]:
    rows = []
    for position, (_, row) in enumerate(detections.iterrows()):
        record = {name: _plain(value) for name, value in row.items() if name not in _HIDDEN}
        record["index"] = position
        # +0.5 puts the marker on the pixel's centre, which is where `detect.geo` placed it.
        record["px"] = float(row["col"]) + 0.5
        record["py"] = float(row["row"]) + 0.5
        record["crop"] = crops[position] if position < len(crops) else None
        rows.append(record)
    return rows


def _declarations(
    scene: Scene,
    ais: gpd.GeoDataFrame | None,
    max_gap: timedelta,
    geometry: Geometry | None,
    detections: gpd.GeoDataFrame,
) -> list[dict[str, Any]]:
    """Each vessel's declared position at the acquisition instant, raw and radar-corrected."""
    if ais is None or ais.empty:
        return []

    if ais.crs != scene.crs:
        ais = ais.to_crs(scene.crs)
    declared = positions_at(ais, scene.acquired_at, max_gap)
    if declared.empty:
        return []

    matched_by_mmsi = {
        str(row["mmsi"]): position
        for position, (_, row) in enumerate(detections.iterrows())
        if row["status"] == "matched" and not pd.isna(row["mmsi"])
    }
    latitude = _latitude_of(scene)

    rows = []
    for _, row in declared.iterrows():
        raw = row.geometry
        east, north = 0.0, 0.0
        if geometry is not None and np.isfinite(row["velocity_east_ms"]) and np.isfinite(
            row["velocity_north_ms"]
        ):
            east, north = geometry.displacement(
                row["velocity_east_ms"], row["velocity_north_ms"], latitude
            )
        drawn = Point(raw.x + east, raw.y + north)
        rows.append(
            {
                "mmsi": str(row["mmsi"]),
                "length_m": _plain(row["length_m"]),
                "position_basis": _plain(row["position_basis"]),
                "position_age_s": _plain(row["position_age_s"]),
                "azimuth_shift_m": float(np.hypot(east, north)),
                "matched_index": matched_by_mmsi.get(str(row["mmsi"])),
                "raw": {"x": raw.x, "y": raw.y, **_to_pixel(scene, raw.x, raw.y)},
                "drawn": {"x": drawn.x, "y": drawn.y, **_to_pixel(scene, drawn.x, drawn.y)},
            }
        )
    return rows


def _ais_summary(ais: gpd.GeoDataFrame | None) -> dict[str, Any] | None:
    """The cleaning report, which is what `declarations_searched` on a dark row rests on."""
    if ais is None:
        return None
    cleaned, report = clean(ais)
    return {
        "rows_in": report.starting_rows,
        "rows_kept": len(cleaned),
        "removed": dict(report.removed),
        "line": report.line(),
        "vessels": int(cleaned["mmsi"].nunique()) if not cleaned.empty else 0,
    }


def _latitude_of(scene: Scene) -> float:
    height, width = scene.image.shape
    x, y = scene.transform * (width / 2, height / 2)
    _, latitude = Transformer.from_crs(scene.crs, "EPSG:4326", always_xy=True).transform(x, y)
    return float(latitude)


def _to_pixel(scene: Scene, x: float, y: float) -> dict[str, float]:
    col, row = ~scene.transform * (x, y)
    return {"px": float(col), "py": float(row)}


def _plain(value: Any) -> Any:
    """Convert numpy/pandas scalars into something `json` will accept."""
    if value is None or value is pd.NaT:
        return None
    if not isinstance(value, (list, tuple)) and pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
