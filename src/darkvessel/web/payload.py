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
from functools import lru_cache
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Point

from darkvessel.data.ais import clean
from darkvessel.data.scene import Scene
from darkvessel.data.tiling import Tiling
from darkvessel.detect.infer import detect_scene
from darkvessel.detect.stub import BrightPixelDetector
from darkvessel.embed.structures import SAME_POSITION_M
from darkvessel.fusion.azimuth import Geometry
from darkvessel.fusion.interpolate import positions_at
from darkvessel.fusion.match import to_wgs84
from darkvessel.fusion.register import Register
from darkvessel.pipeline import fuse
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


class Viewer:
    """One scene and its AIS archive, loaded once and re-run under many requests.

    Everything a request cannot change is prepared here once: the archive is cleaned (so the
    cleaning report the viewer shows is the archive the match actually searched), and the scene
    summary is fixed. Detection — the only step that reads every pixel — is cached per detector
    setting, and declared positions per max gap, so moving the tolerance slider or toggling the
    azimuth correction never re-reads the scene.
    """

    def __init__(self, scene: Scene, ais: gpd.GeoDataFrame | None) -> None:
        self.scene = scene
        if ais is None:
            self.ais, self.ais_summary = None, None
        else:
            cleaned, report = clean(ais)
            if cleaned.crs != scene.crs:
                cleaned = cleaned.to_crs(scene.crs)
            self.ais = cleaned
            self.ais_summary = {
                "rows_in": report.starting_rows,
                "rows_kept": len(cleaned),
                "removed": dict(report.removed),
                "line": report.line(),
                "vessels": int(cleaned["mmsi"].nunique()) if not cleaned.empty else 0,
            }
        self.scene_summary = _scene_summary(scene)
        self._latitude = _latitude_of(scene)
        # Bounded: the HTTP API accepts any float, and a public server must not grow forever.
        self._detect = lru_cache(maxsize=32)(self._detect_uncached)
        self._declared = lru_cache(maxsize=32)(self._declared_uncached)

    def run(self, request: RunRequest) -> dict[str, Any]:
        """Run the chain under `request` and return everything the viewer needs to draw it."""
        scene = self.scene
        max_gap = timedelta(minutes=request.max_gap_minutes)
        geometry = (
            Geometry(heading_deg=scene.heading_deg, incidence_deg=scene.incidence_deg)
            if request.apply_azimuth
            else None
        )
        register = _register(request)
        found, crops = self._detect(request.detector_threshold, request.tile_px, request.overlap_px)

        detections = fuse(
            scene=scene,
            found=found,
            ais=self.ais,
            tolerance_m=request.tolerance_m,
            max_gap=max_gap,
            geometry=geometry,
            structures=register,
        )

        return {
            "scene": self.scene_summary,
            "config": _config_summary(request, register),
            "counts": _counts(detections),
            "detections": _detections(detections, crops),
            "declarations": self._declarations(request.max_gap_minutes, geometry, detections),
            "ais": self.ais_summary,
            "register": [
                {"x": x, "y": y, **_to_pixel(scene, x, y)} for x, y in request.register_xy
            ],
        }

    def detections_key(self, threshold: float, tile_px: int, overlap_px: int) -> str:
        """Equal for two detector settings that find the same pixels, and so give equal runs."""
        found, _ = self._detect(threshold, tile_px, overlap_px)
        return found.to_json()

    def declarations_key(self, max_gap_minutes: float) -> str:
        """Equal for two max gaps that declare the same positions, and so give equal runs."""
        if self.ais is None:
            return ""
        return self._declared(max_gap_minutes).to_json()

    def _detect_uncached(
        self, threshold: float, tile_px: int, overlap_px: int
    ) -> tuple[pd.DataFrame, list[str]]:
        found = detect_scene(
            self.scene.image,
            BrightPixelDetector(threshold=threshold),
            Tiling(tile_px=tile_px, overlap_px=overlap_px),
        )
        return found, crop_data_uris(self.scene.image, found[["row", "col"]])

    def _declared_uncached(self, max_gap_minutes: float) -> gpd.GeoDataFrame:
        return positions_at(self.ais, self.scene.acquired_at, timedelta(minutes=max_gap_minutes))

    def _declarations(
        self,
        max_gap_minutes: float,
        geometry: Geometry | None,
        detections: gpd.GeoDataFrame,
    ) -> list[dict[str, Any]]:
        """Each vessel's declared position at the acquisition instant, raw and radar-corrected."""
        if self.ais is None or self.ais.empty:
            return []
        declared = self._declared(max_gap_minutes)
        if declared.empty:
            return []

        scene = self.scene
        matched_by_mmsi = {
            str(row["mmsi"]): position
            for position, (_, row) in enumerate(detections.iterrows())
            if row["status"] == "matched" and not pd.isna(row["mmsi"])
        }

        rows = []
        for _, row in declared.iterrows():
            raw = row.geometry
            east, north = 0.0, 0.0
            if geometry is not None and np.isfinite(row["velocity_east_ms"]) and np.isfinite(
                row["velocity_north_ms"]
            ):
                east, north = geometry.displacement(
                    row["velocity_east_ms"], row["velocity_north_ms"], self._latitude
                )
            drawn = Point(raw.x + east, raw.y + north)
            rows.append(
                {
                    "mmsi": str(row["mmsi"]),
                    "length_m": _plain(row["length_m"]),
                    "position_basis": _plain(row["position_basis"]),
                    "position_age_s": _plain(row["position_age_s"]),
                    "azimuth_shift_m": float(np.hypot(east, north)),
                    **_course_and_speed(row["velocity_east_ms"], row["velocity_north_ms"]),
                    "matched_index": matched_by_mmsi.get(str(row["mmsi"])),
                    "raw": {"x": raw.x, "y": raw.y, **_to_pixel(scene, raw.x, raw.y)},
                    "drawn": {"x": drawn.x, "y": drawn.y, **_to_pixel(scene, drawn.x, drawn.y)},
                }
            )
        return rows


def build(scene: Scene, ais: gpd.GeoDataFrame | None, request: RunRequest) -> dict[str, Any]:
    """A one-off run; hold a `Viewer` instead when running the same scene more than once."""
    return Viewer(scene, ais).run(request)


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


_MS_TO_KNOTS = 3600 / 1852


def _course_and_speed(east_ms: float, north_ms: float) -> dict[str, float | None]:
    """Course over ground (degrees clockwise from grid north) and speed in knots.

    Absent — not zero — where the track has no velocity: a lone report says nothing about
    heading, and drawing it as stationary would claim more than the data does.
    """
    if not (np.isfinite(east_ms) and np.isfinite(north_ms)):
        return {"course_deg": None, "speed_kn": None}
    speed = float(np.hypot(east_ms, north_ms))
    course = float(np.degrees(np.arctan2(east_ms, north_ms)) % 360) if speed > 0 else None
    return {"course_deg": course, "speed_kn": speed * _MS_TO_KNOTS}


def _latitude_of(scene: Scene) -> float:
    height, width = scene.image.shape
    x, y = scene.transform * (width / 2, height / 2)
    _, latitude = to_wgs84(str(scene.crs)).transform(x, y)
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
