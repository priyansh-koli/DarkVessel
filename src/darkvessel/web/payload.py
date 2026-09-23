"""One pipeline run, shaped for a viewer.

The frontend draws in pixel space, so every ground coordinate is projected back through the
scene transform here rather than duplicating affine maths in JavaScript. Declarations carry
*both* their raw position and where the azimuth correction moved them to, because the gap
between those two points is the thing an analyst most needs to see: it is the difference
between a matched vessel and a reported dark one.

A run has two sides and the payload carries both. `detections` is the radar's account,
`declarations` the archive's, and `agreement` is what the two say about each other. The
reception model that qualifies the dark claims travels with them, as cells the viewer can
draw, so "this water is barely heard" is something to look at rather than a number to trust.
"""

from __future__ import annotations

import hashlib
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
from darkvessel.fusion.declarations import (
    BELOW_DETECTABLE,
    EXPLAINED,
    OUTSIDE_SCENE,
    SMALLEST_DETECTABLE_M,
    UNDETECTED,
)
from darkvessel.fusion.interpolate import positions_at
from darkvessel.fusion.match import to_wgs84
from darkvessel.fusion.reception import (
    MIN_INTERVALS,
    RECEPTION_CELL_M,
    RECEPTION_FLOOR,
    Coverage,
    coverage_for,
)
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
    register_xy: list = field(default_factory=list)
    register_tolerance_m: float = SAME_POSITION_M
    # The bar a dark claim must clear. Like the register, it only ever rewrites a `dark` row
    # after matching, so a static build can apply it to a baked result rather than bake it.
    reception_floor: float = RECEPTION_FLOOR


class Viewer:
    """One scene and its AIS archive, loaded once and re-run under many requests.

    Everything a request cannot change is prepared here once: the archive is cleaned (so the
    cleaning report the viewer shows is the archive the match actually searched), and the scene
    summary is fixed. Detection — the only step that reads every pixel — is cached per detector
    setting, declared positions per max gap, and the reception model per max gap too, so moving
    the tolerance slider or toggling the azimuth correction never re-reads the scene and never
    re-estimates reception.
    """

    def __init__(
        self,
        scene: Scene,
        ais: gpd.GeoDataFrame | None,
        reception_cell_m: float = RECEPTION_CELL_M,
        reception_min_intervals: int = MIN_INTERVALS,
        smallest_detectable_m: float | None = SMALLEST_DETECTABLE_M,
    ) -> None:
        self.scene = scene
        self.reception_cell_m = float(reception_cell_m)
        self.reception_min_intervals = int(reception_min_intervals)
        self.smallest_detectable_m = smallest_detectable_m
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
        self._coverage = lru_cache(maxsize=32)(self._coverage_uncached)

    def run(self, request: RunRequest) -> dict:
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
        coverage = self._coverage(request.max_gap_minutes)

        fusion = fuse(
            scene=scene,
            found=found,
            ais=self.ais,
            tolerance_m=request.tolerance_m,
            max_gap=max_gap,
            geometry=geometry,
            structures=register,
            coverage=coverage,
            reception_floor=request.reception_floor,
            smallest_detectable_m=self.smallest_detectable_m,
        )

        return {
            "scene": self.scene_summary,
            "config": _config_summary(request, register),
            "counts": _counts(fusion.detections),
            "detections": _detections(fusion.detections, crops),
            "declarations": self._declarations(
                request.max_gap_minutes, geometry, fusion.declarations
            ),
            "declaration_counts": _declaration_counts(fusion.declarations),
            "agreement": _agreement_summary(fusion),
            "reception": self._reception(coverage, request.reception_floor),
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
        """Equal for two max gaps that give equal runs.

        The gap decides two things, not one: which positions are declared, *and* how much of
        the archive's time counts as covered. Two gaps can declare the same vessels and still
        disagree about reception, so both go in the key — keying on the positions alone would
        let a static build serve one gap's answer for another's question.
        """
        if self.ais is None:
            return ""
        coverage = self._coverage(max_gap_minutes)
        return self._declared(max_gap_minutes).to_json() + "|" + _coverage_digest(coverage)

    def _detect_uncached(self, threshold: float, tile_px: int, overlap_px: int):
        found = detect_scene(
            self.scene.image,
            BrightPixelDetector(threshold=threshold),
            Tiling(tile_px=tile_px, overlap_px=overlap_px),
        )
        return found, crop_data_uris(self.scene.image, found[["row", "col"]])

    def _declared_uncached(self, max_gap_minutes: float) -> gpd.GeoDataFrame:
        return positions_at(self.ais, self.scene.acquired_at, timedelta(minutes=max_gap_minutes))

    def _coverage_uncached(self, max_gap_minutes: float) -> Coverage | None:
        return coverage_for(
            self.ais,
            timedelta(minutes=max_gap_minutes),
            cell_m=self.reception_cell_m,
            min_intervals=self.reception_min_intervals,
        )

    def _declarations(
        self,
        max_gap_minutes: float,
        geometry: Geometry | None,
        reviewed: gpd.GeoDataFrame,
    ) -> list:
        """Each vessel's declared position at the acquisition instant, raw and radar-corrected.

        The raw position is recomputed here and the rest is taken from `reviewed`, joined by
        index rather than by MMSI: both come from the same `positions_at` call, so the index
        is exact, and a join on identity would be a second search that could disagree with the
        assignment the run actually made.
        """
        if self.ais is None or self.ais.empty:
            return []
        declared = self._declared(max_gap_minutes)
        if declared.empty or reviewed.empty:
            return []

        scene = self.scene
        verdicts = reviewed.reindex(declared.index)

        rows = []
        for position, (index, row) in enumerate(declared.iterrows()):
            raw = row.geometry
            east, north = 0.0, 0.0
            if (
                geometry is not None
                and np.isfinite(row["velocity_east_ms"])
                and np.isfinite(row["velocity_north_ms"])
            ):
                east, north = geometry.displacement(
                    row["velocity_east_ms"], row["velocity_north_ms"], self._latitude
                )
            drawn = Point(raw.x + east, raw.y + north)
            verdict = verdicts.loc[index]
            rows.append(
                {
                    "mmsi": str(row["mmsi"]),
                    "length_m": _plain(row["length_m"]),
                    "position_basis": _plain(row["position_basis"]),
                    "position_age_s": _plain(row["position_age_s"]),
                    "position_span_s": _plain(row["position_span_s"]),
                    "azimuth_shift_m": float(np.hypot(east, north)),
                    **_course_and_speed(row["velocity_east_ms"], row["velocity_north_ms"]),
                    "status": _plain(verdict["status"]),
                    "nearest_detection_m": _plain(verdict["nearest_detection_m"]),
                    "matched_index": _plain(verdict["detection"]),
                    "index": position,
                    "raw": {"x": raw.x, "y": raw.y, **_to_pixel(scene, raw.x, raw.y)},
                    "drawn": {"x": drawn.x, "y": drawn.y, **_to_pixel(scene, drawn.x, drawn.y)},
                }
            )
        return rows

    def _reception(self, coverage: Coverage | None, floor: float) -> dict | None:
        """The reception model as cells the viewer can draw, clipped to the scene.

        Clipped because the archive covers far more water than one image does, and a cell
        nobody can see on screen is bytes the payload does not need to carry.
        """
        if coverage is None:
            return None
        cells = coverage.cells()
        minx, miny, maxx, maxy = self.scene.footprint.bounds
        visible = cells[
            (cells["x1"] > minx) & (cells["x0"] < maxx)
            & (cells["y1"] > miny) & (cells["y0"] < maxy)
        ]
        report = coverage.report
        return {
            "cell_m": coverage.cell_m,
            "min_intervals": coverage.min_intervals,
            "floor": float(floor),
            "max_gap_minutes": report.max_gap_s / 60.0,
            "report": {
                "intervals": report.intervals,
                "used": report.used,
                "dropped": dict(report.dropped),
                "cells": report.cells,
                "line": report.line(),
            },
            # Rounded: these are drawing coordinates and a shade, and a static bundle holds
            # one copy of this per max gap. Sub-hundredth pixels are bytes for nothing.
            "cells": [
                {
                    "reception_p": _rounded(row["reception_p"], 4),
                    "intervals": _plain(row["intervals"]),
                    **_pixel_box(self.scene, row["x0"], row["y0"], row["x1"], row["y1"]),
                }
                for _, row in visible.iterrows()
            ],
        }


def build(scene: Scene, ais: gpd.GeoDataFrame | None, request: RunRequest) -> dict:
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


def _scene_summary(scene: Scene) -> dict:
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


def _config_summary(request: RunRequest, register: Register | None) -> dict:
    return {
        "tolerance_m": request.tolerance_m,
        "max_gap_minutes": request.max_gap_minutes,
        "detector_threshold": request.detector_threshold,
        "tile_px": request.tile_px,
        "overlap_px": request.overlap_px,
        "apply_azimuth": request.apply_azimuth,
        "register_positions": 0 if register is None else len(register.positions),
        "register_tolerance_m": request.register_tolerance_m,
        "reception_floor": request.reception_floor,
    }


def _counts(detections: gpd.GeoDataFrame) -> dict:
    counts = detections["status"].value_counts()
    return {
        "total": int(len(detections)),
        "matched": int(counts.get("matched", 0)),
        "dark": int(counts.get("dark", 0)),
        "shadowed": int(counts.get("shadowed", 0)),
        "structure": int(counts.get("structure", 0)),
        "unsearched": int(counts.get("unsearched", 0)),
    }


def _declaration_counts(declarations: gpd.GeoDataFrame) -> dict:
    counts = (
        declarations["status"].value_counts()
        if "status" in declarations
        else pd.Series(dtype="int64")
    )
    return {
        "total": int(len(declarations)),
        EXPLAINED: int(counts.get(EXPLAINED, 0)),
        UNDETECTED: int(counts.get(UNDETECTED, 0)),
        BELOW_DETECTABLE: int(counts.get(BELOW_DETECTABLE, 0)),
        OUTSIDE_SCENE: int(counts.get(OUTSIDE_SCENE, 0)),
    }


def _agreement_summary(fusion) -> dict:
    found = fusion.agreement()
    return {
        "both": found.both,
        "radar_only": found.radar_only,
        "ais_only": found.ais_only,
        "outside_scene": found.outside_scene,
        "below_detectable": found.below_detectable,
        "apparent_recall": found.apparent_recall,
        "line": found.line(),
    }


def _detections(detections: gpd.GeoDataFrame, crops: list) -> list:
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


def _course_and_speed(east_ms: float, north_ms: float) -> dict:
    """Course over ground (degrees clockwise from grid north) and speed in knots.

    Absent — not zero — where the track has no velocity: a lone report says nothing about
    heading, and drawing it as stationary would claim more than the data does.
    """
    if not (np.isfinite(east_ms) and np.isfinite(north_ms)):
        return {"course_deg": None, "speed_kn": None}
    speed = float(np.hypot(east_ms, north_ms))
    course = float(np.degrees(np.arctan2(east_ms, north_ms)) % 360) if speed > 0 else None
    return {"course_deg": course, "speed_kn": speed * _MS_TO_KNOTS}


def _coverage_digest(coverage: Coverage | None) -> str:
    """A short, exact fingerprint of a reception model, for equivalence classes in the bake."""
    if coverage is None:
        return ""
    digest = hashlib.sha1()
    for array in (coverage.keys, coverage.covered_s, coverage.total_s, coverage.intervals):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(f"{coverage.cell_m}:{coverage.min_intervals}".encode())
    return digest.hexdigest()


def _latitude_of(scene: Scene) -> float:
    height, width = scene.image.shape
    x, y = scene.transform * (width / 2, height / 2)
    _, latitude = to_wgs84(str(scene.crs)).transform(x, y)
    return float(latitude)


def _to_pixel(scene: Scene, x: float, y: float) -> dict:
    col, row = ~scene.transform * (x, y)
    return {"px": float(col), "py": float(row)}


def _pixel_box(scene: Scene, x0: float, y0: float, x1: float, y1: float) -> dict:
    """A ground rectangle as a pixel-space one. North-up transforms flip y, so both corners
    are projected and then ordered, rather than assuming which way round they come out."""
    first = _to_pixel(scene, x0, y0)
    second = _to_pixel(scene, x1, y1)
    return {
        "px": round(min(first["px"], second["px"]), 2),
        "py": round(min(first["py"], second["py"]), 2),
        "pw": round(abs(second["px"] - first["px"]), 2),
        "ph": round(abs(second["py"] - first["py"]), 2),
    }


def _rounded(value: Any, digits: int) -> Any:
    plain = _plain(value)
    return None if plain is None else round(float(plain), digits)


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
