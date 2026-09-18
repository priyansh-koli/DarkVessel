"""The synthetic fixture: no credentials, no weights, no network.

Built so every branch in the pipeline gets exercised on purpose. See the README's Quick start
for the numbers this produces: 5 detections, 4 matched, 1 dark, 0 at a fixed structure.

Five targets, five stories:
  - `stationary_match`     a plain match: one AIS report, close to the detection.
  - `interpolated_match`   the failure case the README calls out. Its two AIS reports bracket
                            the acquisition and interpolate exactly onto the target, but
                            *either report alone* sits 900 m away — outside any sane
                            tolerance. Matched against a report taken as it stands, this
                            vessel would come back dark.
  - `azimuth_corrected_match`  a fast vessel whose *declared* position is hundreds of metres
                            from the detection until the azimuth-shift correction is applied,
                            after which it lands on it exactly.
  - `simple_match`         another plain match, for variety.
  - `dark_vessel`          no AIS report at all: a genuine dark candidate.

The interpolated vessel's velocity is placed along the satellite's flight direction on
purpose, so its own azimuth-shift correction is zero — its story stays about interpolation
alone, uncomplicated by the correction the other vessel is there to demonstrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import cos, radians, sin
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from shapely import Point

from darkvessel.data.scene import Scene, write_scene
from darkvessel.fusion.azimuth import Geometry

CRS = "EPSG:25832"
PIXEL_SIZE_M = 5.0
IMAGE_SHAPE = (240, 240)  # rows, cols
ORIGIN = (499900.0, 6101100.0)  # ground (x, y) of the top-left pixel corner
ACQUIRED_AT = datetime(2026, 8, 9, 5, 31, 24, tzinfo=timezone.utc)
HEADING_DEG = 350.0
INCIDENCE_DEG = 35.0
TOLERANCE_M = 200.0

TRANSFORM = Affine(PIXEL_SIZE_M, 0.0, ORIGIN[0], 0.0, -PIXEL_SIZE_M, ORIGIN[1])


@dataclass(frozen=True)
class _Target:
    name: str
    x: float
    y: float


_TARGETS = [
    _Target("stationary_match", 500100.0, 6100900.0),
    _Target("interpolated_match", 500300.0, 6100900.0),
    _Target("azimuth_corrected_match", 500500.0, 6100900.0),
    _Target("simple_match", 500700.0, 6100900.0),
    _Target("dark_vessel", 500900.0, 6100900.0),
]


def _by_name(name: str) -> _Target:
    return next(t for t in _TARGETS if t.name == name)


def _blob(image: np.ndarray, x: float, y: float, value: float = 1.0) -> None:
    """Paint a 2x2 block of equal-value pixels — the flat plateau `detect.stub` must collapse
    to one detection rather than reporting per pixel."""
    col, row = ~TRANSFORM * (x, y)
    r, c = int(row), int(col)
    image[r : r + 2, c : c + 2] = value


def _scene() -> Scene:
    image = np.zeros(IMAGE_SHAPE, dtype=np.float32)
    for target in _TARGETS:
        _blob(image, target.x, target.y)
    return Scene(
        id="synthetic-scene-1",
        image=image,
        transform=TRANSFORM,
        crs=CRS,
        acquired_at=ACQUIRED_AT,
        heading_deg=HEADING_DEG,
        incidence_deg=INCIDENCE_DEG,
    )


def _flight_direction() -> tuple[float, float]:
    heading = radians(HEADING_DEG)
    return sin(heading), cos(heading)


def _ais_rows() -> list[dict]:
    rows = []

    stationary = _by_name("stationary_match")
    rows.append(
        dict(
            mmsi="219000001",
            timestamp=ACQUIRED_AT,
            length_m=50.0,
            geometry=Point(stationary.x + 20.0, stationary.y - 10.0),
        )
    )

    # Bracketing reports 5 minutes either side, offset 900 m along the flight direction so
    # velocity has no line-of-sight component: this vessel's azimuth shift is exactly zero,
    # keeping its story about interpolation alone. Either single report sits 900 m from the
    # target; the interpolated midpoint sits exactly on it.
    interpolated = _by_name("interpolated_match")
    flight_east, flight_north = _flight_direction()
    offset = 900.0
    rows.append(
        dict(
            mmsi="219000002",
            timestamp=ACQUIRED_AT - timedelta(minutes=5),
            length_m=80.0,
            geometry=Point(
                interpolated.x - offset * flight_east, interpolated.y - offset * flight_north
            ),
        )
    )
    rows.append(
        dict(
            mmsi="219000002",
            timestamp=ACQUIRED_AT + timedelta(minutes=5),
            length_m=80.0,
            geometry=Point(
                interpolated.x + offset * flight_east, interpolated.y + offset * flight_north
            ),
        )
    )

    # One report exactly at the acquisition, and one 10 minutes earlier so a velocity can be
    # derived from history — the same way a real archive would give one. The *raw* declared
    # position is computed by inverting the same `fusion.azimuth.Geometry` the pipeline
    # applies, so this fixture's correctness doesn't depend on hand-computed offsets tracking
    # that formula.
    azimuth = _by_name("azimuth_corrected_match")
    velocity_east, velocity_north = 8.0, 0.0
    geometry = Geometry(heading_deg=HEADING_DEG, incidence_deg=INCIDENCE_DEG)
    shift_east, shift_north = geometry.displacement(velocity_east, velocity_north, latitude=55.7)
    raw_at_acquisition = Point(azimuth.x - shift_east, azimuth.y - shift_north)
    raw_before = Point(
        raw_at_acquisition.x - velocity_east * 600.0,
        raw_at_acquisition.y - velocity_north * 600.0,
    )
    rows.append(
        dict(
            mmsi="219000003",
            timestamp=ACQUIRED_AT - timedelta(minutes=10),
            length_m=140.0,
            geometry=raw_before,
        )
    )
    rows.append(
        dict(
            mmsi="219000003",
            timestamp=ACQUIRED_AT,
            length_m=140.0,
            geometry=raw_at_acquisition,
        )
    )

    simple = _by_name("simple_match")
    rows.append(
        dict(
            mmsi="219000004",
            timestamp=ACQUIRED_AT,
            length_m=60.0,
            geometry=Point(simple.x + 30.0, simple.y),
        )
    )

    # "dark_vessel" gets no AIS row at all — nothing declares it, so it stays dark.
    return rows


def _ais() -> gpd.GeoDataFrame:
    frame = pd.DataFrame(_ais_rows())
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["mmsi"] = frame["mmsi"].astype("string")
    return gpd.GeoDataFrame(frame.drop(columns=["geometry"]), geometry=frame["geometry"], crs=CRS)


def write_synthetic(out_dir: str | Path) -> Path:
    """Write a scene and its AIS archive that exercise every branch in the pipeline on purpose.

    Returns the directory written to.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_scene(out_dir / "scene", _scene())
    _ais().to_file(out_dir / "ais.gpkg", driver="GPKG")
    return out_dir
