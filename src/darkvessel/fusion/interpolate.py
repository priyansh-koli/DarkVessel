"""AIS interpolation to the acquisition instant.

A vessel at 12 knots covers ~370 m a minute — more than any sane match tolerance — so every
declared position is placed at the acquisition timestamp before anything is compared. Nothing
is ever extrapolated past the end of a track: where the acquisition falls outside a vessel's
bracketing reports, the nearest single report is used instead, and the row says so in
`position_basis`, so a match built on it can be told apart from one built on a real bracket.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Point

INTERPOLATED = "interpolated"
NEAREST = "nearest"

_EMPTY_COLUMNS = {
    "mmsi": "string",
    "length_m": "float64",
    "position_basis": "string",
    "position_age_s": "float64",
    "velocity_east_ms": "float64",
    "velocity_north_ms": "float64",
}


def positions_at(
    ais: gpd.GeoDataFrame, acquired_at: datetime, max_gap: timedelta
) -> gpd.GeoDataFrame:
    """One declared position per MMSI at `acquired_at`, or none where nothing is close enough.

    `ais` carries `mmsi`, `timestamp`, `length_m` and point geometry, one row per report.
    """
    acquired_at = pd.Timestamp(acquired_at)
    rows = []
    for mmsi, reports in ais.groupby("mmsi", sort=False):
        reports = reports.sort_values("timestamp")
        row = _position_for(mmsi, reports, acquired_at, max_gap)
        if row is not None:
            rows.append(row)

    if not rows:
        return gpd.GeoDataFrame(
            {name: pd.array([], dtype=dtype) for name, dtype in _EMPTY_COLUMNS.items()},
            geometry=[],
            crs=ais.crs,
        )

    frame = pd.DataFrame(rows)
    return gpd.GeoDataFrame(
        frame.drop(columns=["geometry"]), geometry=frame["geometry"], crs=ais.crs
    )


def _position_for(
    mmsi: str, reports: pd.DataFrame, acquired_at: pd.Timestamp, max_gap: timedelta
) -> dict | None:
    timestamps = pd.DatetimeIndex(reports["timestamp"])
    length_m = float(reports["length_m"].iloc[-1])

    before = reports[timestamps <= acquired_at]
    after = reports[timestamps > acquired_at]

    if not before.empty and not after.empty:
        left, right = before.iloc[-1], after.iloc[0]
        span = (right["timestamp"] - left["timestamp"]).total_seconds()
        fraction = 0.0 if span == 0 else (acquired_at - left["timestamp"]).total_seconds() / span
        x = left.geometry.x + fraction * (right.geometry.x - left.geometry.x)
        y = left.geometry.y + fraction * (right.geometry.y - left.geometry.y)
        velocity_east = (right.geometry.x - left.geometry.x) / span if span else np.nan
        velocity_north = (right.geometry.y - left.geometry.y) / span if span else np.nan
        return {
            "mmsi": mmsi,
            "length_m": length_m,
            "position_basis": INTERPOLATED,
            "position_age_s": 0.0,
            "velocity_east_ms": velocity_east,
            "velocity_north_ms": velocity_north,
            "geometry": Point(x, y),
        }

    nearest = before.iloc[-1] if not before.empty else (after.iloc[0] if not after.empty else None)
    if nearest is None:
        return None
    age_s = abs((acquired_at - nearest["timestamp"]).total_seconds())
    if age_s > max_gap.total_seconds():
        return None

    velocity_east, velocity_north = np.nan, np.nan
    candidates = before if not before.empty else after
    if len(candidates) >= 2:
        prior, current = candidates.iloc[-2], candidates.iloc[-1]
        span = (current["timestamp"] - prior["timestamp"]).total_seconds()
        if span > 0:
            velocity_east = (current.geometry.x - prior.geometry.x) / span
            velocity_north = (current.geometry.y - prior.geometry.y) / span

    return {
        "mmsi": mmsi,
        "length_m": length_m,
        "position_basis": NEAREST,
        "position_age_s": age_s,
        "velocity_east_ms": velocity_east,
        "velocity_north_ms": velocity_north,
        "geometry": nearest.geometry,
    }
