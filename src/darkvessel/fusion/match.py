"""Spatio-temporal matching, and the dark-vessel decision.

Detections are matched to declared AIS positions within a tolerance. What remains unmatched
is reported as a *candidate*, with its tolerance stated on the row — a claim about evidence
searched, not a verdict.

**Why not greedy.** A naive implementation
sorts every (detection, declaration) pair by distance and greedily claims the closest first,
one-to-one. That is *not* equivalent to finding the maximum number of matches: taking the
single globally-closest pair first can consume a declared position that a different
detection needed, leaving that detection unmatched — reported dark — even though a different,
still-valid one-to-one assignment would have explained it. This gets *more* likely, not less,
in exactly the busy-shipping-lane conditions this kind of study is built around.

The fix: pose it as a linear assignment problem (`scipy.optimize.linear_sum_assignment`).
Infeasible pairs (distance > tolerance) get a cost so large that the optimum always prefers
maximising the number of feasible pairs first, and only among those minimises total
distance — i.e. it is the maximum-cardinality matching, tie-broken by total distance, in one
call. See tests/test_match.py for the constructed case that demonstrates the difference.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from functools import cache

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import linear_sum_assignment
from shapely import Point

from darkvessel.fusion.azimuth import Geometry
from darkvessel.fusion.interpolate import positions_at

MATCHED = "matched"
DARK = "dark"
UNSEARCHED = "unsearched"  # no AIS was supplied at all — distinct from "searched, found nothing"

# Cost added to an out-of-tolerance pair. Real distances here are at most a few km; this makes
# any solution using more within-tolerance pairs strictly cheaper than one using fewer.
_INFEASIBLE_PENALTY = 1.0e9


def classify(
    detections: gpd.GeoDataFrame,
    ais: gpd.GeoDataFrame | None,
    acquired_at: datetime,
    tolerance_m: float,
    max_gap: timedelta,
    geometry: Geometry | None = None,
) -> gpd.GeoDataFrame:
    """Mark each detection matched, dark, or unsearched against the declared positions."""
    searched = ais is not None
    declared = _drawn_by_the_radar(
        _positions_at_acquisition(ais, acquired_at, detections.crs, max_gap), geometry
    )

    classified = detections.copy()
    classified["status"] = DARK if searched else UNSEARCHED
    classified["mmsi"] = pd.Series(pd.NA, index=classified.index, dtype="string")
    classified["length_m"] = np.nan
    classified["match_distance_m"] = np.nan
    classified["tolerance_m"] = float(tolerance_m) if searched else np.nan
    classified["declarations_searched"] = float(len(declared)) if searched else np.nan
    classified["position_basis"] = pd.Series(pd.NA, index=classified.index, dtype="string")
    classified["position_age_s"] = np.nan
    classified["azimuth_shift_m"] = np.nan
    classified["acquired_at"] = acquired_at

    for detection_idx, declared_idx, distance_m in _optimal_pairs(
        classified, declared, tolerance_m
    ):
        classified.loc[detection_idx, "status"] = MATCHED
        classified.loc[detection_idx, "mmsi"] = declared.loc[declared_idx, "mmsi"]
        classified.loc[detection_idx, "length_m"] = declared.loc[declared_idx, "length_m"]
        classified.loc[detection_idx, "match_distance_m"] = distance_m
        classified.loc[detection_idx, "position_basis"] = declared.loc[
            declared_idx, "position_basis"
        ]
        classified.loc[detection_idx, "position_age_s"] = declared.loc[
            declared_idx, "position_age_s"
        ]
        classified.loc[detection_idx, "azimuth_shift_m"] = declared.loc[
            declared_idx, "azimuth_shift_m"
        ]

    return classified


def _drawn_by_the_radar(declared: gpd.GeoDataFrame, geometry: Geometry | None) -> gpd.GeoDataFrame:
    """Move each declared position to where the radar would actually have drawn that vessel."""
    moved = declared.copy()
    if geometry is None or moved.empty:
        moved["azimuth_shift_m"] = np.nan
        return moved

    latitude = _latitude_of(moved)
    shifts = [
        (
            geometry.displacement(east, north, latitude)
            if np.isfinite(east) and np.isfinite(north)
            else None
        )
        for east, north in zip(moved["velocity_east_ms"], moved["velocity_north_ms"])
    ]
    moved["azimuth_shift_m"] = [float(np.hypot(*s)) if s is not None else np.nan for s in shifts]
    moved.geometry = gpd.GeoSeries(
        [
            pt if s is None else Point(pt.x + s[0], pt.y + s[1])
            for pt, s in zip(moved.geometry, shifts)
        ],
        index=moved.index,
        crs=declared.crs,
    )
    return moved


def _latitude_of(positions: gpd.GeoDataFrame) -> float:
    centre = positions.geometry.union_all().centroid
    _, latitude = to_wgs84(str(positions.crs)).transform(centre.x, centre.y)
    return float(latitude)


@cache
def to_wgs84(crs: str) -> Transformer:
    """A transformer from `crs` to lon/lat, built once: constructing one costs more than a match."""
    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def _positions_at_acquisition(
    ais: gpd.GeoDataFrame | None, acquired_at: datetime, crs: str, max_gap: timedelta
) -> gpd.GeoDataFrame:
    if ais is None or ais.empty:
        return gpd.GeoDataFrame(
            {"mmsi": [], "length_m": [], "position_basis": [], "position_age_s": []},
            geometry=[],
            crs=crs,
        )
    if ais.crs != crs:
        ais = ais.to_crs(crs)
    return positions_at(ais, acquired_at, max_gap)


def _optimal_pairs(
    detections: gpd.GeoDataFrame, declared: gpd.GeoDataFrame, tolerance_m: float
) -> Iterator[tuple[int, int, float]]:
    """Maximum-cardinality, minimum-total-distance one-to-one pairing within `tolerance_m`."""
    if declared.empty or len(detections) == 0:
        return

    det_idx = list(detections.index)
    dec_idx = list(declared.index)
    distances = np.array(
        [
            [detections.geometry.loc[d].distance(declared.geometry.loc[a]) for a in dec_idx]
            for d in det_idx
        ]
    )

    cost = np.where(distances <= tolerance_m, distances, distances + _INFEASIBLE_PENALTY)
    rows, cols = linear_sum_assignment(cost)

    for r, c in zip(rows, cols):
        distance_m = float(distances[r, c])
        if distance_m <= tolerance_m:
            yield det_idx[r], dec_idx[c], distance_m
