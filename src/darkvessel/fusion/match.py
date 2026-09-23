"""Spatio-temporal matching, and the dark-vessel decision.

Detections are matched to declared AIS positions within a tolerance. What remains unmatched
is reported as a *candidate*, with its tolerance stated on the row — a claim about evidence
searched, not a verdict.

Matching has two sides and `classify` returns both, as a `Match`. The radar side is the
familiar one: detections nothing declared. The AIS side is its mirror — declarations the
radar drew nothing for — and throwing it away discards half of what the assignment already
computed. `fusion.declarations` gives that side its own verdicts.

**Why not greedy.** A naive implementation
sorts every (detection, declaration) pair by distance and greedily claims the closest first,
one-to-one. That is *not* equivalent to finding the maximum number of matches: taking the
single globally-closest pair first can consume a declared position that a different
detection needed, leaving that detection unmatched — reported dark — even though a different,
still-valid one-to-one assignment would have explained it. This gets *more* likely, not less,
in exactly the busy-shipping-lane conditions this kind of study is built around.

The fix: pose it as a linear assignment problem (`scipy.optimize.linear_sum_assignment`).
Out-of-tolerance pairs get a flat cost so large that the optimum always prefers maximising
the number of feasible pairs first, and only among those minimises total distance — i.e. it
is the maximum-cardinality matching, tie-broken by total distance, in one call. See
tests/test_match.py for the constructed case that demonstrates the difference.

**Why it is not one call in practice.** A dense cost matrix is O(n x m) to build and the
assignment O((n+m)^3) to solve, which an archive-wide run cannot afford: a Sentinel-1 scene
over Danish waters can hold thousands of detections and as many declarations, nearly all of
them kilometres apart and so incapable of ever being paired. Only pairs within the tolerance
can be chosen, so the feasibility graph is built with a k-d tree and split into connected
components, and each component is solved on its own. Because no feasible edge crosses a
component and an infeasible pair costs the same flat penalty wherever it sits, the union of
the per-component optima *is* the global optimum — the decomposition changes the cost of the
work, never the answer. `tests/test_match.py` checks that equivalence against the dense
solution on random inputs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache
from itertools import chain

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from darkvessel.fusion.azimuth import Geometry
from darkvessel.fusion.interpolate import positions_at

MATCHED = "matched"
DARK = "dark"
UNSEARCHED = "unsearched"  # no AIS was supplied at all — distinct from "searched, found nothing"

# The flat cost of pairing two things further apart than the tolerance. Real distances here are
# at most a few km, so any solution using more within-tolerance pairs is strictly cheaper than
# one using fewer. It is deliberately *flat* rather than distance + penalty: adding the
# distance would let the arbitrary pairing of the leftovers — pairs that are discarded a line
# later — break ties between assignments that explain the same detections.
_INFEASIBLE_PENALTY = 1.0e9


@dataclass(frozen=True)
class Match:
    """Both sides of one assignment, at the acquisition instant.

    `detections` is the radar side, classified. `declared` is the AIS side: one row per
    vessel placed at the acquisition instant and moved to where the radar would have drawn
    it, carrying `detection` (the index of the detection it explains, or absent) and
    `match_distance_m`. The two are consistent by construction because they come from the
    same assignment rather than from two searches that could disagree.
    """

    detections: gpd.GeoDataFrame
    declared: gpd.GeoDataFrame


def classify(
    detections: gpd.GeoDataFrame,
    ais: gpd.GeoDataFrame | None,
    acquired_at: datetime,
    tolerance_m: float,
    max_gap: timedelta,
    geometry: Geometry | None = None,
) -> Match:
    """Mark each detection matched, dark, or unsearched against the declared positions."""
    searched = ais is not None
    declared = drawn_by_the_radar(
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
    classified["position_span_s"] = np.nan
    classified["azimuth_shift_m"] = np.nan
    classified["acquired_at"] = acquired_at

    declared["detection"] = pd.Series(pd.NA, index=declared.index, dtype="Int64")
    declared["match_distance_m"] = np.nan

    # Columns are written in one vectorised pass per column rather than row by row: a busy
    # scene matches thousands of pairs, and `.loc` assignment per pair dominated the fusion.
    pairs = list(_optimal_pairs(classified, declared, tolerance_m))
    if pairs:
        detection_idx = [d for d, _, _ in pairs]
        declared_idx = [a for _, a, _ in pairs]
        distances = [distance for _, _, distance in pairs]
        source = declared.loc[declared_idx]

        classified.loc[detection_idx, "status"] = MATCHED
        classified.loc[detection_idx, "match_distance_m"] = distances
        carried = (
            "mmsi",
            "length_m",
            "position_basis",
            "position_age_s",
            "position_span_s",
            "azimuth_shift_m",
        )
        for column in carried:
            classified.loc[detection_idx, column] = source[column].to_numpy()

        declared.loc[declared_idx, "detection"] = pd.array(detection_idx, dtype="Int64")
        declared.loc[declared_idx, "match_distance_m"] = distances

    return Match(detections=classified, declared=declared)


def drawn_by_the_radar(
    declared: gpd.GeoDataFrame, geometry: Geometry | None
) -> gpd.GeoDataFrame:
    """Move each declared position to where the radar would actually have drawn that vessel."""
    moved = declared.copy()
    if geometry is None or moved.empty:
        moved["azimuth_shift_m"] = np.nan
        return moved

    east, north = geometry.displacements(
        moved["velocity_east_ms"].to_numpy(dtype=float),
        moved["velocity_north_ms"].to_numpy(dtype=float),
        _latitude_of(moved),
    )
    # A declaration with no derivable velocity is left exactly where it declared: unknown is
    # not zero, and moving it by nan would drop the row out of every comparison downstream.
    known = np.isfinite(east) & np.isfinite(north)
    moved["azimuth_shift_m"] = np.where(known, np.hypot(east, north), np.nan)

    xy = shapely.get_coordinates(moved.geometry.values)
    xy[:, 0] += np.where(known, east, 0.0)
    xy[:, 1] += np.where(known, north, 0.0)
    moved.geometry = gpd.GeoSeries(
        shapely.points(xy), index=moved.index, crs=declared.crs
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
            {
                "mmsi": pd.array([], dtype="string"),
                "length_m": pd.array([], dtype="float64"),
                "position_basis": pd.array([], dtype="string"),
                "position_age_s": pd.array([], dtype="float64"),
                "position_span_s": pd.array([], dtype="float64"),
                "velocity_east_ms": pd.array([], dtype="float64"),
                "velocity_north_ms": pd.array([], dtype="float64"),
            },
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
    detection_xy = shapely.get_coordinates(detections.geometry.values)
    declared_xy = shapely.get_coordinates(declared.geometry.values)
    if len(detection_xy) == 0 or len(declared_xy) == 0:
        return

    detection_idx = np.asarray(detections.index)
    declared_idx = np.asarray(declared.index)

    rows, cols = _feasible_edges(detection_xy, declared_xy, tolerance_m)
    if len(rows) == 0:
        return

    for block_rows, block_cols in _components(rows, cols, len(detection_xy), len(declared_xy)):
        here, there = detection_xy[block_rows], declared_xy[block_cols]
        distances = np.hypot(
            here[:, 0, None] - there[None, :, 0], here[:, 1, None] - there[None, :, 1]
        )
        cost = np.where(distances <= tolerance_m, distances, _INFEASIBLE_PENALTY)
        for row, col in zip(*linear_sum_assignment(cost)):
            distance_m = float(distances[row, col])
            if distance_m <= tolerance_m:
                yield int(detection_idx[block_rows[row]]), int(declared_idx[block_cols[col]]), (
                    distance_m
                )


def _feasible_edges(
    detection_xy: np.ndarray, declared_xy: np.ndarray, tolerance_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Every (detection, declaration) pair close enough to be chosen, found with a k-d tree.

    `query_ball_tree` rather than `sparse_distance_matrix`, which stores distances: a pair
    exactly 0 m apart would be an implicit zero in the sparse matrix and vanish — and a
    perfect match is precisely the pair that must never be lost.
    """
    neighbours = cKDTree(detection_xy).query_ball_tree(cKDTree(declared_xy), r=tolerance_m)
    counts = [len(found) for found in neighbours]
    rows = np.repeat(np.arange(len(neighbours)), counts)
    cols = np.fromiter(chain.from_iterable(neighbours), dtype=np.int64, count=int(sum(counts)))
    return rows, cols


def _components(
    rows: np.ndarray, cols: np.ndarray, detections: int, declarations: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Split the feasibility graph into independent blocks, smallest work first.

    Nodes are the detections followed by the declarations. Only components that carry an edge
    are yielded: an isolated detection can never be matched and an isolated declaration can
    never explain anything, so neither belongs in any cost matrix.
    """
    size = detections + declarations
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols + detections)), shape=(size, size)
    )
    _, labels = connected_components(graph, directed=False)
    wanted = np.unique(labels[rows])
    yield from zip(
        _grouped(labels[:detections], wanted), _grouped(labels[detections:], wanted)
    )


def _grouped(labels: np.ndarray, wanted: np.ndarray) -> list[np.ndarray]:
    """The indices carrying each of `wanted` (sorted), in one sort rather than one scan each."""
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.searchsorted(sorted_labels, wanted, side="left")
    ends = np.searchsorted(sorted_labels, wanted, side="right")
    return [order[start:end] for start, end in zip(starts, ends)]
