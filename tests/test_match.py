"""Matching tests, including the case where greedy matching fails.

A naive greedy matcher (sort every pair by distance, claim closest first, one-to-one) can
leave a detection unmatched even when a valid one-to-one assignment exists that would have
explained every detection. `_optimal_pairs` is checked directly against exactly that case.
"""

from datetime import timedelta

import geopandas as gpd
import pandas as pd
import pytest
from shapely import Point

from darkvessel.fusion.match import DARK, MATCHED, UNSEARCHED, _optimal_pairs, classify


def _points(coords: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in coords], crs="EPSG:25832")


def test_greedy_would_leave_a_detection_unmatched_but_optimal_does_not():
    """D1-A1=90, D1-A2=95, D2-A1=98, D2-A2=283 (out of a 100 m tolerance).

    A closest-first greedy matcher takes (D1, A1) at 90 m first — the single smallest edge —
    which then leaves D2 with no available declaration (its only edge, A1, is taken), even
    though the assignment (D1-A2, D2-A1) fully covers both detections within tolerance.
    """
    detections = _points([(0, 0), (188, 0)])
    declared = _points([(90, 0), (-95, 0)])
    assert detections.geometry[0].distance(declared.geometry[0]) == 90.0
    assert detections.geometry[0].distance(declared.geometry[1]) == 95.0
    assert detections.geometry[1].distance(declared.geometry[0]) == 98.0
    assert detections.geometry[1].distance(declared.geometry[1]) == 283.0

    pairs = list(_optimal_pairs(detections, declared, tolerance_m=100.0))
    matched_detections = {detection_idx for detection_idx, _, _ in pairs}

    assert matched_detections == {0, 1}, "both detections should be explained, not just one"


def test_optimal_pairs_prefers_minimum_total_distance_among_full_matchings():
    """Where more than one assignment achieves full coverage, the cheaper one is chosen."""
    detections = _points([(0, 0), (10, 0)])
    declared = _points([(1, 0), (9, 0)])  # (D1-A1, D2-A2) costs 1+1=2; the cross pairing costs 18
    pairs = {(d, a): dist for d, a, dist in _optimal_pairs(detections, declared, tolerance_m=50.0)}
    assert pairs == {(0, 0): 1.0, (1, 1): 1.0}


def test_a_declaration_explains_at_most_one_detection():
    """Two detections both near one declaration: only one may claim it."""
    detections = _points([(0, 0), (1, 0)])
    declared = _points([(0.5, 0)])
    pairs = list(_optimal_pairs(detections, declared, tolerance_m=10.0))
    assert len(pairs) == 1


def test_classify_marks_unsearched_when_ais_is_none():
    detections = _points([(0, 0)])
    result = classify(
        detections,
        None,
        pd.Timestamp("2026-08-09T05:31:24Z"),
        tolerance_m=100.0,
        max_gap=timedelta(minutes=10),
    ).detections
    assert result["status"].tolist() == [UNSEARCHED]
    assert result["declarations_searched"].isna().all()


def test_classify_marks_dark_when_ais_search_finds_nothing():
    detections = _points([(0, 0)])
    empty_ais = gpd.GeoDataFrame(
        {"mmsi": [], "timestamp": [], "length_m": []}, geometry=[], crs="EPSG:25832"
    )
    result = classify(
        detections,
        empty_ais,
        pd.Timestamp("2026-08-09T05:31:24Z"),
        tolerance_m=100.0,
        max_gap=timedelta(minutes=10),
    ).detections
    assert result["status"].tolist() == [DARK]
    assert result["declarations_searched"].tolist() == [0.0]


def test_classify_matches_within_tolerance():
    acquisition = pd.Timestamp("2026-08-09T05:31:24Z")
    detections = _points([(0, 0)])
    ais = gpd.GeoDataFrame(
        {
            "mmsi": pd.array(["219000001"], dtype="string"),
            "timestamp": [acquisition],
            "length_m": [140.0],
        },
        geometry=[Point(50, 0)],
        crs="EPSG:25832",
    )
    matched = classify(
        detections, ais, acquisition, tolerance_m=100.0, max_gap=timedelta(minutes=10)
    )
    assert matched.detections["status"].tolist() == [MATCHED]
    assert matched.detections["mmsi"].tolist() == ["219000001"]
    assert matched.detections["match_distance_m"].tolist() == [50.0]

    # Both sides of one assignment: the declaration knows which detection it explained.
    assert matched.declared["detection"].tolist() == [0]
    assert matched.declared["match_distance_m"].tolist() == [50.0]


def test_the_component_split_gives_the_dense_solution_exactly():
    """The matcher splits the feasibility graph into components and solves each alone. That is
    an optimisation, so it has to be indistinguishable from solving the whole matrix at once —
    checked here against a dense `linear_sum_assignment` over random point clouds."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    from darkvessel.fusion.match import _INFEASIBLE_PENALTY

    rng = np.random.default_rng(0)
    for trial in range(25):
        tolerance = 60.0
        # Clustered, not uniform: uniform points give one giant component and never exercise
        # the split. Shipping lanes cluster, which is the case that matters.
        centres = rng.uniform(0, 4000, size=(6, 2))
        detections = _points(
            [tuple(centres[i % 6] + rng.normal(0, 40, 2)) for i in range(18)]
        )
        declared = _points([tuple(centres[i % 6] + rng.normal(0, 40, 2)) for i in range(15)])

        here = np.array([[p.x, p.y] for p in detections.geometry])
        there = np.array([[p.x, p.y] for p in declared.geometry])
        distances = np.hypot(
            here[:, 0, None] - there[None, :, 0], here[:, 1, None] - there[None, :, 1]
        )
        cost = np.where(distances <= tolerance, distances, _INFEASIBLE_PENALTY)
        rows, cols = linear_sum_assignment(cost)
        dense = {
            (int(r), int(c)) for r, c in zip(rows, cols) if distances[r, c] <= tolerance
        }

        split = {(d, a) for d, a, _ in _optimal_pairs(detections, declared, tolerance)}
        assert len(split) == len(dense), f"trial {trial}: different number of matches"
        assert sum(distances[d, a] for d, a in split) == pytest.approx(
            sum(distances[d, a] for d, a in dense)
        ), f"trial {trial}: same count but a more expensive assignment"


def test_an_unreachable_detection_never_enters_a_cost_matrix():
    """A detection with no declaration within tolerance cannot be matched, so it is dropped
    before the assignment rather than padded into it — which is what keeps a scene of
    thousands of isolated detections from costing a cubic solve."""
    detections = _points([(0, 0), (100_000, 100_000)])
    declared = _points([(10, 0)])
    pairs = list(_optimal_pairs(detections, declared, tolerance_m=50.0))
    assert pairs == [(0, 0, 10.0)]
