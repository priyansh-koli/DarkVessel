"""The declaration side of the fusion, judged on its own.

`fusion.declarations` answers the question the radar side cannot: which declarations did no
detection explain? Three of its four verdicts exist to say that the radar's silence about a
declared vessel means nothing, and those are the ones worth pinning — a pipeline that called
all of them findings would manufacture them out of geometry and vessel length.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely import Point, box

from darkvessel.fusion.declarations import (
    BELOW_DETECTABLE,
    EXPLAINED,
    OUTSIDE_SCENE,
    UNDETECTED,
    Agreement,
    agreement,
    review,
)
from darkvessel.fusion.match import DARK, MATCHED

CRS = "EPSG:25832"
FOOTPRINT = box(0, 0, 1000, 1000)


def _declared(rows: list) -> gpd.GeoDataFrame:
    """`fusion.match.Match.declared`: positions already moved to where the radar would draw."""
    frame = pd.DataFrame(rows)
    return gpd.GeoDataFrame(
        {
            "mmsi": frame["mmsi"].astype("string"),
            "length_m": frame["length_m"].astype(float),
            "detection": frame["detection"].astype("Int64"),
            "match_distance_m": frame.get("match_distance_m", pd.Series(np.nan, frame.index)),
            "position_basis": pd.array(["nearest"] * len(frame), dtype="string"),
            "position_age_s": np.zeros(len(frame)),
            "position_span_s": np.full(len(frame), np.nan),
            "azimuth_shift_m": np.zeros(len(frame)),
        },
        geometry=list(frame["geometry"]),
        crs=CRS,
    )


def _detections(coords: list) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"status": [DARK] * len(coords)}, geometry=[Point(x, y) for x, y in coords], crs=CRS
    )


def test_a_declaration_a_detection_explains_is_explained():
    declared = _declared(
        [{"mmsi": "1", "length_m": 90.0, "detection": 0, "geometry": Point(500, 500)}]
    )
    reviewed = review(declared, _detections([(505, 500)]), FOOTPRINT, tolerance_m=200.0)
    assert reviewed["status"].tolist() == [EXPLAINED]


def test_a_declared_vessel_the_radar_drew_nothing_for_is_the_finding():
    declared = _declared(
        [{"mmsi": "1", "length_m": 90.0, "detection": pd.NA, "geometry": Point(500, 500)}]
    )
    reviewed = review(declared, _detections([(900, 900)]), FOOTPRINT, tolerance_m=200.0)
    assert reviewed["status"].tolist() == [UNDETECTED]


def test_how_near_the_radar_came_is_reported_whatever_the_tolerance():
    """"The nearest detection was 260 m away and the bar was 200" is the single most useful
    thing to know about an undetected declaration, and a status alone never says it."""
    declared = _declared(
        [{"mmsi": "1", "length_m": 90.0, "detection": pd.NA, "geometry": Point(0, 0)}]
    )
    reviewed = review(declared, _detections([(260, 0)]), FOOTPRINT, tolerance_m=200.0)
    assert reviewed["nearest_detection_m"].tolist() == pytest.approx([260.0])
    assert reviewed["tolerance_m"].tolist() == [200.0]


def test_a_declaration_outside_the_footprint_is_not_a_finding():
    """Nothing about it was searched: the declaration side's `unsearched`. It is checked
    before length, because a vessel out of frame is out of frame whatever size it is."""
    declared = _declared(
        [{"mmsi": "1", "length_m": 5.0, "detection": pd.NA, "geometry": Point(5000, 5000)}]
    )
    reviewed = review(declared, _detections([(500, 500)]), FOOTPRINT, tolerance_m=200.0)
    assert reviewed["status"].tolist() == [OUTSIDE_SCENE]


def test_a_vessel_under_the_detectors_floor_is_excused():
    declared = _declared(
        [{"mmsi": "1", "length_m": 12.0, "detection": pd.NA, "geometry": Point(500, 500)}]
    )
    reviewed = review(
        declared, _detections([(900, 900)]), FOOTPRINT, tolerance_m=200.0,
        smallest_detectable_m=20.0,
    )
    assert reviewed["status"].tolist() == [BELOW_DETECTABLE]


def test_an_unknown_length_is_not_a_small_one():
    """Excusing a miss on a length the archive never gave would turn missing data into an
    explanation — the same error `unsearched` exists to prevent, one column over."""
    declared = _declared(
        [{"mmsi": "1", "length_m": np.nan, "detection": pd.NA, "geometry": Point(500, 500)}]
    )
    reviewed = review(
        declared, _detections([(900, 900)]), FOOTPRINT, tolerance_m=200.0,
        smallest_detectable_m=20.0,
    )
    assert reviewed["status"].tolist() == [UNDETECTED]


def test_turning_the_length_check_off_excuses_nothing():
    declared = _declared(
        [{"mmsi": "1", "length_m": 2.0, "detection": pd.NA, "geometry": Point(500, 500)}]
    )
    reviewed = review(
        declared, _detections([(900, 900)]), FOOTPRINT, tolerance_m=200.0,
        smallest_detectable_m=None,
    )
    assert reviewed["status"].tolist() == [UNDETECTED]
    assert reviewed["smallest_detectable_m"].isna().all()


def test_with_no_footprint_nothing_is_ruled_outside_it():
    """A caller that cannot say what was searched must not have that read as "everything"."""
    declared = _declared(
        [{"mmsi": "1", "length_m": 90.0, "detection": pd.NA, "geometry": Point(9e6, 9e6)}]
    )
    reviewed = review(declared, _detections([(0, 0)]), None, tolerance_m=200.0)
    assert reviewed["status"].tolist() == [UNDETECTED]


def test_a_scene_with_no_detections_leaves_the_distance_absent_not_zero():
    declared = _declared(
        [{"mmsi": "1", "length_m": 90.0, "detection": pd.NA, "geometry": Point(500, 500)}]
    )
    reviewed = review(declared, _detections([]), FOOTPRINT, tolerance_m=200.0)
    assert reviewed["status"].tolist() == [UNDETECTED]
    assert reviewed["nearest_detection_m"].isna().all()


_NOTHING_DECLARED = gpd.GeoDataFrame(
    {
        "mmsi": pd.array([], dtype="string"),
        "length_m": pd.array([], dtype="float64"),
        "detection": pd.array([], dtype="Int64"),
    },
    geometry=[],
    crs=CRS,
)


def test_an_empty_archive_still_has_the_layers_schema():
    """A reader must never have to branch on whether anything was declared."""
    empty = review(_NOTHING_DECLARED, _detections([(0, 0)]), FOOTPRINT, tolerance_m=200.0)
    assert empty.empty
    assert {"mmsi", "status", "nearest_detection_m"} <= set(empty.columns)


def test_the_agreement_is_a_confusion_matrix_over_two_sensors():
    detections = gpd.GeoDataFrame(
        {"status": [MATCHED, MATCHED, DARK]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs=CRS,
    )
    declarations = gpd.GeoDataFrame(
        {"status": pd.array([EXPLAINED, EXPLAINED, UNDETECTED, OUTSIDE_SCENE], dtype="string")},
        geometry=[Point(0, 0)] * 4,
        crs=CRS,
    )
    found = agreement(detections, declarations)
    assert (found.both, found.radar_only, found.ais_only) == (2, 1, 1)
    assert found.apparent_recall == pytest.approx(2 / 3)


def test_recall_is_absent_rather_than_perfect_when_nothing_declared():
    """Zero out of zero is not 100% recall, and printing it as one would be the most
    flattering possible way to report having measured nothing."""
    nothing = Agreement(both=0, radar_only=3, ais_only=0, outside_scene=0, below_detectable=0)
    assert nothing.apparent_recall is None
    assert "no declared" in nothing.line()
