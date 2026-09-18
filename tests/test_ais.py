"""AIS cleaning rules, and the auditable per-rule counts they leave behind.

See `data/ais.py`'s module docstring: a silent cleaning rule that drops the one report that
would have explained a detection is indistinguishable from a genuinely dark vessel, so every
rule's removed-row count must be visible on the `CleaningReport`, not just the total.
"""

import geopandas as gpd
import pandas as pd
from shapely import Point

from darkvessel.data.ais import clean
from darkvessel.data.study_area import StudyArea


def _frame(rows: list[dict]) -> gpd.GeoDataFrame:
    frame = pd.DataFrame(rows)
    geometry = frame.pop("geometry") if "geometry" in frame.columns else None
    return gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:25832")


def test_rows_missing_mmsi_timestamp_or_position_are_removed():
    raw = _frame(
        [
            dict(mmsi="1", timestamp=pd.Timestamp("2026-01-01"), geometry=Point(0, 0)),
            dict(mmsi=None, timestamp=pd.Timestamp("2026-01-01"), geometry=Point(0, 0)),
            dict(mmsi="2", timestamp=None, geometry=Point(0, 0)),
            dict(mmsi="3", timestamp=pd.Timestamp("2026-01-01"), geometry=None),
        ]
    )
    cleaned, report = clean(raw)
    assert len(cleaned) == 1
    assert report.removed["missing_identity_time_or_position"] == 3


def test_duplicate_mmsi_timestamp_pairs_keep_only_the_first():
    raw = _frame(
        [
            dict(mmsi="1", timestamp=pd.Timestamp("2026-01-01"), geometry=Point(0, 0)),
            dict(mmsi="1", timestamp=pd.Timestamp("2026-01-01"), geometry=Point(1, 1)),
        ]
    )
    cleaned, report = clean(raw)
    assert len(cleaned) == 1
    assert cleaned.geometry.iloc[0] == Point(0, 0)
    assert report.removed["duplicate_mmsi_timestamp"] == 1


def test_implausibly_fast_reports_are_removed_when_speed_is_present():
    raw = _frame(
        [
            dict(
                mmsi="1", timestamp=pd.Timestamp("2026-01-01"), geometry=Point(0, 0), speed_ms=10.0
            ),
            dict(
                mmsi="2", timestamp=pd.Timestamp("2026-01-01"), geometry=Point(0, 0), speed_ms=100.0
            ),
        ]
    )
    cleaned, report = clean(raw)
    assert cleaned["mmsi"].tolist() == ["1"]
    assert report.removed["implausible_speed"] == 1


def test_speed_rule_is_absent_from_the_report_when_no_speed_column_exists():
    raw = _frame([dict(mmsi="1", timestamp=pd.Timestamp("2026-01-01"), geometry=Point(0, 0))])
    _, report = clean(raw)
    assert "implausible_speed" not in report.removed


def test_a_missing_speed_value_is_not_treated_as_implausible():
    raw = _frame(
        [dict(mmsi="1", timestamp=pd.Timestamp("2026-01-01"), geometry=Point(0, 0), speed_ms=None)]
    )
    cleaned, report = clean(raw)
    assert len(cleaned) == 1
    assert report.removed["implausible_speed"] == 0


def test_rows_outside_the_study_area_are_removed_when_an_area_is_given():
    raw = _frame(
        [
            dict(mmsi="1", timestamp=pd.Timestamp("2026-01-01"), geometry=Point(50, 50)),
            dict(mmsi="2", timestamp=pd.Timestamp("2026-01-01"), geometry=Point(500, 500)),
        ]
    )
    area = StudyArea(crs="EPSG:25832", minx=0.0, miny=0.0, maxx=100.0, maxy=100.0)
    cleaned, report = clean(raw, area=area)
    assert cleaned["mmsi"].tolist() == ["1"]
    assert report.removed["outside_study_area"] == 1


def test_no_area_given_means_no_area_rule_runs():
    raw = _frame([dict(mmsi="1", timestamp=pd.Timestamp("2026-01-01"), geometry=Point(500, 500))])
    _, report = clean(raw)
    assert "outside_study_area" not in report.removed


def test_report_line_summarises_totals():
    raw = _frame(
        [
            dict(mmsi="1", timestamp=pd.Timestamp("2026-01-01"), geometry=Point(0, 0)),
            dict(mmsi=None, timestamp=pd.Timestamp("2026-01-01"), geometry=Point(0, 0)),
        ]
    )
    _, report = clean(raw)
    line = report.line()
    assert line.startswith("2 rows in, 1 removed")
    assert "1 kept" in line
