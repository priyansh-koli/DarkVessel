"""The reception estimate, checked against its own arithmetic.

`fusion.reception` asks one question: had a transmitting vessel been here at the acquisition
instant, would this archive have placed it? The answer is the share of a track's observed time
that lies within `max_gap` of some report, so every test here is a track whose gaps make that
share a number that can be written down.
"""

from datetime import datetime, timedelta, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely import Point

from darkvessel.fusion.match import DARK, MATCHED
from darkvessel.fusion.reception import (
    ESTIMATED,
    INSUFFICIENT,
    SHADOWED,
    Coverage,
    coverage_for,
    without_coverage,
)

CRS = "EPSG:25832"
START = datetime(2026, 8, 9, 5, 0, 0, tzinfo=timezone.utc)


def _track(mmsi: str, x: float, y: float, gap_s: float, reports: int) -> list:
    """A vessel sitting still at (x, y), reporting every `gap_s` seconds."""
    return [
        {
            "mmsi": mmsi,
            "timestamp": START + timedelta(seconds=gap_s * i),
            "length_m": 50.0,
            "geometry": Point(x, y),
        }
        for i in range(reports)
    ]


def _archive(*tracks) -> gpd.GeoDataFrame:
    rows = [row for track in tracks for row in track]
    if not rows:
        return gpd.GeoDataFrame(
            {"mmsi": pd.array([], dtype="string"), "timestamp": [], "length_m": []},
            geometry=[],
            crs=CRS,
        )
    frame = pd.DataFrame(rows)
    frame["mmsi"] = frame["mmsi"].astype("string")
    return gpd.GeoDataFrame(frame.drop(columns=["geometry"]), geometry=frame["geometry"], crs=CRS)


def _detections(coords: list) -> gpd.GeoDataFrame:
    frame = gpd.GeoDataFrame(
        {"status": [DARK] * len(coords)}, geometry=[Point(x, y) for x, y in coords], crs=CRS
    )
    return frame


def test_reports_closer_together_than_the_window_cover_all_of_it():
    """Every gap is under 2 x max_gap, so no instant in the track is ever more than max_gap
    from a report: reception is exactly 1."""
    coverage = Coverage.from_archive(
        _archive(_track("1", 0, 0, gap_s=60, reports=10)), timedelta(minutes=10), cell_m=1000
    )
    probability, intervals = coverage.at(np.array([0.0]), np.array([0.0]))
    assert probability[0] == pytest.approx(1.0)
    assert intervals[0] == 9


def test_a_gap_wider_than_the_window_is_covered_only_at_its_ends():
    """A 60-minute gap with a 10-minute window is covered for 10 minutes at each end and not
    at all in between: 20/60."""
    coverage = Coverage.from_archive(
        _archive(_track("1", 0, 0, gap_s=3600, reports=10)), timedelta(minutes=10), cell_m=1000
    )
    probability, _ = coverage.at(np.array([0.0]), np.array([0.0]))
    assert probability[0] == pytest.approx(1 / 3)


def test_widening_the_window_raises_reception_on_the_same_reports():
    """Reception is defined against `max_gap`, so the pipeline's own control moves it. Nothing
    about the archive changed between these two."""
    archive = _archive(_track("1", 0, 0, gap_s=3600, reports=10))
    third = Coverage.from_archive(archive, timedelta(minutes=10), cell_m=1000)
    whole = Coverage.from_archive(archive, timedelta(minutes=30), cell_m=1000)
    assert third.at(np.array([0.0]), np.array([0.0]))[0][0] == pytest.approx(1 / 3)
    assert whole.at(np.array([0.0]), np.array([0.0]))[0][0] == pytest.approx(1.0)


def test_a_neighbourhood_with_too_little_evidence_reports_nothing_at_all():
    """Two reports say almost nothing about whether a second vessel would have been heard.
    An estimate nothing supports is worse than none, because it reads like one that does."""
    coverage = Coverage.from_archive(
        _archive(_track("1", 0, 0, gap_s=60, reports=3)),
        timedelta(minutes=10),
        cell_m=1000,
        min_intervals=5,
    )
    probability, intervals = coverage.at(np.array([0.0]), np.array([0.0]))
    assert np.isnan(probability[0])
    assert intervals[0] == 2  # the evidence is still reported, just not an estimate from it


def test_evidence_is_pooled_over_a_cell_and_the_eight_around_it():
    """Otherwise an estimate would be an artefact of which side of a grid line a detection
    fell on. A neighbouring cell counts; one two cells away does not."""
    coverage = Coverage.from_archive(
        _archive(_track("1", 500.0, 500.0, gap_s=60, reports=10)),
        timedelta(minutes=10),
        cell_m=1000,
    )
    x = np.array([500.0, 1500.0, 2500.0])
    y = np.array([500.0, 500.0, 500.0])
    probability, intervals = coverage.at(x, y)
    assert intervals.tolist() == [9, 9, 0]
    assert np.isnan(probability[2])


def test_a_gap_that_implies_an_impossible_speed_is_dropped_and_counted():
    """Above the plausible-speed cap the vessel did not travel that straight line, so its
    midpoint is not a place the gap can be attributed to. The same cap `data.ais.clean`
    applies to a *reported* speed, applied here to an implied one."""
    teleport = [
        {"mmsi": "1", "timestamp": START, "length_m": 50.0, "geometry": Point(0, 0)},
        {
            "mmsi": "1",
            "timestamp": START + timedelta(seconds=10),
            "length_m": 50.0,
            "geometry": Point(100_000, 0),
        },
    ]
    coverage = Coverage.from_archive(_archive(teleport), timedelta(minutes=10), cell_m=1000)
    assert coverage.report.dropped["implausible_speed"] == 1
    assert coverage.report.used == 0


def test_reception_goes_on_every_row_but_only_dark_rows_change_status():
    """Reception describes the place, so a match made where the archive barely reaches is
    worth knowing about too — it just is not reclassified."""
    coverage = Coverage.from_archive(
        _archive(_track("1", 0, 0, gap_s=3600, reports=10)), timedelta(minutes=10), cell_m=1000
    )
    detections = _detections([(0, 0), (10, 10)])
    detections.loc[1, "status"] = MATCHED

    marked = coverage.mark(detections, floor=0.5)
    assert marked["status"].tolist() == [SHADOWED, MATCHED]
    assert marked["reception_p"].tolist() == pytest.approx([1 / 3, 1 / 3])
    assert set(marked["reception_basis"]) == {ESTIMATED}


def test_a_missing_estimate_never_shadows_a_detection():
    """The whole point of the `unsearched` status, pointed the other way: reclassifying on an
    absence of evidence is the same mistake as reporting a dark vessel on one."""
    coverage = Coverage.from_archive(
        _archive(_track("1", 0, 0, gap_s=3600, reports=10)), timedelta(minutes=10), cell_m=1000
    )
    far_away = _detections([(500_000, 500_000)])
    marked = coverage.mark(far_away, floor=1.0)
    assert marked["status"].tolist() == [DARK]
    assert marked["reception_basis"].tolist() == [INSUFFICIENT]
    assert marked["reception_p"].isna().all()


def test_a_floor_of_zero_reclassifies_nothing():
    """The floor is a policy, and the library's default is to state reception without acting
    on it. A run configuration decides where the bar sits."""
    coverage = Coverage.from_archive(
        _archive(_track("1", 0, 0, gap_s=3600, reports=10)), timedelta(minutes=10), cell_m=1000
    )
    marked = coverage.mark(_detections([(0, 0)]), floor=0.0)
    assert marked["status"].tolist() == [DARK]
    assert marked["reception_p"].tolist() == pytest.approx([1 / 3])


def test_southern_and_northern_cells_pack_and_unpack_alike():
    """Cell indices are packed into one int64 for a binary-search lookup. Negative indices are
    what a CRS south or west of its origin produces, and they must not fold onto each other."""
    coverage = Coverage.from_archive(
        _archive(
            _track("1", -5000.0, -5000.0, gap_s=60, reports=10),
            _track("2", 5000.0, 5000.0, gap_s=3600, reports=10),
        ),
        timedelta(minutes=10),
        cell_m=1000,
    )
    probability, _ = coverage.at(np.array([-5000.0, 5000.0]), np.array([-5000.0, 5000.0]))
    assert probability.tolist() == pytest.approx([1.0, 1 / 3])

    cells = coverage.cells()
    assert (cells["x0"] < 0).any() and (cells["x0"] > 0).any()


def test_without_an_archive_there_is_no_model_and_the_columns_are_still_there():
    """A column's presence must never depend on whether an optional stage ran — the same rule
    `context.schema` and `fusion.register` follow."""
    assert coverage_for(None, timedelta(minutes=10)) is None
    assert coverage_for(_archive([]), timedelta(minutes=10)) is None

    filled = without_coverage(_detections([(0, 0)]))
    assert filled["reception_p"].isna().all()
    assert filled["reception_basis"].isna().all()
    assert filled["status"].tolist() == [DARK]


def test_the_report_accounts_for_every_interval_it_was_given():
    """A dark claim rests on this, so the counts have to add up rather than nearly add up."""
    coverage = Coverage.from_archive(
        _archive(
            _track("1", 0, 0, gap_s=60, reports=10),
            _track("2", 3000, 3000, gap_s=60, reports=5),
        ),
        timedelta(minutes=10),
        cell_m=1000,
    )
    report = coverage.report
    assert report.intervals == report.used + sum(report.dropped.values())
    assert report.dropped["not_a_pair"] == 1  # one boundary between the two vessels
    assert report.used == 9 + 4
    assert "at a max gap of 10 min" in report.line()
