"""AIS-to-acquisition-instant interpolation, including the never-extrapolate rule."""

from datetime import timedelta

import geopandas as gpd
import pandas as pd
import pytest
from shapely import Point

from darkvessel.fusion.interpolate import INTERPOLATED, NEAREST, positions_at


def _ais(rows: list[dict]) -> gpd.GeoDataFrame:
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["mmsi"] = frame["mmsi"].astype("string")
    return gpd.GeoDataFrame(
        frame.drop(columns=["geometry"]), geometry=frame["geometry"], crs="EPSG:25832"
    )


def test_bracketed_report_interpolates_by_time_fraction():
    """A report 4 minutes before and one 6 minutes after a 10-minute span: 40% of the way."""
    ais = _ais(
        [
            dict(mmsi="1", timestamp="2026-01-01T00:00:00Z", length_m=50.0, geometry=Point(0, 0)),
            dict(mmsi="1", timestamp="2026-01-01T00:10:00Z", length_m=50.0, geometry=Point(100, 0)),
        ]
    )
    result = positions_at(ais, pd.Timestamp("2026-01-01T00:04:00Z"), timedelta(minutes=30))
    assert result.geometry.iloc[0].x == pytest.approx(40.0)
    assert result["position_basis"].iloc[0] == INTERPOLATED
    assert result["position_age_s"].iloc[0] == 0.0


def test_bracketed_report_records_velocity():
    ais = _ais(
        [
            dict(mmsi="1", timestamp="2026-01-01T00:00:00Z", length_m=50.0, geometry=Point(0, 0)),
            dict(mmsi="1", timestamp="2026-01-01T00:10:00Z", length_m=50.0, geometry=Point(600, 0)),
        ]
    )
    result = positions_at(ais, pd.Timestamp("2026-01-01T00:05:00Z"), timedelta(minutes=30))
    assert result["velocity_east_ms"].iloc[0] == pytest.approx(1.0)  # 600 m / 600 s
    assert result["velocity_north_ms"].iloc[0] == pytest.approx(0.0)


def test_acquisition_outside_track_falls_back_to_nearest_report():
    """Nothing is ever extrapolated past the end of a track — the nearest report is used
    instead, and the row says so via `position_basis`."""
    ais = _ais(
        [
            dict(mmsi="1", timestamp="2026-01-01T00:00:00Z", length_m=50.0, geometry=Point(0, 0)),
            dict(mmsi="1", timestamp="2026-01-01T00:05:00Z", length_m=50.0, geometry=Point(50, 0)),
        ]
    )
    result = positions_at(ais, pd.Timestamp("2026-01-01T00:20:00Z"), timedelta(minutes=30))
    assert result["position_basis"].iloc[0] == NEAREST
    assert result.geometry.iloc[0].x == 50.0
    assert result["position_age_s"].iloc[0] == pytest.approx(900.0)  # 15 minutes


def test_nearest_report_still_derives_velocity_from_its_own_history():
    ais = _ais(
        [
            dict(mmsi="1", timestamp="2026-01-01T00:00:00Z", length_m=50.0, geometry=Point(0, 0)),
            dict(mmsi="1", timestamp="2026-01-01T00:10:00Z", length_m=50.0, geometry=Point(600, 0)),
        ]
    )
    result = positions_at(ais, pd.Timestamp("2026-01-01T00:30:00Z"), timedelta(minutes=60))
    assert result["position_basis"].iloc[0] == NEAREST
    assert result["velocity_east_ms"].iloc[0] == pytest.approx(1.0)


def test_single_report_with_no_history_has_no_velocity():
    ais = _ais(
        [dict(mmsi="1", timestamp="2026-01-01T00:00:00Z", length_m=50.0, geometry=Point(0, 0))]
    )
    result = positions_at(ais, pd.Timestamp("2026-01-01T00:01:00Z"), timedelta(minutes=30))
    assert pd.isna(result["velocity_east_ms"].iloc[0])
    assert pd.isna(result["velocity_north_ms"].iloc[0])


def test_report_older_than_max_gap_is_dropped_not_returned_stale():
    ais = _ais(
        [dict(mmsi="1", timestamp="2026-01-01T00:00:00Z", length_m=50.0, geometry=Point(0, 0))]
    )
    result = positions_at(ais, pd.Timestamp("2026-01-01T01:00:00Z"), timedelta(minutes=30))
    assert result.empty


def test_each_mmsi_is_resolved_independently():
    ais = _ais(
        [
            dict(mmsi="1", timestamp="2026-01-01T00:00:00Z", length_m=50.0, geometry=Point(0, 0)),
            dict(mmsi="2", timestamp="2026-01-01T00:00:00Z", length_m=60.0, geometry=Point(100, 0)),
        ]
    )
    result = positions_at(ais, pd.Timestamp("2026-01-01T00:00:00Z"), timedelta(minutes=30))
    assert set(result["mmsi"]) == {"1", "2"}


def test_empty_ais_gives_an_empty_answer_with_the_right_columns():
    empty = gpd.GeoDataFrame(
        {"mmsi": [], "timestamp": [], "length_m": []}, geometry=[], crs="EPSG:25832"
    )
    result = positions_at(empty, pd.Timestamp("2026-01-01T00:00:00Z"), timedelta(minutes=30))
    assert result.empty
    assert result.crs == "EPSG:25832"
    assert set(result.columns) >= {"mmsi", "length_m", "position_basis", "position_age_s"}


def test_rows_come_back_in_first_appearance_order_whatever_the_report_order():
    """Reports arrive interleaved and out of time order; each vessel is still resolved from its
    own sorted track, and vessels keep the order they first appear in."""
    ais = _ais(
        [
            dict(mmsi="b", timestamp="2026-01-01T00:10:00Z", length_m=60.0, geometry=Point(600, 0)),
            dict(mmsi="a", timestamp="2026-01-01T00:10:00Z", length_m=50.0, geometry=Point(0, 100)),
            dict(mmsi="b", timestamp="2026-01-01T00:00:00Z", length_m=60.0, geometry=Point(0, 0)),
            dict(mmsi="a", timestamp="2026-01-01T00:00:00Z", length_m=50.0, geometry=Point(0, 0)),
        ]
    )
    result = positions_at(ais, pd.Timestamp("2026-01-01T00:05:00Z"), timedelta(minutes=30))
    assert list(result["mmsi"]) == ["b", "a"]
    assert result.geometry.iloc[0].x == pytest.approx(300.0)
    assert result.geometry.iloc[1].y == pytest.approx(50.0)
    assert list(result.index) == [0, 1]


def test_reports_without_an_identity_are_skipped_and_do_not_shift_other_vessels():
    """A vessel whose only report has no MMSI must vanish without misaligning the others."""
    ais = _ais(
        [
            dict(mmsi="1", timestamp="2026-01-01T00:00:00Z", length_m=50.0, geometry=Point(0, 0)),
            dict(mmsi=None, timestamp="2026-01-01T00:00:00Z", length_m=9.0, geometry=Point(5, 5)),
            dict(mmsi="2", timestamp="2026-01-01T00:00:00Z", length_m=70.0, geometry=Point(9, 0)),
            dict(mmsi="2", timestamp="2026-01-01T00:10:00Z", length_m=70.0, geometry=Point(9, 60)),
        ]
    )
    result = positions_at(ais, pd.Timestamp("2026-01-01T00:05:00Z"), timedelta(minutes=30))
    assert list(result["mmsi"]) == ["1", "2"]
    assert list(result["position_basis"]) == [NEAREST, INTERPOLATED]
    assert result.geometry.iloc[1].y == pytest.approx(30.0)
    assert list(result["length_m"]) == [50.0, 70.0]


def test_a_naive_acquisition_time_against_a_utc_archive_is_refused():
    ais = _ais(
        [dict(mmsi="1", timestamp="2026-01-01T00:00:00Z", length_m=50.0, geometry=Point(0, 0))]
    )
    with pytest.raises(TypeError):
        positions_at(ais, pd.Timestamp("2026-01-01T00:00:00"), timedelta(minutes=30))
