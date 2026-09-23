"""The synthetic fixture's five stories, asserted rather than assumed.

Each target exists to exercise a different branch, and each only does its job if a specific
numeric relationship holds — the interpolated vessel's single reports must be *outside* any
sane tolerance, the azimuth vessel's declared position must be far enough off to be reported
dark uncorrected. These tests pin those relationships so a later edit to the fixture cannot
quietly turn five stories into five plain matches.
"""

from datetime import timedelta

import numpy as np
import pytest
from shapely import Point

from darkvessel.data.scene import read_scene
from darkvessel.data.synthetic import (
    _TARGETS,
    ACQUIRED_AT,
    CRS,
    HEADING_DEG,
    IMAGE_SHAPE,
    INCIDENCE_DEG,
    TOLERANCE_M,
    _ais,
    _by_name,
    _scene,
    write_synthetic,
)
from darkvessel.fusion.azimuth import Geometry
from darkvessel.fusion.interpolate import positions_at


def test_write_synthetic_produces_a_readable_scene_and_ais_archive(tmp_path):
    out = write_synthetic(tmp_path / "fixture")
    scene = read_scene(out / "scene")
    assert scene.image.shape == IMAGE_SHAPE
    assert scene.crs == CRS
    assert (out / "ais.gpkg").exists()


def test_the_scene_carries_one_blob_per_target():
    scene = _scene()
    bright = np.count_nonzero(scene.image >= 0.5)
    assert bright == 5 * 4  # five targets, each a 2x2 plateau


def test_every_target_blob_sits_at_its_declared_ground_position():
    scene = _scene()
    for target in [_by_name(n) for n in ("stationary_match", "dark_vessel")]:
        col, row = ~scene.transform * (target.x, target.y)
        assert scene.image[int(row), int(col)] >= 0.5


def test_the_dark_vessel_declares_nothing_of_its_own():
    """No AIS row belongs to the dark target. Of the story vessels, the default max gap
    declares exactly the other four; the two stale ones report too long ago. The `2191*`
    vessels exist only in the AIS and carry the declaration side's four verdicts."""
    declared = positions_at(_ais(), ACQUIRED_AT, timedelta(minutes=10))
    stories = {m for m in declared["mmsi"] if m.startswith("21900")}
    assert stories == {"219000001", "219000002", "219000003", "219000004"}
    assert set(_ais()["mmsi"]) == {f"21900000{i}" for i in range(1, 7)} | {
        f"21910000{i}" for i in range(1, 5)
    }


def test_the_two_lanes_give_the_reception_estimate_something_to_measure():
    """One lane reports every two minutes, the other once an hour. Without that contrast every
    cell would come back `insufficient_evidence` and the reception control would be inert."""
    ais = _ais()
    gaps = {}
    for mmsi in ("219100001", "219100002"):
        times = sorted(ais[ais["mmsi"] == mmsi]["timestamp"])
        gaps[mmsi] = [(b - a).total_seconds() for a, b in zip(times, times[1:])]

    assert len(gaps["219100001"]) >= 5, "too few intervals to clear the evidence floor"
    assert max(gaps["219100001"]) <= 2 * 600, "the busy lane must be fully covered at a 10 min gap"

    assert len(gaps["219100002"]) >= 5
    assert min(gaps["219100002"]) > 2 * 600, "the quiet lane must be a shadow at a 10 min gap"


def test_the_ais_only_vessels_keep_clear_of_every_target():
    """They exist to be *undetected*, so none of them may sit inside the default tolerance of a
    painted target — otherwise one would match and quietly become a fifth story."""
    declared = positions_at(_ais(), ACQUIRED_AT, timedelta(minutes=10))
    for _, row in declared.iterrows():
        if not str(row["mmsi"]).startswith("2191"):
            continue
        nearest = min(
            np.hypot(target.x - row.geometry.x, target.y - row.geometry.y) for target in _TARGETS
        )
        assert nearest > TOLERANCE_M, f"{row['mmsi']} is within tolerance of a target"


def test_the_extra_targets_are_below_the_default_threshold():
    """The faint targets exist for the viewer's threshold slider; at the default they must not
    change the quick start's numbers."""
    scene = _scene()
    for name in ("faint_trawler", "faint_dark"):
        target = _by_name(name)
        col, row = ~scene.transform * (target.x, target.y)
        assert 0.05 <= scene.image[int(row), int(col)] < 0.5


def test_nothing_but_the_targets_is_detectable_at_any_threshold_the_viewer_offers():
    scene = _scene()
    assert np.count_nonzero(scene.image >= 0.05) == len(_TARGETS) * 4


def test_the_stale_reports_are_declared_only_under_a_longer_max_gap():
    at_fifteen = positions_at(_ais(), ACQUIRED_AT, timedelta(minutes=15))
    at_thirty = positions_at(_ais(), ACQUIRED_AT, timedelta(minutes=30))
    assert "219000005" in set(at_fifteen["mmsi"]) and "219000006" not in set(at_fifteen["mmsi"])
    assert "219000006" in set(at_thirty["mmsi"])


def test_the_dark_vessel_stays_dark_only_because_of_one_to_one_matching():
    """A subtler property than "nothing is near it": `simple_match`'s declaration lies 170 m
    from the dark target, *inside* the 200 m tolerance. The dark vessel cannot claim it
    because the same declaration sits 30 m from `simple_match`'s own detection, and the
    assignment is one-to-one. A matcher that let a declaration explain two detections, or
    that claimed pairs in the wrong order, would hide this target."""
    ais = _ais()
    dark, simple = _by_name("dark_vessel"), _by_name("simple_match")
    contested = ais[ais["mmsi"] == "219000004"].geometry.iloc[0]
    assert contested.distance(Point(dark.x, dark.y)) < TOLERANCE_M
    assert contested.distance(Point(simple.x, simple.y)) < contested.distance(
        Point(dark.x, dark.y)
    )


def test_either_single_report_of_the_interpolated_vessel_is_outside_tolerance():
    """The README's failure case: matched against a report taken as it stands, this vessel
    comes back dark. Both bracketing reports must be far enough out for that to be true."""
    target = _by_name("interpolated_match")
    reports = _ais()[_ais()["mmsi"] == "219000002"]
    assert len(reports) == 2
    distances = reports.geometry.distance(Point(target.x, target.y))
    assert (distances > TOLERANCE_M).all()
    assert distances.min() == pytest.approx(900.0)


def test_interpolating_the_bracketing_reports_lands_on_the_target_exactly():
    target = _by_name("interpolated_match")
    declared = positions_at(_ais(), ACQUIRED_AT, timedelta(minutes=10))
    position = declared[declared["mmsi"] == "219000002"].geometry.iloc[0]
    assert position.distance(Point(target.x, target.y)) == pytest.approx(0.0, abs=1e-6)


def test_the_interpolated_vessel_has_no_azimuth_shift_of_its_own():
    """Its velocity is placed along the flight direction on purpose, so its story stays
    about interpolation alone, uncomplicated by the correction."""
    declared = positions_at(_ais(), ACQUIRED_AT, timedelta(minutes=10))
    row = declared[declared["mmsi"] == "219000002"].iloc[0]
    geometry = Geometry(heading_deg=HEADING_DEG, incidence_deg=INCIDENCE_DEG)
    east, north = geometry.displacement(
        row["velocity_east_ms"], row["velocity_north_ms"], latitude=55.7
    )
    assert np.hypot(east, north) == pytest.approx(0.0, abs=1e-6)


def test_the_azimuth_vessel_is_declared_outside_tolerance_before_correction():
    target = _by_name("azimuth_corrected_match")
    declared = positions_at(_ais(), ACQUIRED_AT, timedelta(minutes=10))
    raw = declared[declared["mmsi"] == "219000003"].geometry.iloc[0]
    assert raw.distance(Point(target.x, target.y)) > TOLERANCE_M


def test_the_azimuth_vessel_lands_on_its_detection_once_corrected():
    """The fixture computes its raw position by *inverting* `Geometry.displacement`, so this
    stays true even if the formula in `fusion/azimuth.py` is later changed."""
    target = _by_name("azimuth_corrected_match")
    declared = positions_at(_ais(), ACQUIRED_AT, timedelta(minutes=10))
    row = declared[declared["mmsi"] == "219000003"].iloc[0]
    geometry = Geometry(heading_deg=HEADING_DEG, incidence_deg=INCIDENCE_DEG)
    east, north = geometry.displacement(
        row["velocity_east_ms"], row["velocity_north_ms"], latitude=55.7
    )
    corrected = Point(row.geometry.x + east, row.geometry.y + north)
    assert corrected.distance(Point(target.x, target.y)) == pytest.approx(0.0, abs=1e-6)


def test_the_plain_matches_are_declared_within_tolerance():
    ais = _ais()
    for name, mmsi in (("stationary_match", "219000001"), ("simple_match", "219000004")):
        target = _by_name(name)
        report = ais[ais["mmsi"] == mmsi].geometry.iloc[0]
        assert report.distance(Point(target.x, target.y)) < TOLERANCE_M
