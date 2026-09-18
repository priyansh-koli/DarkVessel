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
    """No AIS row belongs to the dark target: the four declared MMSIs are the other four."""
    assert set(_ais()["mmsi"]) == {"219000001", "219000002", "219000003", "219000004"}


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
