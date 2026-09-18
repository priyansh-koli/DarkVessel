"""Azimuth-shift correction: velocity along the line of sight draws a target off, velocity
along the flight track does not."""

from math import hypot

import pytest

from darkvessel.fusion.azimuth import Geometry


def test_zero_velocity_causes_zero_shift():
    geometry = Geometry(heading_deg=350.0, incidence_deg=35.0)
    east, north = geometry.displacement(0.0, 0.0, latitude=55.7)
    assert (east, north) == (0.0, 0.0)


def test_velocity_along_the_flight_track_has_no_line_of_sight_component():
    """A vessel moving exactly along the satellite's ground track has no velocity component
    towards the radar, so it is drawn at its true position — this is why the synthetic
    fixture's interpolated-match vessel (velocity along the flight direction) has an
    azimuth shift of exactly zero, keeping its story about interpolation alone."""
    from math import cos, radians, sin

    heading_deg = 350.0
    heading = radians(heading_deg)
    flight_east, flight_north = sin(heading), cos(heading)

    geometry = Geometry(heading_deg=heading_deg, incidence_deg=35.0)
    east, north = geometry.displacement(flight_east * 8.0, flight_north * 8.0, latitude=55.7)
    assert east == pytest.approx(0.0, abs=1e-9)
    assert north == pytest.approx(0.0, abs=1e-9)


def test_velocity_purely_towards_the_radar_shifts_purely_along_track():
    """With heading due north (0 deg) and incidence 90 deg, a purely eastward (line-of-sight)
    velocity should displace the target purely northward (along-track) by shift_per_mps."""
    geometry = Geometry(heading_deg=0.0, incidence_deg=90.0, shift_per_mps=113.0)
    east, north = geometry.displacement(velocity_east_ms=1.0, velocity_north_ms=0.0, latitude=55.7)
    assert east == pytest.approx(0.0, abs=1e-9)
    assert north == pytest.approx(113.0)


def test_shift_magnitude_scales_linearly_with_velocity():
    geometry = Geometry(heading_deg=0.0, incidence_deg=90.0, shift_per_mps=113.0)
    east1, north1 = geometry.displacement(2.0, 0.0, latitude=55.7)
    east2, north2 = geometry.displacement(4.0, 0.0, latitude=55.7)
    assert hypot(east2, north2) == pytest.approx(2 * hypot(east1, north1))


def test_incidence_of_zero_gives_no_shift_regardless_of_velocity():
    """At normal incidence there is no line-of-sight component of a horizontal velocity."""
    geometry = Geometry(heading_deg=0.0, incidence_deg=0.0)
    east, north = geometry.displacement(10.0, 10.0, latitude=55.7)
    assert east == pytest.approx(0.0, abs=1e-9)
    assert north == pytest.approx(0.0, abs=1e-9)
