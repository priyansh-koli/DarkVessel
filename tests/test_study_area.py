"""The study area: bounds membership, inclusive at the edges."""

from darkvessel.data.study_area import StudyArea


def _area() -> StudyArea:
    return StudyArea(crs="EPSG:25832", minx=0.0, miny=0.0, maxx=100.0, maxy=100.0)


def test_a_point_inside_the_bounds_is_contained():
    assert _area().contains(50.0, 50.0)


def test_a_point_outside_the_bounds_is_not_contained():
    assert not _area().contains(150.0, 50.0)
    assert not _area().contains(50.0, -10.0)


def test_the_boundary_itself_counts_as_contained():
    area = _area()
    assert area.contains(0.0, 0.0)
    assert area.contains(100.0, 100.0)


def test_bounds_property_matches_the_constructor_order():
    assert _area().bounds == (0.0, 0.0, 100.0, 100.0)
