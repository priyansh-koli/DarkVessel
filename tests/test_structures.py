"""Structure-recurrence tests, including the case where greedy grouping is order dependent."""

import itertools

import pandas as pd
import pytest

from darkvessel.embed.structures import standing, verify


def _provenance(names: list[str], coords: dict, scenes: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [coords[n][0] for n in names],
            "y": [coords[n][1] for n in names],
            "scene": [scenes[n] for n in names],
        }
    )


def test_grouping_is_independent_of_row_order():
    """The case that breaks a greedy seed-and-claim grouping.

    A-B = 90 m and B-C = 90 m are both within a 100 m tolerance, but A-C = 180 m is not.
    A greedy seed-and-claim grouping returns two groups for order A,B,C and one group for
    order B,A,C. Connected components return the same single group for every ordering.
    """
    coords = {"A": (0.0, 0.0), "B": (90.0, 0.0), "C": (180.0, 0.0)}
    scenes = {"A": "scene1", "B": "scene2", "C": "scene3"}

    answers = set()
    for order in itertools.permutations(["A", "B", "C"]):
        result = standing(_provenance(list(order), coords, scenes), tolerance_m=100.0)
        answers.add((len(result.positions), int(result.positions["acquisitions"].max())))

    assert answers == {(1, 3)}, f"grouping varied with row order: {answers}"


def test_well_separated_positions_stay_distinct():
    coords = {"A": (0.0, 0.0), "B": (600.0, 0.0)}
    scenes = {"A": "scene1", "B": "scene1"}
    result = standing(_provenance(["A", "B"], coords, scenes), tolerance_m=100.0)
    assert len(result.positions) == 2


def test_repeat_sightings_of_one_mast_count_distinct_acquisitions():
    """Three sightings of one position across two scenes is two acquisitions, not three."""
    coords = {"A": (0.0, 0.0), "B": (5.0, 0.0), "C": (3.0, 2.0)}
    scenes = {"A": "scene1", "B": "scene2", "C": "scene2"}
    result = standing(_provenance(["A", "B", "C"], coords, scenes), tolerance_m=100.0)
    assert len(result.positions) == 1
    assert result.positions["acquisitions"].iloc[0] == 2
    assert result.positions["crops"].iloc[0] == 3


def test_a_position_is_the_centroid_of_its_sightings():
    coords = {"A": (0.0, 0.0), "B": (10.0, 0.0)}
    scenes = {"A": "scene1", "B": "scene2"}
    result = standing(_provenance(["A", "B"], coords, scenes), tolerance_m=100.0)
    assert result.positions["x"].iloc[0] == pytest.approx(5.0)


def test_seen_in_filters_by_acquisition_floor():
    coords = {"A": (0.0, 0.0), "B": (5.0, 0.0), "C": (900.0, 0.0)}
    scenes = {"A": "s1", "B": "s2", "C": "s1"}
    result = standing(_provenance(["A", "B", "C"], coords, scenes), tolerance_m=100.0)
    assert len(result.seen_in(2)) == 1  # only the recurring position clears a floor of 2


def test_empty_archive_gives_an_empty_answer_of_the_right_shape():
    result = standing(pd.DataFrame({"x": [], "y": [], "scene": []}))
    assert result.positions.empty
    assert list(result.positions.columns) == ["x", "y", "acquisitions", "crops"]


def test_verify_reports_both_directions():
    registered = pd.DataFrame({"x": [0.0, 1000.0], "y": [0.0, 0.0]})
    known = pd.DataFrame({"x": [2.0], "y": [0.0]})
    result = verify(registered, known, tolerance_m=10.0)
    assert result.found == 1  # the published position is covered
    assert result.unpublished == 1  # the 1000 m one matches nothing published


def test_verify_refuses_an_empty_reference():
    with pytest.raises(ValueError):
        verify(pd.DataFrame({"x": [0.0], "y": [0.0]}), pd.DataFrame({"x": [], "y": []}), 10.0)


def test_verify_of_an_empty_register_is_a_number_not_an_exception():
    result = verify(pd.DataFrame({"x": [], "y": []}), pd.DataFrame({"x": [0.0], "y": [0.0]}), 10.0)
    assert result.found == 0
    assert result.median_m == float("inf")  # never 0.0, which would read as perfect agreement
