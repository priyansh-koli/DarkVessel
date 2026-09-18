"""The structure register: reclassifying dark detections that recur at a fixed position.

Structure exclusion must happen strictly after matching — a detection AIS already explains is
never overridden, whatever else stands at that coordinate. See `fusion/register.py`'s
module docstring.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Point

from darkvessel.embed.structures import Standing
from darkvessel.fusion.match import DARK, MATCHED, UNSEARCHED
from darkvessel.fusion.register import STRUCTURE, Register, without_a_register


def _detections(rows: list[tuple[str, float, float]]) -> gpd.GeoDataFrame:
    statuses, xs, ys = zip(*rows)
    return gpd.GeoDataFrame(
        {"status": list(statuses)}, geometry=[Point(x, y) for x, y in zip(xs, ys)], crs="EPSG:25832"
    )


def test_dark_detection_at_a_registered_position_becomes_a_structure():
    positions = pd.DataFrame({"x": [0.0], "y": [0.0], "acquisitions": [5], "crops": [5]})
    register = Register(positions=positions, tolerance_m=50.0)
    result = register.mark(_detections([(DARK, 10.0, 0.0)]))
    assert result["status"].tolist() == [STRUCTURE]


def test_dark_detection_far_from_any_registered_position_stays_dark():
    positions = pd.DataFrame({"x": [0.0], "y": [0.0], "acquisitions": [5], "crops": [5]})
    register = Register(positions=positions, tolerance_m=50.0)
    result = register.mark(_detections([(DARK, 500.0, 0.0)]))
    assert result["status"].tolist() == [DARK]


def test_matched_and_unsearched_rows_are_never_touched():
    """A match AIS already explains is a match whatever else stands at that coordinate."""
    positions = pd.DataFrame({"x": [0.0], "y": [0.0], "acquisitions": [5], "crops": [5]})
    register = Register(positions=positions, tolerance_m=50.0)
    result = register.mark(_detections([(MATCHED, 0.0, 0.0), (UNSEARCHED, 0.0, 0.0)]))
    assert result["status"].tolist() == [MATCHED, UNSEARCHED]


def test_empty_register_leaves_detections_unchanged():
    register = Register(positions=pd.DataFrame({"x": [], "y": [], "acquisitions": [], "crops": []}))
    result = register.mark(_detections([(DARK, 0.0, 0.0)]))
    assert result["status"].tolist() == [DARK]


def test_from_standing_applies_the_recurrence_floor():
    positions = pd.DataFrame(
        {"x": [0.0, 500.0], "y": [0.0, 0.0], "acquisitions": [1, 3], "crops": [1, 3]}
    )
    standing = Standing(positions=positions, of_crop=np.zeros(0, dtype="int64"))
    register = Register.from_standing(standing, floor=2, tolerance_m=50.0)
    assert register.positions["x"].tolist() == [500.0]  # only the acquisitions>=2 position survives


def test_without_a_register_is_an_unchanged_copy():
    detections = _detections([(DARK, 0.0, 0.0)])
    result = without_a_register(detections)
    assert result["status"].tolist() == [DARK]
    assert result is not detections
