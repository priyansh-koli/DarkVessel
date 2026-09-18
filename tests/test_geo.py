"""Pixel space to ground, via the scene's affine transform."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from affine import Affine

from darkvessel.data.scene import Scene
from darkvessel.detect.geo import to_ground

_TRANSFORM = Affine(5.0, 0.0, 500000.0, 0.0, -5.0, 6101000.0)


def _scene() -> Scene:
    return Scene(
        id="scene-1",
        image=np.zeros((100, 100), dtype=np.float32),
        transform=_TRANSFORM,
        crs="EPSG:25832",
        acquired_at=datetime(2026, 8, 9, 5, 31, 24, tzinfo=timezone.utc),
        heading_deg=350.0,
        incidence_deg=35.0,
    )


def test_a_detection_lands_at_its_pixel_centre_not_its_corner():
    """Pixel (0,0) covers ground x 500000..500005; its centre is 500002.5, not the origin."""
    result = to_ground(pd.DataFrame({"row": [0.0], "col": [0.0]}), _scene())
    assert result.geometry.iloc[0].x == pytest.approx(500002.5)
    assert result.geometry.iloc[0].y == pytest.approx(6100997.5)


def test_row_increases_southward_and_col_eastward():
    result = to_ground(pd.DataFrame({"row": [0.0, 10.0], "col": [0.0, 20.0]}), _scene())
    assert result.geometry.iloc[1].x > result.geometry.iloc[0].x
    assert result.geometry.iloc[1].y < result.geometry.iloc[0].y


def test_the_pixel_position_is_carried_through_alongside_the_geometry():
    result = to_ground(pd.DataFrame({"row": [3.0], "col": [7.0]}), _scene())
    assert result["row"].tolist() == [3.0]
    assert result["col"].tolist() == [7.0]


def test_the_scene_crs_is_applied():
    result = to_ground(pd.DataFrame({"row": [0.0], "col": [0.0]}), _scene())
    assert result.crs == "EPSG:25832"


def test_no_detections_gives_an_empty_georeferenced_frame():
    result = to_ground(pd.DataFrame({"row": [], "col": []}), _scene())
    assert result.empty
    assert result.crs == "EPSG:25832"
