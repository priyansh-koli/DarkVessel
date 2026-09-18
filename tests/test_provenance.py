"""Provenance attachment: exactly what `embed.structures.standing` needs to group by."""

from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely import Point

from darkvessel.data.provenance import attach_provenance
from darkvessel.data.scene import Scene


def _scene(scene_id: str = "scene-1") -> Scene:
    return Scene(
        id=scene_id,
        image=np.zeros((2, 2), dtype=np.float32),
        transform=Affine.identity(),
        crs="EPSG:25832",
        acquired_at=datetime(2026, 8, 9, 5, 31, 24, tzinfo=timezone.utc),
        heading_deg=350.0,
        incidence_deg=35.0,
    )


def test_attaches_scene_id_and_ground_coordinates():
    detections = gpd.GeoDataFrame(
        {"status": ["dark"]}, geometry=[Point(10.0, 20.0)], crs="EPSG:25832"
    )
    result = attach_provenance(detections, _scene("scene-42"))
    assert result["scene"].tolist() == ["scene-42"]
    assert result["x"].tolist() == [10.0]
    assert result["y"].tolist() == [20.0]


def test_does_not_mutate_the_input():
    detections = gpd.GeoDataFrame(
        {"status": ["dark"]}, geometry=[Point(0.0, 0.0)], crs="EPSG:25832"
    )
    attach_provenance(detections, _scene())
    assert "scene" not in detections.columns
