"""Attaching embeddings to detections, in row order."""

import geopandas as gpd
import numpy as np
from shapely import Point

from darkvessel.embed.embedder import attach


def _detections(count: int) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"status": ["dark"] * count},
        geometry=[Point(float(i), 0.0) for i in range(count)],
        crs="EPSG:25832",
    )


def test_each_detection_gets_the_embedding_at_its_own_row():
    detections = _detections(3)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = attach(detections, embeddings)
    assert np.array_equal(result["embedding"].iloc[1], [0.0, 1.0])


def test_attaching_does_not_mutate_the_input():
    detections = _detections(2)
    attach(detections, np.zeros((2, 4)))
    assert "embedding" not in detections.columns


def test_the_geometry_survives_attachment():
    result = attach(_detections(2), np.zeros((2, 4)))
    assert result.crs == "EPSG:25832"
    assert result.geometry.iloc[1] == Point(1.0, 0.0)
