"""Pixel space to ground: turn detector output into georeferenced points."""

import geopandas as gpd
import pandas as pd
from shapely import Point

from darkvessel.data.scene import Scene


def to_ground(found: pd.DataFrame, scene: Scene) -> gpd.GeoDataFrame:
    """Place each (row, col) detection on the ground using the scene's pixel->ground transform."""
    points = [
        scene.transform * (col + 0.5, row + 0.5)
        for row, col in zip(found["row"], found["col"])
    ]
    return gpd.GeoDataFrame(
        {"row": found["row"].to_numpy(), "col": found["col"].to_numpy()},
        geometry=[Point(x, y) for x, y in points],
        crs=scene.crs,
    )
