"""Attach the provenance every fixed-structure claim rests on: which scene, and where."""

import geopandas as gpd

from darkvessel.data.scene import Scene


def attach_provenance(detections: gpd.GeoDataFrame, scene: Scene) -> gpd.GeoDataFrame:
    """Record which scene each detection came from, and its ground coordinates.

    `scene` and `x`/`y` are exactly what `embed.structures.standing` groups by — see there for
    why recurrence needs nothing else.
    """
    attached = detections.copy()
    attached["scene"] = scene.id
    attached["x"] = attached.geometry.x
    attached["y"] = attached.geometry.y
    return attached
