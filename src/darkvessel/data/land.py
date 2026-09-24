"""Land, and a buffer off it: the part of a scene the detector is not asked to search.

Both detectors are poor inshore (`models/README.md`): canal banks, quays and bright urban
land read as hulls. A dark-vessel search is a search of open water, so land is masked before
detection rather than cleaned up after it. That also saves the work of running a detector
over it.

Masking changes what was searched, and the declaration side has to know. A vessel declared
at its berth, inside the mask, was never looked for; reporting it `undetected` would count the
mask against the detector. `fusion.declarations` reports it `masked` instead.

The land polygons come from any vector file geopandas can read (OSM land polygons, GSHHG, a
national coastline). Only the part near the scene is read. `buffer_m` pushes the mask out to
sea, which removes near-shore clutter and also any vessel moored within it: it is a policy
for a run configuration to state, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from affine import Affine
from pyproj import Transformer

from darkvessel.data.scene import Scene
from darkvessel.data.tiling import Tile


@dataclass(frozen=True)
class LandMask:
    """Land plus its buffer, as one geometry in the scene's CRS."""

    geometry: shapely.Geometry
    buffer_m: float = 0.0

    @classmethod
    def read(cls, path: str | Path, scene: Scene, buffer_m: float = 0.0) -> LandMask:
        """Read the land near `scene` from a vector file, reprojected and buffered."""
        import geopandas as gpd
        import pyogrio

        # Anything further than the buffer from the scene cannot reach into it.
        reach = scene.footprint.buffer(max(float(buffer_m), 0.0) + 1.0)
        file_crs = pyogrio.read_info(path).get("crs")
        bbox = reach.bounds
        if file_crs:
            bbox = Transformer.from_crs(scene.crs, file_crs, always_xy=True).transform_bounds(
                *bbox, densify_pts=21
            )
        land = gpd.read_file(path, bbox=bbox)
        if land.empty:
            return cls(geometry=shapely.Polygon(), buffer_m=float(buffer_m))
        if land.crs is None:
            raise ValueError(f"{path} has no CRS; the land mask cannot be placed on the scene")
        return cls.from_geometries(land.to_crs(scene.crs).geometry.values, reach, buffer_m)

    @classmethod
    def from_geometries(
        cls, geometries, within: shapely.Geometry | None = None, buffer_m: float = 0.0
    ) -> LandMask:
        """Union, clip to `within`, then buffer. Geometries must already be in the scene's CRS."""
        geometries = shapely.make_valid(np.asarray(geometries, dtype=object))
        if within is not None:
            geometries = shapely.intersection(geometries, within)
        land = shapely.union_all(geometries)
        if buffer_m > 0:
            land = land.buffer(float(buffer_m))
        return cls(geometry=land, buffer_m=float(buffer_m))

    @property
    def empty(self) -> bool:
        return self.geometry.is_empty

    def mask_for(self, scene: Scene):
        """A `detect.infer.Mask` for `scene`: True where a tile's pixel centre is on the mask."""
        prepared = self.geometry
        shapely.prepare(prepared)
        transform = scene.transform

        def mask(tile: Tile) -> np.ndarray | None:
            if self.empty:
                return None
            shape = (tile.row1 - tile.row0, tile.col1 - tile.col0)
            window = transform @ Affine.translation(tile.col0, tile.row0)
            box = _window_polygon(window, shape)
            if not shapely.intersects(prepared, box):
                return None
            if shapely.contains(prepared, box):
                return np.ones(shape, dtype=bool)
            from rasterio.features import rasterize

            burned = rasterize(
                [shapely.intersection(prepared, box)],
                out_shape=shape,
                transform=window,
                fill=0,
                default_value=1,
                dtype="uint8",
            )
            return burned.astype(bool)

        return mask


def _window_polygon(transform: Affine, shape: tuple[int, int]) -> shapely.Polygon:
    height, width = shape
    corners = ((0, 0), (width, 0), (width, height), (0, height))
    return shapely.Polygon([transform @ corner for corner in corners])
