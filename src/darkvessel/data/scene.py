"""Scene loading: pixel-space SAR imagery paired with the geometry to place it on the ground.

A scene is a directory holding `scene.json` (identity, acquisition time, viewing geometry) and
one image. The image is never read whole unless something asks for all of it: a full
Sentinel-1 IW scene is some 25,000 x 17,000 pixels, 1.7 GB as float32, and detection only
ever needs one tile at a time.

- `image.npy` (the default, and what `write_scene` writes) is opened as a memory map.
- A GeoTIFF, named by `"image"` in `scene.json`, is read window by window through rasterio.
  Its transform and CRS come from the file unless `scene.json` states them. Its no-data value
  and any non-finite pixel are read as zero, the value every detector treats as outside the
  radar swath.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
from affine import Affine
from shapely import Polygon

_GEOTIFF_SUFFIXES = {".tif", ".tiff"}

# GDAL's block cache defaults to 5% of RAM and, window by window, fills with the whole scene:
# measured on a 768 MB scene, reading every tile peaked at 825 MB of memory uncapped and 53 MB
# at this cap. Each tile is read about once, so a larger cache buys nothing. A `GDAL_CACHEMAX`
# set in the environment is left alone.
_GDAL_CACHE_MB = 64


class GeoTiffImage:
    """One band of a GeoTIFF, sliced like a 2-D array but read only where it is sliced.

    `image[r0:r1, c0:c1]` reads that window from disk. `np.asarray(image)` reads the whole band,
    which is what the viewer's scene rendering still does.
    """

    ndim = 2

    def __init__(self, path: str | Path, band: int = 1) -> None:
        import rasterio

        self._source = None  # first, so `close` is safe even if opening fails below
        self.path = Path(path)
        self.band = int(band)
        with rasterio.open(self.path) as source:
            if not 1 <= self.band <= source.count:
                raise ValueError(f"{self.path} has {source.count} band(s); band {band} asked for")
            self.shape = (source.height, source.width)
            self.dtype = np.dtype(source.dtypes[self.band - 1])
            self.transform = source.transform
            self.crs = source.crs.to_string() if source.crs else None
            self.nodata = source.nodata

    def __getitem__(self, key) -> np.ndarray:
        from rasterio.windows import Window

        two_slices = isinstance(key, tuple) and len(key) == 2
        if not (two_slices and all(isinstance(k, slice) for k in key)):
            raise TypeError("a GeoTiffImage is sliced by two slices, as image[r0:r1, c0:c1]")
        (row0, row1, row_step), (col0, col1, col_step) = (
            key[0].indices(self.shape[0]),
            key[1].indices(self.shape[1]),
        )
        if row_step != 1 or col_step != 1:
            raise ValueError("a GeoTiffImage window cannot be strided")
        height, width = max(row1 - row0, 0), max(col1 - col0, 0)
        return self._read(window=Window(col0, row0, width, height))

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        data = self._read()
        return data if dtype is None else data.astype(dtype, copy=False)

    def _read(self, window=None) -> np.ndarray:
        import rasterio

        options = {} if "GDAL_CACHEMAX" in os.environ else {"GDAL_CACHEMAX": _GDAL_CACHE_MB}
        with rasterio.Env(**options):
            return self._cleaned(self._open().read(self.band, window=window))

    def close(self) -> None:
        if getattr(self, "_source", None) is not None:
            self._source.close()
            self._source = None

    def _open(self):
        if self._source is None:
            import rasterio

            self._source = rasterio.open(self.path)
        return self._source

    def _cleaned(self, data: np.ndarray) -> np.ndarray:
        if self.nodata is not None and not np.isnan(self.nodata):
            data = np.where(data == self.nodata, 0, data).astype(data.dtype, copy=False)
        if np.issubdtype(data.dtype, np.floating):
            data = np.where(np.isfinite(data), data, 0).astype(data.dtype, copy=False)
        return data

    def __getstate__(self) -> dict:
        # An open dataset handle cannot be pickled; a copy reopens the file when it is read.
        return {**self.__dict__, "_source": None}

    def __del__(self) -> None:
        self.close()


Image = Union[np.ndarray, GeoTiffImage]


@dataclass(frozen=True)
class Scene:
    """One SAR acquisition: pixels, the geometry to place them on the ground, and its identity.

    `id` is the provenance key structures are grouped by — see `data.provenance` and
    `embed.structures.standing`. It must be stable and unique per acquisition.

    `image` is an array, a read-only memory map, or a `GeoTiffImage`. Code that needs pixels
    slices it; code that needs every pixel calls `np.asarray` on it.
    """

    id: str
    image: Image
    transform: Affine
    crs: str
    acquired_at: datetime
    heading_deg: float
    incidence_deg: float

    @property
    def footprint(self) -> Polygon:
        """The ground polygon these pixels cover, in `crs`: what a search of this scene searched.

        Built from the four image corners rather than from a bounding box, so a transform with
        any rotation in it still describes the ground the radar actually looked at.
        """
        height, width = self.image.shape
        corners = ((0, 0), (width, 0), (width, height), (0, height))
        return Polygon([self.transform @ corner for corner in corners])


def read_scene(directory: str | Path) -> Scene:
    """Load a scene directory: `scene.json` plus `image.npy` or the GeoTIFF it names.

    `scene.json` may carry `"image"` (a file name, relative to the directory) and `"band"`
    (1-based, for a multi-band GeoTIFF). `"transform"` and `"crs"` are required for `.npy`
    and optional for a GeoTIFF, whose own georeferencing is used when they are left out.
    """
    directory = Path(directory)
    meta = json.loads((directory / "scene.json").read_text())
    image_path = directory / meta.get("image", "image.npy")

    if image_path.suffix.lower() in _GEOTIFF_SUFFIXES:
        image: Image = GeoTiffImage(image_path, band=meta.get("band", 1))
        transform = Affine(*meta["transform"]) if "transform" in meta else image.transform
        crs = meta.get("crs") or image.crs
        if crs is None:
            raise ValueError(f"{image_path} has no CRS, and scene.json does not give one")
    else:
        # Memory-mapped: a tile's pixels are paged in when the detector reads them.
        image = np.load(image_path, mmap_mode="r")
        transform, crs = Affine(*meta["transform"]), meta["crs"]

    return Scene(
        id=meta["id"],
        image=image,
        transform=transform,
        crs=crs,
        acquired_at=datetime.fromisoformat(meta["acquired_at"]),
        heading_deg=meta["heading_deg"],
        incidence_deg=meta["incidence_deg"],
    )


def write_scene(directory: str | Path, scene: Scene) -> None:
    """Write a scene in the layout `read_scene` expects, always as `image.npy`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "image.npy", np.asarray(scene.image))
    meta = {
        "id": scene.id,
        "transform": list(scene.transform)[:6],
        "crs": scene.crs,
        "acquired_at": scene.acquired_at.isoformat(),
        "heading_deg": scene.heading_deg,
        "incidence_deg": scene.incidence_deg,
    }
    (directory / "scene.json").write_text(json.dumps(meta, indent=2))
