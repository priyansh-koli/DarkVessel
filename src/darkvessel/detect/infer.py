"""Whole-scene inference: tile, run the detector on each window, keep what each tile owns.

An optional **mask** marks pixels not to search (land, and a buffer off it). A tile whose core
is all masked is skipped before its pixels are read. Elsewhere masked pixels are set to zero,
the value every detector already treats as outside the radar swath, and a detection still
landing on one is dropped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
import pandas as pd

from darkvessel.data.tiling import Tile, Tiling
from darkvessel.detect.detector import Detector

# Called with a tile; returns True where the tile's window must not be searched, or `None`
# where nothing in the window is masked.
Mask = Callable[[Tile], Optional[np.ndarray]]


def detect_scene(
    image: np.ndarray, detector: Detector, tiling: Tiling, mask: Mask | None = None
) -> pd.DataFrame:
    """Run `detector` over every tile of `image`, deduplicated by tile ownership.

    `image` is anything with a 2-D `shape` that slices to an array: an array, a memory map, or
    a `data.scene.GeoTiffImage` read window by window. Returns one row per detection with
    `row`, `col` in the full scene's pixel frame.
    """
    height, width = image.shape
    rows: list[float] = []
    cols: list[float] = []
    for tile in tiling.tiles(height, width):
        masked = mask(tile) if mask is not None else None
        if masked is not None and _core_masked(masked, tile):
            continue
        window = np.asarray(image[tile.row0 : tile.row1, tile.col0 : tile.col1])
        if masked is not None:
            window = np.where(masked, 0, window).astype(window.dtype, copy=False)
        for local_row, local_col in detector(window):
            row, col = local_row + tile.row0, local_col + tile.col0
            if not tile.owns(row, col):
                continue
            if masked is not None and masked[_pixel(local_row, masked.shape[0]),
                                             _pixel(local_col, masked.shape[1])]:
                continue
            rows.append(row)
            cols.append(col)
    return pd.DataFrame({"row": rows, "col": cols})


def _core_masked(masked: np.ndarray, tile: Tile) -> bool:
    core = masked[
        tile.core_row0 - tile.row0 : tile.core_row1 - tile.row0,
        tile.core_col0 - tile.col0 : tile.core_col1 - tile.col0,
    ]
    return bool(core.all())


def _pixel(position: float, size: int) -> int:
    return min(max(int(position), 0), size - 1)
