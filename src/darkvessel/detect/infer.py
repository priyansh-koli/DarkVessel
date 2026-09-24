"""Whole-scene inference: tile, run the detector on each window, keep what each tile owns."""

import numpy as np
import pandas as pd

from darkvessel.data.tiling import Tiling
from darkvessel.detect.detector import Detector


def detect_scene(image: np.ndarray, detector: Detector, tiling: Tiling) -> pd.DataFrame:
    """Run `detector` over every tile of `image`, deduplicated by tile ownership.

    `image` is anything with a 2-D `shape` that slices to an array: an array, a memory map, or
    a `data.scene.GeoTiffImage` read window by window. Returns one row per detection with
    `row`, `col` in the full scene's pixel frame.
    """
    height, width = image.shape
    rows: list[float] = []
    cols: list[float] = []
    for tile in tiling.tiles(height, width):
        window = np.asarray(image[tile.row0 : tile.row1, tile.col0 : tile.col1])
        for local_row, local_col in detector(window):
            row, col = local_row + tile.row0, local_col + tile.col0
            if tile.owns(row, col):
                rows.append(row)
                cols.append(col)
    return pd.DataFrame({"row": rows, "col": cols})
