"""Fixed-size crops around each detection, padded at the scene's edges."""

import numpy as np
import pandas as pd


def crops_for(image: np.ndarray, found: pd.DataFrame, crop_px: int, margin_px: int) -> np.ndarray:
    """One `(crop_px + 2*margin_px)` square crop per detection, centred on its pixel position.

    Each crop is read as its own window, so `image` can be a memory map or a
    `data.scene.GeoTiffImage` without the whole scene being loaded. Beyond the edge is zero.
    """
    half = crop_px // 2 + margin_px
    size = 2 * half
    height, width = image.shape
    crops = np.zeros((len(found), size, size), dtype=image.dtype)
    for i, (row, col) in enumerate(zip(found["row"], found["col"])):
        r, c = int(round(row)), int(round(col))
        row0, row1 = max(r - half, 0), min(r + half, height)
        col0, col1 = max(c - half, 0), min(c + half, width)
        if row0 >= row1 or col0 >= col1:
            continue
        crops[i, row0 - (r - half) : row1 - (r - half), col0 - (c - half) : col1 - (c - half)] = (
            np.asarray(image[row0:row1, col0:col1])
        )
    return crops
