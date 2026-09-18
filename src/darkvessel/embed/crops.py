"""Fixed-size crops around each detection, padded at the scene's edges."""

import numpy as np
import pandas as pd


def crops_for(image: np.ndarray, found: pd.DataFrame, crop_px: int, margin_px: int) -> np.ndarray:
    """One `(crop_px + 2*margin_px)` square crop per detection, centred on its pixel position."""
    half = crop_px // 2 + margin_px
    size = 2 * half
    padded = np.pad(image, half, mode="constant", constant_values=0)
    crops = np.zeros((len(found), size, size), dtype=image.dtype)
    for i, (row, col) in enumerate(zip(found["row"], found["col"])):
        r, c = int(round(row)) + half, int(round(col)) + half
        crops[i] = padded[r - half : r + half, c - half : c + half]
    return crops
