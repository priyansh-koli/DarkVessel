"""A deterministic stand-in detector: no weights, no GPU, no network.

Flags every pixel at or above `threshold` and collapses connected components to one detection
each. Fixed from an earlier version that reported one detection per pixel across a flat
plateau of equal bright values — a stand-in that scores a whole superstructure as a dozen
separate "vessels" is not a useful contract for testing the rest of the chain.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label


@dataclass(frozen=True)
class BrightPixelDetector:
    threshold: float

    def __call__(self, window: np.ndarray) -> list[tuple[float, float]]:
        mask = window >= self.threshold
        if not mask.any():
            return []
        labelled, count = label(mask)
        centres = []
        for component in range(1, count + 1):
            rows, cols = np.nonzero(labelled == component)
            centres.append((float(rows.mean()), float(cols.mean())))
        return centres
