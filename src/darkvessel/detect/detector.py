"""The detector contract: pixels of one tile's window in, pixel centres of detections out.

The detector arrives as a parameter, never an import — this Protocol is the whole contract.
A deterministic stand-in (`detect.stub.BrightPixelDetector`) and a trained CNN both satisfy
it, and nothing else in the chain changes between them.
"""

from typing import Protocol

import numpy as np


class Detector(Protocol):
    def __call__(self, window: np.ndarray) -> list[tuple[float, float]]:
        """Return (row, col) pixel centres of detections found in `window`, in its own frame."""
        ...
