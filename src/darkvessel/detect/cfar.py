"""Two-parameter CA-CFAR: the classical radar ship detector, and the baseline to beat.

Constant False Alarm Rate detection compares each pixel with the sea clutter around it rather
than with a fixed level, so a calm and a rough patch of the same scene are judged on the same
terms. For every pixel, clutter is estimated as the mean and spread of a ring of background
pixels — outside a guard window wide enough that the ship itself does not contaminate its own
background — and the pixel is scored by how many clutter standard deviations it stands above.

No training, a few box filters per window: fast, explainable, and what any learned detector has
to beat to justify itself (`docs/final-goal.md`, milestone 4).

Pixels at exactly zero are outside the radar swath. They are excluded from every clutter
estimate and can never be detected, which is what stops the swath edge — a cliff from zero to
sea — reading as a line of ships.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Windows in pixels. The guard must cover the largest ship expected (LS-SSDD p95 is ~45 px).
TARGET_PX = 3
GUARD_PX = 41
BACKGROUND_PX = 81


class CFARDetector:
    """Satisfies the `Detector` protocol; `threshold` is in clutter standard deviations.

    The default, 6.0, is the threshold that maximised F1 on the LS-SSDD validation scenes
    (`models/benchmark_cfar.json`) — the operating point the baseline is reported at.
    """

    def __init__(
        self,
        threshold: float = 6.0,
        target_px: int = TARGET_PX,
        guard_px: int = GUARD_PX,
        background_px: int = BACKGROUND_PX,
        min_pixels: int = 2,
    ) -> None:
        self.threshold = threshold
        self.target_px = target_px
        self.guard_px = guard_px
        self.background_px = background_px
        self.min_pixels = min_pixels

    def __call__(self, window: np.ndarray) -> list[tuple[float, float]]:
        points = self.scored(window, floor=self.threshold)
        return [(float(r), float(c)) for r, c, _ in points]

    def scored(self, window: np.ndarray, floor: float = 2.0) -> np.ndarray:
        """(n, 3) array of (row, col, score) for every blob scoring at least `floor`."""
        z = self.statistic(window)
        blobs, count = ndimage.label(z >= floor)
        if count == 0:
            return np.zeros((0, 3), dtype=np.float32)
        index = np.arange(1, count + 1)
        sizes = ndimage.sum_labels(np.ones_like(z), blobs, index)
        peaks = ndimage.maximum(z, blobs, index)
        # Weighted by excess over the floor: the centroid follows the bright core of the hull.
        centres = ndimage.center_of_mass(np.clip(z - floor, 0, None) + 1e-6, blobs, index)
        keep = sizes >= self.min_pixels
        out = [(r, c, s) for (r, c), s, k in zip(centres, peaks, keep) if k]
        return np.asarray(out, dtype=np.float32).reshape(-1, 3)

    def statistic(self, window: np.ndarray) -> np.ndarray:
        """Per-pixel CFAR score: (target mean - clutter mean) / clutter std."""
        image = np.asarray(window, dtype=np.float64)
        valid = np.isfinite(image) & (image != 0)
        image = np.where(valid, image, 0.0)

        target = _box_mean(image, valid, self.target_px)
        outer_sum, outer_n = _box_sums(image, valid, self.background_px)
        guard_sum, guard_n = _box_sums(image, valid, self.guard_px)
        outer_sq, _ = _box_sums(image**2, valid, self.background_px)
        guard_sq, _ = _box_sums(image**2, valid, self.guard_px)

        n = np.maximum(outer_n - guard_n, 1.0)
        mean = (outer_sum - guard_sum) / n
        var = np.maximum((outer_sq - guard_sq) / n - mean**2, 0.0)
        # A floor on the spread: over perfectly flat clutter any speck would be infinitely
        # significant.
        std = np.sqrt(var) + 1e-3
        z = (target - mean) / std
        z[~valid | (outer_n - guard_n < 0.25 * (self.background_px**2 - self.guard_px**2))] = 0.0
        return z.astype(np.float32)


def _box_sums(values: np.ndarray, valid: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum of valid `values`, and how many valid pixels, in a size x size box at every pixel."""
    area = float(size * size)
    total = ndimage.uniform_filter(np.where(valid, values, 0.0), size=size, mode="constant") * area
    count = ndimage.uniform_filter(valid.astype(np.float64), size=size, mode="constant") * area
    return total, count


def _box_mean(values: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    total, count = _box_sums(values, valid, size)
    return total / np.maximum(count, 1.0)
