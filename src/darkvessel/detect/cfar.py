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

from darkvessel.data.tiling import Tiling

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
        # Measured over the labelled pixels only. `ndimage.maximum` and friends sort or scan
        # every pixel of the window, which took half of CFAR's time on real Sentinel-1 tiles
        # where only a handful of pixels clear the floor.
        rows, cols = np.nonzero(blobs)
        labels = blobs[rows, cols]
        scores = z[rows, cols].astype(np.float64)
        sizes = np.bincount(labels, minlength=count + 1)[1:]
        peaks = np.full(count + 1, -np.inf)
        np.maximum.at(peaks, labels, scores)
        # Weighted by excess over the floor: the centroid follows the bright core of the hull.
        weight = np.maximum(scores - floor, 0.0) + 1e-6
        total = np.bincount(labels, weights=weight, minlength=count + 1)[1:]
        centre_row = np.bincount(labels, weights=weight * rows, minlength=count + 1)[1:] / total
        centre_col = np.bincount(labels, weights=weight * cols, minlength=count + 1)[1:] / total
        keep = sizes >= self.min_pixels
        out = np.column_stack([centre_row, centre_col, peaks[1:]])[keep]
        return out.astype(np.float32).reshape(-1, 3)

    @property
    def context_px(self) -> int:
        """How far from a pixel its score reads: half the background window."""
        return self.background_px // 2

    @property
    def preferred_tiling(self) -> Tiling:
        """Tiles this detector runs best on, used when a run configuration names none.

        CFAR is a local filter, so it needs no tiling at all, only enough memory. What it does
        need is its full background window around every pixel it owns: with an overlap of at
        least half that window, each core pixel is scored exactly as a whole-scene pass would
        score it, and a tile edge never truncates anyone's clutter estimate. Large tiles keep
        the overlap's share of the work small (about 30% at 1024 px, against 300% at 128).
        """
        return Tiling(tile_px=1024, overlap_px=max(64, self.context_px + 8))

    def statistic(self, window: np.ndarray) -> np.ndarray:
        """Per-pixel CFAR score: (target mean - clutter mean) / clutter std."""
        image = np.asarray(window, dtype=np.float64)
        valid = np.isfinite(image) & (image != 0)
        image = np.where(valid, image, 0.0)
        squared = image**2
        weights = valid.astype(np.float64)

        # Invalid pixels are already zero in `image` and `squared`, so a plain box sum over
        # them is the sum over the valid pixels. Each window's count is filtered once.
        target_n = _box_sum(weights, self.target_px)
        target = _box_sum(image, self.target_px) / np.maximum(target_n, 1.0)
        outer_n = _box_sum(weights, self.background_px)
        guard_n = _box_sum(weights, self.guard_px)
        outer_sum = _box_sum(image, self.background_px)
        guard_sum = _box_sum(image, self.guard_px)
        outer_sq = _box_sum(squared, self.background_px)
        guard_sq = _box_sum(squared, self.guard_px)

        n = np.maximum(outer_n - guard_n, 1.0)
        mean = (outer_sum - guard_sum) / n
        var = np.maximum((outer_sq - guard_sq) / n - mean**2, 0.0)
        # A floor on the spread: over perfectly flat clutter any speck would be infinitely
        # significant.
        std = np.sqrt(var) + 1e-3
        z = (target - mean) / std
        z[~valid | (outer_n - guard_n < 0.25 * (self.background_px**2 - self.guard_px**2))] = 0.0
        return z.astype(np.float32)


def _box_sum(values: np.ndarray, size: int) -> np.ndarray:
    """Sum of `values` in a size x size box at every pixel, zero beyond the edge."""
    return ndimage.uniform_filter(values, size=size, mode="constant") * float(size * size)
