"""Overlapping tiles, each owning a non-overlapping core.

Overlapping tiles see edge targets twice. Rather than merge detections afterwards — which
needs a radius to tune, and risks merging two genuinely close hulls — each tile owns a
non-overlapping core and reports only what falls inside it. Every pixel is in exactly one
core, so a target is claimed exactly once, by construction.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    """A window to run the detector on, and the core within it this tile owns."""

    row0: int
    col0: int
    row1: int
    col1: int
    core_row0: int
    core_col0: int
    core_row1: int
    core_col1: int

    def owns(self, row: float, col: float) -> bool:
        return self.core_row0 <= row < self.core_row1 and self.core_col0 <= col < self.core_col1


@dataclass(frozen=True)
class Tiling:
    """Tiles of `tile_px` with `overlap_px` of shared context between neighbours."""

    tile_px: int
    overlap_px: int

    def __post_init__(self) -> None:
        if self.overlap_px * 2 >= self.tile_px:
            raise ValueError("overlap_px must be less than half of tile_px")

    def tiles(self, height: int, width: int) -> Iterator[Tile]:
        """Cores partition `(height, width)` exactly; windows extend into the overlap."""
        step = self.tile_px - 2 * self.overlap_px
        core_row = 0
        while core_row < height:
            core_row1 = min(core_row + step, height)
            core_col = 0
            while core_col < width:
                core_col1 = min(core_col + step, width)
                yield Tile(
                    row0=max(core_row - self.overlap_px, 0),
                    col0=max(core_col - self.overlap_px, 0),
                    row1=min(core_row1 + self.overlap_px, height),
                    col1=min(core_col1 + self.overlap_px, width),
                    core_row0=core_row,
                    core_col0=core_col,
                    core_row1=core_row1,
                    core_col1=core_col1,
                )
                core_col = core_col1
            core_row = core_row1
