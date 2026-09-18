"""Tiling: overlapping windows whose non-overlapping cores partition the scene exactly."""

import pytest

from darkvessel.data.tiling import Tiling


def test_overlap_must_be_less_than_half_the_tile():
    with pytest.raises(ValueError):
        Tiling(tile_px=100, overlap_px=50)


def test_cores_partition_the_image_with_no_gap_or_double_coverage():
    """Every pixel must belong to exactly one tile's core — see the module docstring on why
    that's the whole point of tiling with an owned core in the first place."""
    tiling = Tiling(tile_px=64, overlap_px=8)
    height, width = 200, 150
    owner = [[None for _ in range(width)] for _ in range(height)]
    for tile_id, tile in enumerate(tiling.tiles(height, width)):
        for row in range(tile.core_row0, tile.core_row1):
            for col in range(tile.core_col0, tile.core_col1):
                assert owner[row][col] is None, f"pixel ({row},{col}) claimed twice"
                owner[row][col] = tile_id
    assert all(cell is not None for row in owner for cell in row), "some pixel was never claimed"


def test_windows_extend_into_the_overlap_but_clip_at_the_image_edge():
    tiling = Tiling(tile_px=64, overlap_px=8)
    tiles = list(tiling.tiles(100, 100))
    first = tiles[0]
    assert first.core_row0 == 0 and first.core_col0 == 0
    # top-left tile's core touches the edge, so its window can't extend past it
    assert first.row0 == 0
    assert first.col0 == 0

    last = tiles[-1]
    assert last.core_row1 == 100 and last.core_col1 == 100
    assert last.row1 == 100
    assert last.col1 == 100


def test_an_interior_tile_window_extends_overlap_px_beyond_its_core():
    tiling = Tiling(tile_px=64, overlap_px=8)
    tiles = list(tiling.tiles(300, 300))
    interior = next(t for t in tiles if t.core_row0 > 0 and t.core_col0 > 0)
    assert interior.row0 == interior.core_row0 - 8
    assert interior.col0 == interior.core_col0 - 8


def test_owns_is_true_only_within_the_core_bounds():
    tiling = Tiling(tile_px=64, overlap_px=8)
    tile = next(iter(tiling.tiles(200, 200)))
    assert tile.owns(tile.core_row0, tile.core_col0)
    assert not tile.owns(tile.core_row1, tile.core_col0)  # half-open: upper bound excluded
    assert not tile.owns(tile.row0, tile.col0) if tile.row0 < tile.core_row0 else True


def test_a_scene_smaller_than_one_tile_still_yields_a_single_full_core():
    tiling = Tiling(tile_px=128, overlap_px=16)
    tiles = list(tiling.tiles(40, 30))
    assert len(tiles) == 1
    tile = tiles[0]
    assert (tile.core_row0, tile.core_row1) == (0, 40)
    assert (tile.core_col0, tile.core_col1) == (0, 30)
