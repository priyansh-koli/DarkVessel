"""Whole-scene inference: every target claimed exactly once, by the tile whose core owns it.

The point of tile ownership is that a target sitting in the overlap between two tiles is seen
twice by the detector but reported once — without a merge radius to tune afterwards. See
`data/tiling.py`.
"""

import numpy as np

from darkvessel.data.tiling import Tiling
from darkvessel.detect.infer import detect_scene
from darkvessel.detect.stub import BrightPixelDetector

_DETECTOR = BrightPixelDetector(threshold=0.5)
_TILING = Tiling(tile_px=64, overlap_px=8)


def test_detections_are_reported_in_the_full_scene_pixel_frame():
    image = np.zeros((200, 200), dtype=np.float32)
    image[150, 170] = 1.0
    found = detect_scene(image, _DETECTOR, _TILING)
    assert found["row"].tolist() == [150.0]
    assert found["col"].tolist() == [170.0]


def test_a_target_in_the_overlap_is_claimed_exactly_once():
    """A target placed on a core boundary is inside two tiles' *windows* but only one core."""
    step = _TILING.tile_px - 2 * _TILING.overlap_px
    image = np.zeros((200, 200), dtype=np.float32)
    image[step, step] = 1.0  # exactly on the boundary between neighbouring cores
    found = detect_scene(image, _DETECTOR, _TILING)
    assert len(found) == 1


def test_every_target_in_a_full_scene_is_found_once():
    image = np.zeros((200, 200), dtype=np.float32)
    targets = [(10, 10), (48, 100), (96, 48), (150, 150), (199, 199)]
    for row, col in targets:
        image[row, col] = 1.0
    found = detect_scene(image, _DETECTOR, _TILING)
    assert sorted(zip(found["row"], found["col"])) == [(float(r), float(c)) for r, c in targets]


def test_an_empty_scene_gives_an_empty_frame_with_the_right_columns():
    image = np.zeros((100, 100), dtype=np.float32)
    found = detect_scene(image, _DETECTOR, _TILING)
    assert found.empty
    assert list(found.columns) == ["row", "col"]


def test_the_detector_is_only_ever_shown_a_tile_window():
    """The Detector protocol is the whole contract: a detector sees one window at a time,
    never the scene, so a trained CNN with a fixed input size can satisfy it."""
    seen = []

    def recording_detector(window):
        seen.append(window.shape)
        return []

    detect_scene(np.zeros((300, 300), dtype=np.float32), recording_detector, _TILING)
    assert seen, "the detector was never called"
    assert all(rows <= _TILING.tile_px and cols <= _TILING.tile_px for rows, cols in seen)
