"""The deterministic stand-in detector, and the flat-plateau bug it was fixed for."""

import numpy as np

from darkvessel.detect.stub import BrightPixelDetector


def test_a_flat_bright_plateau_is_one_detection_not_one_per_pixel():
    """The fix this module's docstring records: a stand-in that scores a whole superstructure
    as a dozen separate "vessels" is not a useful contract for testing the rest of the chain."""
    window = np.zeros((10, 10), dtype=np.float32)
    window[4:7, 4:7] = 1.0  # nine equal bright pixels, one object
    assert len(BrightPixelDetector(threshold=0.5)(window)) == 1


def test_a_component_is_reported_at_its_centre():
    window = np.zeros((10, 10), dtype=np.float32)
    window[4:6, 4:6] = 1.0
    (row, col) = BrightPixelDetector(threshold=0.5)(window)[0]
    assert (row, col) == (4.5, 4.5)


def test_separated_components_are_reported_separately():
    window = np.zeros((10, 10), dtype=np.float32)
    window[1, 1] = 1.0
    window[8, 8] = 1.0
    assert len(BrightPixelDetector(threshold=0.5)(window)) == 2


def test_nothing_above_threshold_gives_nothing():
    window = np.full((10, 10), 0.1, dtype=np.float32)
    assert BrightPixelDetector(threshold=0.5)(window) == []


def test_the_threshold_is_inclusive():
    window = np.zeros((4, 4), dtype=np.float32)
    window[2, 2] = 0.5
    assert len(BrightPixelDetector(threshold=0.5)(window)) == 1


def test_the_detector_is_deterministic():
    window = np.zeros((20, 20), dtype=np.float32)
    window[3:5, 3:5] = 1.0
    window[15, 15] = 2.0
    detector = BrightPixelDetector(threshold=0.5)
    assert detector(window) == detector(window)
