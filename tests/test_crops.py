"""Fixed-size crops around each detection, zero-padded where a detection sits at the edge."""

import numpy as np
import pandas as pd

from darkvessel.embed.crops import crops_for


def test_every_crop_has_the_size_the_embedder_asked_for():
    image = np.arange(10000, dtype=np.float32).reshape(100, 100)
    found = pd.DataFrame({"row": [50.0, 20.0], "col": [50.0, 80.0]})
    crops = crops_for(image, found, crop_px=32, margin_px=4)
    assert crops.shape == (2, 40, 40)  # 32//2 + 4 = 20 either side


def test_a_crop_is_centred_on_its_detection():
    image = np.zeros((100, 100), dtype=np.float32)
    image[50, 50] = 1.0
    crops = crops_for(image, pd.DataFrame({"row": [50.0], "col": [50.0]}), crop_px=8, margin_px=0)
    half = 8 // 2
    assert crops[0][half, half] == 1.0


def test_a_detection_at_the_scene_edge_is_padded_rather_than_clipped():
    """A crop must keep its declared size wherever the detection is, or an embedder with a
    fixed input shape could not consume it."""
    image = np.ones((100, 100), dtype=np.float32)
    crops = crops_for(image, pd.DataFrame({"row": [0.0], "col": [0.0]}), crop_px=8, margin_px=0)
    assert crops.shape == (1, 8, 8)
    assert crops[0][0, 0] == 0.0  # padded region
    assert crops[0][4, 4] == 1.0  # the detection itself


def test_fractional_pixel_positions_are_rounded_to_a_whole_pixel():
    image = np.zeros((100, 100), dtype=np.float32)
    image[50, 50] = 1.0
    crops = crops_for(image, pd.DataFrame({"row": [49.6], "col": [50.4]}), crop_px=8, margin_px=0)
    assert crops[0][4, 4] == 1.0


def test_no_detections_gives_an_empty_stack_of_the_right_shape():
    image = np.zeros((100, 100), dtype=np.float32)
    crops = crops_for(image, pd.DataFrame({"row": [], "col": []}), crop_px=8, margin_px=2)
    assert crops.shape == (0, 12, 12)


def test_crops_keep_the_image_dtype():
    image = np.zeros((50, 50), dtype=np.float32)
    crops = crops_for(image, pd.DataFrame({"row": [25.0], "col": [25.0]}), crop_px=8, margin_px=0)
    assert crops.dtype == np.float32
