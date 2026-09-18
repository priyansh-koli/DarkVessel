"""PNG rendering of SAR pixels."""

import base64
import io
import struct

import numpy as np
import pandas as pd
from PIL import Image

from darkvessel.render import crop_data_uris, scene_png, stretch


def _png_size(data: bytes) -> tuple[int, int]:
    """Width and height straight out of the IHDR chunk."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _decode(uri: str) -> bytes:
    return base64.b64decode(uri.split(",", 1)[1])


def test_stretch_maps_the_full_range_onto_8_bit():
    image = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    out = stretch(image)
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255


def test_a_mostly_empty_scene_falls_back_to_min_max():
    """Both percentiles collapse to zero on a synthetic scene, so the percentile stretch alone
    would render every target black."""
    image = np.zeros((50, 50), dtype=np.float32)
    image[10, 10] = 1.0
    out = stretch(image)
    assert out[10, 10] == 255
    assert out[0, 0] == 0


def test_a_constant_scene_renders_black_rather_than_dividing_by_zero():
    out = stretch(np.full((8, 8), 3.0, dtype=np.float32))
    assert out.shape == (8, 8)
    assert set(np.unique(out)) == {0}


def test_an_all_nan_scene_is_handled():
    out = stretch(np.full((4, 4), np.nan, dtype=np.float32))
    assert set(np.unique(out)) == {0}


def test_scene_png_is_a_png_at_native_resolution():
    image = np.zeros((240, 180), dtype=np.float32)
    image[100, 100] = 1.0
    assert _png_size(scene_png(image)) == (180, 240)  # PNG is (width, height)


def test_one_crop_data_uri_per_detection():
    image = np.zeros((100, 100), dtype=np.float32)
    image[50, 50] = 1.0
    found = pd.DataFrame({"row": [50.0, 20.0], "col": [50.0, 20.0]})
    uris = crop_data_uris(image, found, crop_px=32, margin_px=8)
    assert len(uris) == 2
    assert all(uri.startswith("data:image/png;base64,") for uri in uris)


def test_crops_are_the_requested_size():
    image = np.zeros((100, 100), dtype=np.float32)
    found = pd.DataFrame({"row": [50.0], "col": [50.0]})
    uris = crop_data_uris(image, found, crop_px=32, margin_px=8)
    assert _png_size(_decode(uris[0])) == (48, 48)  # 32//2 + 8 either side


def test_no_detections_gives_no_crops():
    empty = pd.DataFrame({"row": [], "col": []})
    assert crop_data_uris(np.zeros((10, 10), dtype=np.float32), empty) == []


def test_crops_are_stretched_over_the_whole_scene_not_each_crop():
    """A crop normalised to itself would make empty water look as bright as a hull."""
    image = np.zeros((60, 60), dtype=np.float32)
    image[10, 10] = 1.0   # the bright target
    image[40, 40] = 0.1   # a faint one
    found = pd.DataFrame({"row": [40.0], "col": [40.0]})
    uris = crop_data_uris(image, found, crop_px=8, margin_px=0)

    faint = np.array(Image.open(io.BytesIO(_decode(uris[0]))))
    assert faint.max() < 200, "the faint target should not be stretched up to full white"
