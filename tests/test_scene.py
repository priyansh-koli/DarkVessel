"""Scene round-tripping: what `write_scene` writes, `read_scene` must read back exactly."""

from datetime import datetime, timezone

import numpy as np
from affine import Affine

from darkvessel.data.scene import Scene, read_scene, write_scene


def _scene() -> Scene:
    return Scene(
        id="scene-1",
        image=np.arange(12, dtype=np.float32).reshape(3, 4),
        transform=Affine(5.0, 0.0, 500000.0, 0.0, -5.0, 6100000.0),
        crs="EPSG:25832",
        acquired_at=datetime(2026, 8, 9, 5, 31, 24, tzinfo=timezone.utc),
        heading_deg=350.0,
        incidence_deg=35.0,
    )


def test_round_trip_preserves_every_field(tmp_path):
    original = _scene()
    write_scene(tmp_path / "scene", original)
    loaded = read_scene(tmp_path / "scene")

    assert loaded.id == original.id
    assert np.array_equal(loaded.image, original.image)
    assert loaded.transform == original.transform
    assert loaded.crs == original.crs
    assert loaded.acquired_at == original.acquired_at
    assert loaded.heading_deg == original.heading_deg
    assert loaded.incidence_deg == original.incidence_deg


def test_write_scene_creates_the_directory(tmp_path):
    out = tmp_path / "nested" / "scene"
    write_scene(out, _scene())
    assert (out / "image.npy").exists()
    assert (out / "scene.json").exists()


def test_image_dtype_is_preserved(tmp_path):
    write_scene(tmp_path / "scene", _scene())
    loaded = read_scene(tmp_path / "scene")
    assert loaded.image.dtype == np.float32
