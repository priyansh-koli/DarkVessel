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


def _write_geotiff(path, image, transform, crs="EPSG:25832", nodata=None):
    import rasterio

    with rasterio.open(
        path, "w", driver="GTiff", height=image.shape[0], width=image.shape[1], count=1,
        dtype=image.dtype, crs=crs, transform=transform, nodata=nodata,
    ) as target:
        target.write(image, 1)


def _geotiff_scene(tmp_path, image, nodata=None, **meta):
    import json

    original = _scene()
    directory = tmp_path / "tif-scene"
    directory.mkdir()
    _write_geotiff(directory / "scene.tif", image, original.transform, nodata=nodata)
    (directory / "scene.json").write_text(
        json.dumps(
            {
                "id": "tif-1",
                "image": "scene.tif",
                "acquired_at": original.acquired_at.isoformat(),
                "heading_deg": 350.0,
                "incidence_deg": 35.0,
                **meta,
            }
        )
    )
    return read_scene(directory)


def test_npy_scene_is_memory_mapped_not_loaded(tmp_path):
    write_scene(tmp_path / "scene", _scene())
    assert isinstance(read_scene(tmp_path / "scene").image, np.memmap)


def test_a_geotiff_scene_takes_its_georeferencing_from_the_file(tmp_path):
    image = np.arange(60 * 80, dtype=np.float32).reshape(60, 80) + 1
    scene = _geotiff_scene(tmp_path, image)
    assert scene.image.shape == (60, 80)
    assert scene.transform == _scene().transform
    assert scene.crs == "EPSG:25832"
    assert np.array_equal(scene.image[10:20, 30:45], image[10:20, 30:45])
    assert np.array_equal(np.asarray(scene.image), image)


def test_a_geotiff_window_past_the_edge_is_clipped_like_an_array(tmp_path):
    image = np.ones((30, 40), dtype=np.float32)
    scene = _geotiff_scene(tmp_path, image)
    assert scene.image[20:50, 35:60].shape == image[20:50, 35:60].shape


def test_geotiff_no_data_and_nan_read_as_zero(tmp_path):
    """Zero is what every detector treats as outside the swath."""
    image = np.full((10, 10), 5.0, dtype=np.float32)
    image[0, 0] = -9999.0
    image[1, 1] = np.nan
    scene = _geotiff_scene(tmp_path, image, nodata=-9999.0)
    window = scene.image[0:10, 0:10]
    assert window[0, 0] == 0.0 and window[1, 1] == 0.0 and window[2, 2] == 5.0


def test_a_geotiff_band_that_does_not_exist_is_refused(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="band"):
        _geotiff_scene(tmp_path, np.ones((5, 5), dtype=np.float32), band=2)


def test_the_synthetic_fixture_runs_the_same_from_a_geotiff(tmp_path):
    """The whole chain must not care whether pixels come from memory or from disk windows."""
    from datetime import timedelta

    from darkvessel.data.synthetic import _ais
    from darkvessel.data.synthetic import _scene as fixture_scene
    from darkvessel.data.tiling import Tiling
    from darkvessel.detect.stub import BrightPixelDetector
    from darkvessel.pipeline import run

    in_memory = fixture_scene()
    import json

    directory = tmp_path / "fixture"
    directory.mkdir()
    _write_geotiff(directory / "scene.tif", in_memory.image, in_memory.transform, in_memory.crs)
    (directory / "scene.json").write_text(
        json.dumps(
            {
                "id": in_memory.id,
                "image": "scene.tif",
                "acquired_at": in_memory.acquired_at.isoformat(),
                "heading_deg": in_memory.heading_deg,
                "incidence_deg": in_memory.incidence_deg,
            }
        )
    )
    from_disk = read_scene(directory)

    def statuses(scene):
        return run(
            scene=scene,
            ais=_ais(),
            detector=BrightPixelDetector(threshold=0.5),
            tiling=Tiling(tile_px=128, overlap_px=32),
            tolerance_m=200.0,
            max_gap=timedelta(minutes=10),
        ).detections[["row", "col", "status"]]

    assert statuses(from_disk).equals(statuses(in_memory))
