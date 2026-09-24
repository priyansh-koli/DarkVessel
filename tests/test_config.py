"""Run-configuration loading, including the defaults an omitted key falls back to."""

from datetime import timedelta
from pathlib import Path

import pytest

from darkvessel.config import load
from darkvessel.data.tiling import Tiling
from darkvessel.detect.cfar import CFARDetector

_MINIMAL = """
scene_dir: data/synthetic/scene
tolerance_m: 200
"""


def _config(tmp_path, text: str) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(text)
    return path


def test_a_minimal_config_fills_in_every_default(tmp_path):
    cfg = load(_config(tmp_path, _MINIMAL))
    assert cfg.scene_dir == Path("data/synthetic/scene")
    assert cfg.max_gap == timedelta(minutes=10)
    assert cfg.tiling_for(None) == Tiling(tile_px=128, overlap_px=32)
    assert cfg.detector_threshold == 0.5
    assert cfg.output_path == Path("outputs/detections.gpkg")


def test_an_absent_ais_path_is_none_not_an_empty_path(tmp_path):
    """`None` here is what makes the run report `unsearched` rather than `dark` — the
    distinction between "searched, found nothing" and "never searched"."""
    cfg = load(_config(tmp_path, _MINIMAL))
    assert cfg.ais_path is None


def test_an_absent_geometry_means_no_azimuth_correction(tmp_path):
    cfg = load(_config(tmp_path, _MINIMAL))
    assert cfg.geometry is None


def test_geometry_is_built_when_the_block_is_present(tmp_path):
    cfg = load(
        _config(
            tmp_path,
            _MINIMAL + "geometry:\n  heading_deg: 350.0\n  incidence_deg: 35.0\n",
        )
    )
    assert cfg.geometry.heading_deg == 350.0
    assert cfg.geometry.incidence_deg == 35.0
    assert cfg.geometry.shift_per_mps == 113.0  # the Sentinel-1 IW default


def test_shift_per_mps_can_be_overridden(tmp_path):
    cfg = load(
        _config(
            tmp_path,
            _MINIMAL
            + "geometry:\n  heading_deg: 350.0\n  incidence_deg: 35.0\n  shift_per_mps: 90.0\n",
        )
    )
    assert cfg.geometry.shift_per_mps == 90.0


def test_max_gap_minutes_becomes_a_timedelta(tmp_path):
    cfg = load(_config(tmp_path, _MINIMAL + "max_gap_minutes: 45\n"))
    assert cfg.max_gap == timedelta(minutes=45)


def test_a_config_without_a_scene_is_refused(tmp_path):
    with pytest.raises(KeyError):
        load(_config(tmp_path, "tolerance_m: 200\n"))


def test_the_shipped_config_loads(tmp_path):
    """`configs/pipeline.yaml` is what the README's quick start runs — it must stay loadable."""
    cfg = load(Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml")
    assert cfg.tolerance_m == 200.0
    assert cfg.geometry is not None


def test_left_out_tiling_comes_from_the_detector_and_a_stated_one_wins(tmp_path):
    """CFAR needs its full clutter window around each pixel it owns, so it states its own
    tiling; a configuration that names one still gets exactly what it named."""
    cfar = CFARDetector()
    cfg = load(_config(tmp_path, _MINIMAL))
    assert cfg.tiling_for(cfar) == cfar.preferred_tiling
    assert cfar.preferred_tiling.overlap_px >= cfar.context_px

    stated = load(_config(tmp_path, _MINIMAL + "tile_px: 256\n"))
    overlap = cfar.preferred_tiling.overlap_px
    assert stated.tiling_for(cfar) == Tiling(tile_px=256, overlap_px=overlap)

