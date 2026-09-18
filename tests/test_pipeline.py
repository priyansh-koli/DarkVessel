"""End-to-end runs over the synthetic fixture, including the README's quoted numbers."""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from darkvessel.context.schema import CONTEXT_COLUMNS
from darkvessel.data.synthetic import (
    ACQUIRED_AT,
    HEADING_DEG,
    INCIDENCE_DEG,
    PIXEL_SIZE_M,
    TOLERANCE_M,
    _ais,
    _by_name,
    _scene,
)
from darkvessel.data.tiling import Tiling
from darkvessel.detect.stub import BrightPixelDetector
from darkvessel.fusion.azimuth import Geometry
from darkvessel.fusion.match import DARK, MATCHED, UNSEARCHED
from darkvessel.fusion.register import STRUCTURE, Register
from darkvessel.pipeline import run

_GEOMETRY = Geometry(heading_deg=HEADING_DEG, incidence_deg=INCIDENCE_DEG)

# A fixture target is painted as a 2x2 plateau whose centre sits half a pixel off the target's
# own ground coordinate, so a perfectly-placed declaration still matches a pixel-scale distance
# away rather than at exactly zero.
_PIXEL_RESIDUAL_M = 2 * PIXEL_SIZE_M


def _run(**overrides):
    kwargs = dict(
        scene=_scene(),
        ais=_ais(),
        detector=BrightPixelDetector(threshold=0.5),
        tiling=Tiling(tile_px=128, overlap_px=32),
        tolerance_m=TOLERANCE_M,
        max_gap=timedelta(minutes=10),
        geometry=_GEOMETRY,
    )
    kwargs.update(overrides)
    return run(**kwargs)


def test_the_synthetic_run_reproduces_the_readme_numbers():
    """"5 detections ... 4 matched, 1 dark, 0 at a fixed structure" — the quick start's
    quoted output, which the fixture exists to produce."""
    detections = _run()
    counts = detections["status"].value_counts()
    assert len(detections) == 5
    assert int(counts.get(MATCHED, 0)) == 4
    assert int(counts.get(DARK, 0)) == 1
    assert int(counts.get(STRUCTURE, 0)) == 0


def test_the_one_dark_detection_is_the_vessel_that_declares_nothing():
    detections = _run()
    dark = detections[detections["status"] == DARK].iloc[0]
    target = _by_name("dark_vessel")
    assert dark["x"] == pytest.approx(target.x, abs=5.0)
    assert dark["y"] == pytest.approx(target.y, abs=5.0)


def test_the_dark_vessel_stays_dark_despite_a_declaration_within_tolerance_of_it():
    """`simple_match`'s declaration lies inside the tolerance of the dark detection too, and
    is only kept from it by the one-to-one assignment awarding it to the nearer detection.
    If this ever comes back matched, the matcher has started letting one declaration explain
    two detections — the exact failure that would hide a real dark vessel."""
    detections = _run()
    dark_row = detections[detections["status"] == DARK].iloc[0]
    ais = _ais()
    contested = ais[ais["mmsi"] == "219000004"].geometry.iloc[0]

    reach_of_the_dark_detection = dark_row.geometry.distance(contested)
    assert reach_of_the_dark_detection < TOLERANCE_M

    matched = detections[detections["status"] == MATCHED]
    assert matched[matched["mmsi"] == "219000004"]["match_distance_m"].iloc[0] < (
        reach_of_the_dark_detection
    )


def test_without_the_azimuth_correction_the_fast_vessel_is_reported_dark():
    """The correction is not cosmetic: dropping it turns a declared vessel into a dark one,
    which is the false positive the whole step exists to prevent."""
    detections = _run(geometry=None)
    counts = detections["status"].value_counts()
    assert int(counts.get(MATCHED, 0)) == 3
    assert int(counts.get(DARK, 0)) == 2


def test_the_interpolated_vessel_is_matched_on_a_bracket_not_a_single_report():
    detections = _run()
    matched = detections[detections["status"] == MATCHED]
    interpolated = matched[matched["mmsi"] == "219000002"].iloc[0]
    assert interpolated["position_basis"] == "interpolated"
    assert interpolated["match_distance_m"] < _PIXEL_RESIDUAL_M


def test_the_azimuth_vessel_is_matched_on_a_nearest_report_after_a_large_shift():
    detections = _run()
    matched = detections[detections["status"] == MATCHED]
    azimuth = matched[matched["mmsi"] == "219000003"].iloc[0]
    assert azimuth["position_basis"] == "nearest"
    assert azimuth["azimuth_shift_m"] > TOLERANCE_M  # corrected further than the tolerance itself
    assert azimuth["match_distance_m"] < _PIXEL_RESIDUAL_M


def test_no_ais_at_all_is_unsearched_not_dark():
    """"Searched and found nothing" and "never searched" are different claims; only one of
    them supports calling a vessel dark."""
    detections = _run(ais=None)
    assert set(detections["status"]) == {UNSEARCHED}
    assert detections["declarations_searched"].isna().all()


def test_a_registered_structure_reclassifies_the_dark_detection():
    dark = _by_name("dark_vessel")
    register = Register(
        positions=pd.DataFrame({"x": [dark.x], "y": [dark.y], "acquisitions": [4], "crops": [4]}),
        tolerance_m=100.0,
    )
    detections = _run(structures=register)
    counts = detections["status"].value_counts()
    assert int(counts.get(STRUCTURE, 0)) == 1
    assert int(counts.get(DARK, 0)) == 0
    assert int(counts.get(MATCHED, 0)) == 4  # matches are never overridden


def test_context_and_provenance_columns_are_always_present():
    detections = _run()
    assert set(CONTEXT_COLUMNS) <= set(detections.columns)
    assert set(detections["scene"]) == {"synthetic-scene-1"}
    assert detections["acquired_at"].iloc[0] == pd.Timestamp(ACQUIRED_AT)


def test_an_embedder_is_optional_and_attaches_one_row_per_detection():
    @dataclass
    class _Embedder:
        crop_px: int = 32
        margin_px: int = 4

        def __call__(self, crops: np.ndarray) -> np.ndarray:
            return np.arange(len(crops) * 3, dtype=float).reshape(len(crops), 3)

    assert "embedding" not in _run().columns
    with_embeddings = _run(embedder=_Embedder())
    assert len(with_embeddings["embedding"].iloc[0]) == 3


def test_the_tolerance_is_recorded_on_every_row():
    """A dark claim is a claim about evidence searched, so the row has to say how far it looked."""
    detections = _run()
    assert (detections["tolerance_m"] == TOLERANCE_M).all()
    assert (detections["declarations_searched"] == 4).all()
