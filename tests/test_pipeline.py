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
    RECEPTION_CELL_M,
    RECEPTION_FLOOR,
    TOLERANCE_M,
    _ais,
    _by_name,
    _scene,
)
from darkvessel.data.tiling import Tiling
from darkvessel.detect.stub import BrightPixelDetector
from darkvessel.fusion.azimuth import Geometry
from darkvessel.fusion.declarations import (
    BELOW_DETECTABLE,
    EXPLAINED,
    OUTSIDE_SCENE,
    UNDETECTED,
)
from darkvessel.fusion.match import DARK, MATCHED, UNSEARCHED
from darkvessel.fusion.reception import SHADOWED, coverage_for
from darkvessel.fusion.register import STRUCTURE, Register
from darkvessel.pipeline import run

_GEOMETRY = Geometry(heading_deg=HEADING_DEG, incidence_deg=INCIDENCE_DEG)

# A fixture target is painted as a 2x2 plateau whose centre sits half a pixel off the target's
# own ground coordinate, so a perfectly-placed declaration still matches a pixel-scale distance
# away rather than at exactly zero.
_PIXEL_RESIDUAL_M = 2 * PIXEL_SIZE_M


def _fuse(**overrides):
    """A run at the fixture's intended settings, both sides returned."""
    ais = overrides.get("ais", _ais())
    max_gap = overrides.get("max_gap", timedelta(minutes=10))
    kwargs = dict(
        scene=_scene(),
        ais=ais,
        detector=BrightPixelDetector(threshold=0.5),
        tiling=Tiling(tile_px=128, overlap_px=32),
        tolerance_m=TOLERANCE_M,
        max_gap=max_gap,
        geometry=_GEOMETRY,
        coverage=coverage_for(ais, max_gap, cell_m=RECEPTION_CELL_M),
        reception_floor=RECEPTION_FLOOR,
    )
    kwargs.update(overrides)
    return run(**kwargs)


def _run(**overrides):
    return _fuse(**overrides).detections


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
    assert (detections["declarations_searched"] == 8).all()


# ───────────────────────── AIS reception ─────────────────────────


def test_the_dark_vessel_is_dark_where_the_archive_hears_well():
    """The point of the reception estimate: this detection's silence means something because
    a busy lane beside it shows the archive would have placed a transmitting vessel here."""
    detections = _run()
    dark = detections[detections["status"] == DARK].iloc[0]
    assert dark["reception_basis"] == "estimated"
    assert dark["reception_p"] == pytest.approx(1.0)
    assert dark["reception_intervals"] >= 5


def test_a_dark_detection_in_a_reception_shadow_is_not_reported_dark():
    """`faint_dark` stands beside a vessel heard once an hour, so the archive would have placed
    a transmitting vessel there only a third of the time. Silence there is not evidence."""
    detections = _fuse(detector=BrightPixelDetector(threshold=0.3)).detections
    shadowed = detections[detections["status"] == SHADOWED]
    assert len(shadowed) == 1
    target = _by_name("faint_dark")
    assert shadowed.iloc[0]["x"] == pytest.approx(target.x, abs=5.0)
    assert shadowed.iloc[0]["reception_p"] == pytest.approx(1 / 3, abs=0.01)


def test_no_reception_estimate_leaves_a_detection_dark_rather_than_shadowed():
    """An absence of evidence is not evidence. `faint_trawler` sits beside neither lane, so
    nothing measures the archive's reach there — and it stays dark, next to a shadowed row."""
    detections = _fuse(detector=BrightPixelDetector(threshold=0.3)).detections
    trawler = _by_name("faint_trawler")
    near = detections[
        (detections["x"] - trawler.x).abs().lt(10) & (detections["y"] - trawler.y).abs().lt(10)
    ]
    assert near.iloc[0]["status"] == DARK
    assert near.iloc[0]["reception_basis"] == "insufficient_evidence"
    assert pd.isna(near.iloc[0]["reception_p"])


def test_widening_the_max_gap_lifts_the_shadow():
    """Reception is defined against `max_gap`, so the one control moves both halves of the
    claim: a wider gap covers the same hourly reports and the shadow lifts."""
    tight = _fuse(detector=BrightPixelDetector(threshold=0.3))
    wide = _fuse(detector=BrightPixelDetector(threshold=0.3), max_gap=timedelta(minutes=90))
    assert (tight.detections["status"] == SHADOWED).sum() == 1
    assert (wide.detections["status"] == SHADOWED).sum() == 0


def test_a_floor_of_zero_reports_reception_without_acting_on_it():
    """The floor is a policy. At zero, every row still carries its estimate and no status moves."""
    detections = _fuse(detector=BrightPixelDetector(threshold=0.3), reception_floor=0.0).detections
    assert SHADOWED not in set(detections["status"])
    assert detections["reception_p"].notna().any()


def test_without_an_archive_the_reception_columns_are_present_and_empty():
    """A column's presence must never depend on whether an optional stage ran."""
    detections = _run(ais=None, coverage=None)
    assert {"reception_p", "reception_basis", "reception_intervals"} <= set(detections.columns)
    assert detections["reception_p"].isna().all()


# ───────────────────────── the declaration side ─────────────────────────


def test_every_declaration_gets_a_verdict_against_the_radar():
    declarations = _fuse().declarations
    counts = declarations["status"].value_counts()
    assert int(counts.get(EXPLAINED, 0)) == 4
    assert int(counts.get(UNDETECTED, 0)) == 2
    assert int(counts.get(BELOW_DETECTABLE, 0)) == 1
    assert int(counts.get(OUTSIDE_SCENE, 0)) == 1


def test_an_undetected_declaration_reports_how_near_the_radar_came():
    """A status alone never says "the nearest detection was 240 m away and the bar was 200"."""
    declarations = _fuse().declarations
    undetected = declarations[declarations["mmsi"] == "219100001"].iloc[0]
    assert undetected["status"] == UNDETECTED
    assert undetected["nearest_detection_m"] > TOLERANCE_M
    assert undetected["detection"] is pd.NA or pd.isna(undetected["detection"])


def test_a_short_vessel_is_excused_but_one_of_unknown_length_is_not():
    """A miss below the detector's floor says nothing about that vessel. An *unknown* length is
    not a small one, so it stays a finding — the same rule `unsearched` follows."""
    declarations = _fuse().declarations
    assert declarations[declarations["mmsi"] == "219100003"].iloc[0]["status"] == BELOW_DETECTABLE

    blind = _fuse(smallest_detectable_m=None).declarations
    assert BELOW_DETECTABLE not in set(blind["status"])
    assert int((blind["status"] == UNDETECTED).sum()) == 3


def test_a_declaration_outside_the_image_is_not_a_finding():
    """Nothing about it was searched: the declaration side's `unsearched`."""
    declarations = _fuse().declarations
    outside = declarations[declarations["mmsi"] == "219100004"].iloc[0]
    assert outside["status"] == OUTSIDE_SCENE
    assert not _scene().footprint.contains(outside.geometry)


def test_the_two_sides_agree_on_one_set_of_counts():
    """`Agreement` is derived from both frames, so it cannot drift from either."""
    fusion = _fuse()
    found = fusion.agreement()
    assert found.both == int((fusion.detections["status"] == MATCHED).sum())
    assert found.radar_only == int(
        fusion.detections["status"].isin([DARK, SHADOWED]).sum()
    )
    assert found.ais_only == int((fusion.declarations["status"] == UNDETECTED).sum())
    assert found.apparent_recall == pytest.approx(found.both / (found.both + found.ais_only))


def test_with_no_archive_there_is_no_other_side_to_report():
    fusion = _fuse(ais=None, coverage=None)
    assert fusion.declarations.empty
    assert fusion.agreement().apparent_recall is None
