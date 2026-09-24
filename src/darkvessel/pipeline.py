"""The seam: a scene goes in, a detector is injected, both sides of the fusion come out.

Everything else in this project — the CLI, config loading, synthetic fixtures, real
ingestion — exists to build the arguments this one function takes. The detector (and the
optional embedder) arrive as parameters rather than imports, which is what lets the whole
chain run, and be tested, with a deterministic stand-in: no weights, no GPU, no network.

A run answers two questions, not one. *Which detections did no declaration explain* is the
dark-vessel question; *which declarations did no detection explain* is its mirror, and the
assignment settles both at once. `Fusion` carries them together because separating them
invites two answers that disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import geopandas as gpd
import pandas as pd

from darkvessel.context.gee_layers import without_context
from darkvessel.data.land import LandMask
from darkvessel.data.provenance import attach_provenance
from darkvessel.data.scene import Scene
from darkvessel.data.tiling import Tiling
from darkvessel.detect.detector import Detector
from darkvessel.detect.geo import to_ground
from darkvessel.detect.infer import detect_scene
from darkvessel.embed.crops import crops_for
from darkvessel.embed.embedder import Embedder, attach
from darkvessel.fusion.azimuth import Geometry
from darkvessel.fusion.declarations import SMALLEST_DETECTABLE_M, Agreement, review
from darkvessel.fusion.declarations import agreement as _agreement
from darkvessel.fusion.match import classify
from darkvessel.fusion.reception import (
    RECEPTION_FLOOR,
    Coverage,
    coverage_for,
    without_coverage,
)
from darkvessel.fusion.register import Register, without_a_register


@dataclass(frozen=True)
class Fusion:
    """One scene's answer, from both sensors' points of view.

    `detections` is the radar side, classified and carrying its reception estimate.
    `declarations` is the AIS side, each declaration judged against what the radar drew.
    `coverage` is the reception model the two were qualified with, or `None` where there was
    no archive to estimate one from.
    """

    detections: gpd.GeoDataFrame
    declarations: gpd.GeoDataFrame
    coverage: Coverage | None = None

    def agreement(self) -> Agreement:
        """Where the two sensors agreed and where each saw something alone."""
        return _agreement(self.detections, self.declarations)


def run(
    *,
    scene: Scene,
    ais: gpd.GeoDataFrame | None,
    detector: Detector,
    tiling: Tiling,
    tolerance_m: float,
    max_gap: timedelta,
    geometry: Geometry | None = None,
    embedder: Embedder | None = None,
    structures: Register | None = None,
    coverage: Coverage | None = None,
    reception_floor: float = RECEPTION_FLOOR,
    smallest_detectable_m: float | None = SMALLEST_DETECTABLE_M,
    land: LandMask | None = None,
) -> Fusion:
    """Run the chain over one scene and return its detections, georeferenced and classified.

    `land`, when given, is not searched: see `data.land`.
    """
    mask = None if land is None or land.empty else land.mask_for(scene)
    return fuse(
        scene=scene,
        found=detect_scene(scene.image, detector, tiling, mask=mask),
        ais=ais,
        tolerance_m=tolerance_m,
        max_gap=max_gap,
        geometry=geometry,
        embedder=embedder,
        structures=structures,
        coverage=coverage,
        reception_floor=reception_floor,
        smallest_detectable_m=smallest_detectable_m,
        land=land,
    )


def fuse(
    *,
    scene: Scene,
    found: pd.DataFrame,
    ais: gpd.GeoDataFrame | None,
    tolerance_m: float,
    max_gap: timedelta,
    geometry: Geometry | None = None,
    embedder: Embedder | None = None,
    structures: Register | None = None,
    coverage: Coverage | None = None,
    reception_floor: float = RECEPTION_FLOOR,
    smallest_detectable_m: float | None = SMALLEST_DETECTABLE_M,
    land: LandMask | None = None,
) -> Fusion:
    """Everything after detection: `found` is `detect_scene`'s pixel detections for `scene`.

    Split out because detection is the one step that reads every pixel, and none of the fusion
    parameters change it — a caller re-running with a new tolerance can reuse `found`.

    `coverage` is estimated from `ais` when it is not supplied. A caller re-running one scene
    many times should build it once with `fusion.reception.coverage_for` and pass it in: it
    depends only on the archive and `max_gap`, so rebuilding it per tolerance is wasted work.

    `land` must be the mask `found` was detected under: it decides which declarations were
    never searched for.
    """
    matched = classify(
        to_ground(found, scene), ais, scene.acquired_at, tolerance_m, max_gap, geometry
    )
    detections = matched.detections

    # Structure exclusion happens after matching, never before: a detection AIS explains is a
    # match whatever else stands at that coordinate.
    detections = (
        without_a_register(detections) if structures is None else structures.mark(detections)
    )
    # Reception comes last of the three, so it only ever judges what is still unexplained: a
    # registered structure is a better account of a detection than "we could not have heard it".
    if coverage is None:
        coverage = coverage_for(ais, max_gap)
    detections = (
        without_coverage(detections)
        if coverage is None
        else coverage.mark(detections, floor=reception_floor)
    )
    # Contextual columns are always present, empty here, filled later by `darkvessel context` —
    # a layer's schema must not depend on whether a credentialed stage ran.
    detections = without_context(detections)
    detections = attach_provenance(detections, scene)

    if embedder is not None:
        crops = crops_for(
            scene.image, found, crop_px=embedder.crop_px, margin_px=embedder.margin_px
        )
        detections = attach(detections, embedder(crops))

    declarations = review(
        matched.declared,
        detections,
        scene.footprint,
        tolerance_m=tolerance_m,
        smallest_detectable_m=smallest_detectable_m,
        masked=None if land is None else land.geometry,
    )
    if not declarations.empty:
        declarations["acquired_at"] = scene.acquired_at
        declarations = attach_provenance(declarations, scene)

    return Fusion(detections=detections, declarations=declarations, coverage=coverage)
