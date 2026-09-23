"""The seam: a scene goes in, a detector is injected, classified detections come out.

Everything else in this project — the CLI, config loading, synthetic fixtures, real
ingestion — exists to build the arguments this one function takes. The detector (and the
optional embedder) arrive as parameters rather than imports, which is what lets the whole
chain run, and be tested, with a deterministic stand-in: no weights, no GPU, no network.
"""

from __future__ import annotations

from datetime import timedelta

import geopandas as gpd
import pandas as pd

from darkvessel.context.gee_layers import without_context
from darkvessel.data.provenance import attach_provenance
from darkvessel.data.scene import Scene
from darkvessel.data.tiling import Tiling
from darkvessel.detect.detector import Detector
from darkvessel.detect.geo import to_ground
from darkvessel.detect.infer import detect_scene
from darkvessel.embed.crops import crops_for
from darkvessel.embed.embedder import Embedder, attach
from darkvessel.fusion.azimuth import Geometry
from darkvessel.fusion.match import classify
from darkvessel.fusion.register import Register, without_a_register


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
) -> gpd.GeoDataFrame:
    """Run the chain over one scene and return its detections, georeferenced and classified."""
    return fuse(
        scene=scene,
        found=detect_scene(scene.image, detector, tiling),
        ais=ais,
        tolerance_m=tolerance_m,
        max_gap=max_gap,
        geometry=geometry,
        embedder=embedder,
        structures=structures,
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
) -> gpd.GeoDataFrame:
    """Everything after detection: `found` is `detect_scene`'s pixel detections for `scene`.

    Split out because detection is the one step that reads every pixel, and none of the fusion
    parameters change it — a caller re-running with a new tolerance can reuse `found`.
    """
    detections = classify(
        to_ground(found, scene), ais, scene.acquired_at, tolerance_m, max_gap, geometry
    ).detections
    # Structure exclusion happens after matching, never before: a detection AIS explains is a
    # match whatever else stands at that coordinate.
    detections = (
        without_a_register(detections) if structures is None else structures.mark(detections)
    )
    # Contextual columns are always present, empty here, filled later by `darkvessel context` —
    # a layer's schema must not depend on whether a credentialed stage ran.
    detections = without_context(detections)
    detections = attach_provenance(detections, scene)

    if embedder is None:
        return detections

    crops = crops_for(scene.image, found, crop_px=embedder.crop_px, margin_px=embedder.margin_px)
    return attach(detections, embedder(crops))
