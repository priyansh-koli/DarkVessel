"""The other side of the assignment: declarations the radar drew nothing for.

`fusion.match` asks which detections no declaration explains. The same assignment answers the
mirror question for free — which declarations no detection explains — and the pipeline used to
throw that half away. It is worth keeping, for two reasons.

**It is a finding of its own.** A vessel broadcasting a position where C-band radar sees
nothing is either a position that is not true, or a hull the detector missed. Neither is
nothing. The first is what AIS spoofing looks like from orbit; the second is what a detector's
recall looks like on real water rather than on a labelled benchmark.

**It measures the detector without labels.** Take the vessels that declared themselves, inside
the scene, large enough for this detector to have a fair chance: the fraction the radar also
drew is an estimate of recall on this scene, from data nobody annotated. `Agreement` reports
it, with the biases that make it an estimate rather than a measurement stated on the tin.

Five verdicts, and the four that are not findings matter as much as the one that is:

- **explained** — a detection stands where this vessel declared.
- **undetected** — inside the scene, big enough to expect, and the radar drew nothing. The
  finding.
- **outside_scene** — the declared position is not in the image. Nothing was searched, so
  nothing can be concluded; the exact counterpart of a detection's `unsearched`.
- **masked** — inside the image but on land, or within the land buffer, which the detector
  was not run over (`data.land`). Not searched, so not a finding.
- **below_detectable** — shorter than the smallest vessel this detector is trusted to find, so
  a miss was expected and says nothing about this vessel. A declaration whose length the
  archive never gave is *not* put here: an unknown length is not a small one.

Positions are the radar-corrected ones, because those are the positions that were searched.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from darkvessel.fusion.match import DARK, MATCHED
from darkvessel.fusion.reception import SHADOWED

EXPLAINED = "explained"
UNDETECTED = "undetected"
OUTSIDE_SCENE = "outside_scene"
BELOW_DETECTABLE = "below_detectable"
MASKED = "masked"

# The shortest vessel a Sentinel-1 IW detector is given a fair chance at. At 10 m ground
# spacing a 20 m hull is two pixels, which is the floor of what any of this project's
# detectors resolve; below it a miss is a statement about the sensor, not about the vessel.
# A run configuration should set this from its own detector's benchmark.
SMALLEST_DETECTABLE_M = 20.0

_COLUMNS = {
    "mmsi": "string",
    "status": "string",
    "length_m": "float64",
    "detection": "Int64",
    "match_distance_m": "float64",
    "nearest_detection_m": "float64",
    "position_basis": "string",
    "position_age_s": "float64",
    "position_span_s": "float64",
    "azimuth_shift_m": "float64",
    "tolerance_m": "float64",
    "smallest_detectable_m": "float64",
}


@dataclass(frozen=True)
class Agreement:
    """Where the two sensors agreed, where each saw something alone, and what that implies.

    `apparent_recall` is the share of declared, detectable, in-scene vessels the radar also
    drew. It is an *estimate* of the detector's recall on this scene, and it is biased in
    ways worth naming: vessels that declare themselves are larger and more cooperative than
    those that do not, a spoofed declaration counts against the detector although nothing was
    ever there to find, and a declaration placed by interpolation carries its own error into
    the tolerance. It is reported because a number with its biases stated is more use than no
    number at all, and because it needs no labels.
    """

    both: int
    radar_only: int
    ais_only: int
    outside_scene: int
    below_detectable: int
    masked: int = 0

    @property
    def apparent_recall(self) -> float | None:
        declared = self.both + self.ais_only
        return self.both / declared if declared else None

    def line(self) -> str:
        if self.apparent_recall is None:
            return "no declared, detectable, in-scene vessel to compare the radar against"
        return (
            f"radar and AIS agree on {self.both} of {self.both + self.ais_only} declared, "
            f"detectable, in-scene vessels (apparent recall {self.apparent_recall:.2f}); "
            f"{self.radar_only} detection{'' if self.radar_only == 1 else 's'} "
            f"no declaration explains"
        )


def review(
    declared: gpd.GeoDataFrame,
    detections: gpd.GeoDataFrame,
    footprint: shapely.Geometry | None,
    tolerance_m: float,
    smallest_detectable_m: float | None = SMALLEST_DETECTABLE_M,
    masked: shapely.Geometry | None = None,
) -> gpd.GeoDataFrame:
    """Give every declaration a verdict against what the radar drew. See the module docstring.

    `declared` is `fusion.match.Match.declared` — positions already moved to where the radar
    would have drawn them, carrying the detection each one explains. `footprint` bounds what
    was searched; `None` means no footprint was supplied, and then nothing is ruled outside it.
    `masked` is the part of the footprint the detector was not run over (`data.land`), or
    `None` where nothing was masked.
    """
    reviewed = declared.copy()
    if reviewed.empty:
        return _empty(declared.crs)

    reviewed["nearest_detection_m"] = _nearest(reviewed, detections)
    reviewed["tolerance_m"] = float(tolerance_m)
    reviewed["smallest_detectable_m"] = (
        np.nan if smallest_detectable_m is None else float(smallest_detectable_m)
    )

    explained = reviewed["detection"].notna().to_numpy()
    inside = _inside(reviewed, footprint)
    on_mask = (
        np.zeros(len(reviewed), dtype=bool)
        if masked is None or masked.is_empty
        else shapely.intersects(masked, reviewed.geometry.values)
    )
    length = reviewed["length_m"].to_numpy(dtype=float)
    # `length < floor` is False for nan, which is the wanted answer: an unknown length is not
    # evidence of a small vessel, so such a declaration stays a finding rather than an excuse.
    with np.errstate(invalid="ignore"):
        too_small = (
            np.zeros(len(reviewed), dtype=bool)
            if smallest_detectable_m is None
            else length < float(smallest_detectable_m)
        )

    status = np.where(
        explained,
        EXPLAINED,
        np.where(
            ~inside,
            OUTSIDE_SCENE,
            np.where(on_mask, MASKED, np.where(too_small, BELOW_DETECTABLE, UNDETECTED)),
        ),
    )
    reviewed["status"] = pd.Series(status, index=reviewed.index, dtype="string")

    ordered = [name for name in _COLUMNS if name in reviewed.columns]
    rest = [c for c in reviewed.columns if c not in ordered and c != reviewed.geometry.name]
    return reviewed[ordered + rest + [reviewed.geometry.name]]


def agreement(detections: gpd.GeoDataFrame, declarations: gpd.GeoDataFrame) -> Agreement:
    """Count the two sides against each other. Both frames come from one `fuse`."""
    detection_status = detections["status"] if "status" in detections else pd.Series(dtype="string")
    declaration_status = (
        declarations["status"] if "status" in declarations else pd.Series(dtype="string")
    )
    return Agreement(
        both=int((detection_status == MATCHED).sum()),
        radar_only=int(detection_status.isin([DARK, SHADOWED]).sum()),
        ais_only=int((declaration_status == UNDETECTED).sum()),
        outside_scene=int((declaration_status == OUTSIDE_SCENE).sum()),
        below_detectable=int((declaration_status == BELOW_DETECTABLE).sum()),
        masked=int((declaration_status == MASKED).sum()),
    )


def _nearest(declared: gpd.GeoDataFrame, detections: gpd.GeoDataFrame) -> np.ndarray:
    """Distance to the closest detection, whatever the tolerance.

    Reported even when it is far outside the tolerance, because "the nearest detection was
    260 m away and the bar was 200" is the single most useful thing to know about an
    undetected declaration, and a status alone never says it.
    """
    if detections is None or len(detections) == 0:
        return np.full(len(declared), np.nan)
    from scipy.spatial import cKDTree

    tree = cKDTree(shapely.get_coordinates(detections.geometry.values))
    distances, _ = tree.query(shapely.get_coordinates(declared.geometry.values), k=1)
    return np.asarray(distances, dtype=float)


def _inside(declared: gpd.GeoDataFrame, footprint: shapely.Geometry | None) -> np.ndarray:
    if footprint is None:
        return np.ones(len(declared), dtype=bool)
    return shapely.contains(footprint, declared.geometry.values)


def _empty(crs) -> gpd.GeoDataFrame:
    """The declaration layer's schema with no rows in it, so a reader never has to branch."""
    return gpd.GeoDataFrame(
        {name: pd.array([], dtype=dtype) for name, dtype in _COLUMNS.items()},
        geometry=[],
        crs=crs,
    )
