"""Run configuration: the arguments `pipeline.run` needs, gathered from a config file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import yaml

from darkvessel.data.tiling import Tiling
from darkvessel.fusion.azimuth import Geometry
from darkvessel.fusion.declarations import SMALLEST_DETECTABLE_M
from darkvessel.fusion.reception import MIN_INTERVALS, RECEPTION_CELL_M, RECEPTION_FLOOR


@dataclass(frozen=True)
class RunConfig:
    scene_dir: Path
    ais_path: Path | None
    tolerance_m: float
    max_gap: timedelta
    # `None` leaves the choice to the detector's `preferred_tiling` (see `tiling_for`).
    tile_px: int | None
    overlap_px: int | None
    detector_threshold: float | None
    output_path: Path
    geometry: Geometry | None
    detector: str = "stub"
    detector_weights: Path | None = None
    # Reception, and the bar a dark claim has to clear. `reception_floor` is a policy, not a
    # measurement: it says how well the archive must reach a place before its silence there
    # counts as evidence. Zero reports reception without ever acting on it.
    reception_cell_m: float = RECEPTION_CELL_M
    reception_floor: float = RECEPTION_FLOOR
    reception_min_intervals: int = MIN_INTERVALS
    # The shortest vessel this detector is trusted to find. `None` turns the check off, and
    # then no declaration is ever excused as too small to expect.
    smallest_detectable_m: float | None = SMALLEST_DETECTABLE_M
    # Land polygons not to search (any vector file), and how far out to sea to extend them.
    # See `data.land`. `None` searches the whole scene.
    land_path: Path | None = None
    land_buffer_m: float = 0.0

    def tiling_for(self, detector) -> Tiling:
        """The configured tiling, or the detector's own where the configuration names none.

        Either value can be left out on its own: the other still comes from the detector.
        """
        preferred = getattr(detector, "preferred_tiling", None) or DEFAULT_TILING
        return Tiling(
            tile_px=self.tile_px if self.tile_px is not None else preferred.tile_px,
            overlap_px=self.overlap_px if self.overlap_px is not None else preferred.overlap_px,
        )


# For a detector that states no preference: the stand-in, whose fixture was built at this size.
DEFAULT_TILING = Tiling(tile_px=128, overlap_px=32)


def load(path: str | Path) -> RunConfig:
    """Load a run configuration from YAML. See `configs/pipeline.yaml` for the shape."""
    raw = yaml.safe_load(Path(path).read_text())

    geometry_raw = raw.get("geometry")
    geometry = (
        Geometry(
            heading_deg=geometry_raw["heading_deg"],
            incidence_deg=geometry_raw["incidence_deg"],
            shift_per_mps=geometry_raw.get("shift_per_mps", 113.0),
        )
        if geometry_raw
        else None
    )

    return RunConfig(
        scene_dir=Path(raw["scene_dir"]),
        ais_path=Path(raw["ais_path"]) if raw.get("ais_path") else None,
        tolerance_m=float(raw["tolerance_m"]),
        max_gap=timedelta(minutes=float(raw.get("max_gap_minutes", 10))),
        tile_px=int(raw["tile_px"]) if raw.get("tile_px") is not None else None,
        overlap_px=int(raw["overlap_px"]) if raw.get("overlap_px") is not None else None,
        # Each detector reads its threshold in its own units (brightness, clutter standard
        # deviations, heatmap score); left out, cfar and cnn use their calibrated defaults.
        detector_threshold=(
            float(raw["detector_threshold"])
            if raw.get("detector_threshold") is not None
            else (0.5 if raw.get("detector", "stub") == "stub" else None)
        ),
        output_path=Path(raw.get("output_path", "outputs/detections.gpkg")),
        geometry=geometry,
        detector=str(raw.get("detector", "stub")),
        detector_weights=Path(raw["detector_weights"]) if raw.get("detector_weights") else None,
        reception_cell_m=float(raw.get("reception_cell_m", RECEPTION_CELL_M)),
        reception_floor=float(raw.get("reception_floor", RECEPTION_FLOOR)),
        reception_min_intervals=int(raw.get("reception_min_intervals", MIN_INTERVALS)),
        # Written out rather than `raw.get(...) or default`, so an explicit `null` turns the
        # check off instead of quietly restoring the default.
        smallest_detectable_m=(
            None
            if "smallest_detectable_m" in raw and raw["smallest_detectable_m"] is None
            else float(raw.get("smallest_detectable_m", SMALLEST_DETECTABLE_M))
        ),
        land_path=Path(raw["land_path"]) if raw.get("land_path") else None,
        land_buffer_m=float(raw.get("land_buffer_m") or 0.0),
    )
