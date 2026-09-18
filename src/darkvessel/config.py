"""Run configuration: the arguments `pipeline.run` needs, gathered from a config file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import yaml

from darkvessel.fusion.azimuth import Geometry


@dataclass(frozen=True)
class RunConfig:
    scene_dir: Path
    ais_path: Path | None
    tolerance_m: float
    max_gap: timedelta
    tile_px: int
    overlap_px: int
    detector_threshold: float
    output_path: Path
    geometry: Geometry | None


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
        tile_px=int(raw.get("tile_px", 128)),
        overlap_px=int(raw.get("overlap_px", 32)),
        detector_threshold=float(raw.get("detector_threshold", 0.5)),
        output_path=Path(raw.get("output_path", "outputs/detections.gpkg")),
        geometry=geometry,
    )
