"""A static viewer whose controls still work: every control position, run ahead of time.

A static host cannot re-run the pipeline, so `darkvessel render` runs it for every position
the fusion and detector controls can take and writes each distinct result once. The frontend
looks a position up in `manifest.json` instead of calling `api/run`.

The grid is large (tolerance x azimuth x threshold x max gap) but the work is not: detection
depends only on the threshold, and declared positions only on the max gap, so thresholds that
find the same pixels, and gaps that declare the same positions, are run once and shared.
Results are stored without their `config` block — the frontend already knows the request it
made — which is what lets identical outcomes collapse into one file.

The structure register is not baked: a click can land anywhere. It does not need to be,
because registering only ever turns a `dark` row into `structure` after matching (see
`fusion.register`), so the frontend applies it to a baked result exactly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

from darkvessel.web.payload import RunRequest, Viewer

# The live viewer's slider ranges (see static/index.html), so both modes offer the same
# positions. Max gap is the exception: its minute-by-minute slider becomes these steps.
TOLERANCES_M = [float(v) for v in range(25, 1001, 25)]
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 21)]
MAX_GAPS_MINUTES = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0]
AZIMUTH = [True, False]

# Row-major order of `manifest["runs"]`; the frontend indexes it the same way.
ORDER = ["detector_threshold", "max_gap_minutes", "apply_azimuth", "tolerance_m"]


def bake(viewer: Viewer, default: RunRequest, out_dir: Path) -> dict[str, Any]:
    """Write `runs/<id>.json` for every distinct result and return the manifest indexing them."""
    grid = {
        "detector_threshold": _with(THRESHOLDS, default.detector_threshold),
        "max_gap_minutes": _with(MAX_GAPS_MINUTES, default.max_gap_minutes),
        "apply_azimuth": AZIMUTH,
        "tolerance_m": _with(TOLERANCES_M, default.tolerance_m),
    }

    # One representative per equivalence class: the first threshold that found these pixels,
    # the first gap that declared these positions.
    threshold_rep = _representatives(
        grid["detector_threshold"],
        lambda t: viewer.detections_key(t, default.tile_px, default.overlap_px),
    )
    gap_rep = _representatives(grid["max_gap_minutes"], viewer.declarations_key)

    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ids: dict[str, int] = {}  # content hash -> file id
    by_inputs: dict[tuple, int] = {}
    runs = []
    for threshold, gap, azimuth, tolerance in product(*(grid[name] for name in ORDER)):
        inputs = (threshold_rep[threshold], gap_rep[gap], azimuth, tolerance)
        if inputs not in by_inputs:
            payload = viewer.run(
                replace(
                    default,
                    detector_threshold=inputs[0],
                    max_gap_minutes=inputs[1],
                    apply_azimuth=azimuth,
                    tolerance_m=tolerance,
                    register_xy=[],
                )
            )
            payload.pop("config")
            text = json.dumps(payload, separators=(",", ":"))
            digest = hashlib.sha1(text.encode()).hexdigest()
            if digest not in ids:
                ids[digest] = len(ids)
                (runs_dir / f"{ids[digest]}.json").write_text(text)
            by_inputs[inputs] = ids[digest]
        runs.append(by_inputs[inputs])

    return {
        "version": 1,
        "order": ORDER,
        "grid": grid,
        "defaults": {
            "detector_threshold": default.detector_threshold,
            "max_gap_minutes": default.max_gap_minutes,
            "apply_azimuth": default.apply_azimuth,
            "tolerance_m": default.tolerance_m,
            "register_tolerance_m": default.register_tolerance_m,
        },
        "tile_px": default.tile_px,
        "overlap_px": default.overlap_px,
        "distinct_runs": len(ids),
        "runs": runs,
    }


def _with(values: list[float], value: float) -> list[float]:
    """The grid always contains the configured value, so the default position is exact."""
    return sorted({*values, float(value)})


def _representatives(values: list[float], key) -> dict[float, float]:
    first: dict[str, float] = {}
    return {value: first.setdefault(key(value), value) for value in values}
