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

The reception floor is left out for the same reason, and it is the reason worth stating twice:
every detection already carries its own `reception_p`, and the floor only ever turns a still-
`dark` row into `shadowed` after matching, so the frontend re-derives it exactly rather than
multiplying this grid by a fifth axis. It does so by first putting every `shadowed` row back
to `dark`, which is why the floor a run happens to be baked at does not matter.

Reception *itself* is baked, because it moves with the max gap — which is why
`Viewer.declarations_key` folds the reception model into the gap's equivalence class. Two gaps
that declare the same vessels can still disagree about how much of the archive's time counts
as covered, and sharing a run between them would serve one gap's answer to the other's
question. The models are written to `reception/`, one per gap rather than one per run, because
they are identical across every tolerance and threshold and were a quarter of each run file.
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
    reception_by_gap = _reception(viewer, default, grid["max_gap_minutes"], out_dir / "reception")

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
            payload.pop("reception")  # written once per gap by `_reception`, not per run
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
            "reception_floor": default.reception_floor,
        },
        "tile_px": default.tile_px,
        "overlap_px": default.overlap_px,
        "distinct_runs": len(ids),
        "runs": runs,
        # One entry per `grid["max_gap_minutes"]`, positionally: the id of the
        # `reception/<id>.json` holding that gap's model, which the frontend fetches once and
        # reuses across every tolerance, threshold and azimuth setting. A list rather than a
        # map keyed by the gap, because 10.0 is a JSON number that reaches JavaScript as 10
        # and would never find a key Python wrote as "10.0".
        "reception": reception_by_gap,
    }


def _reception(
    viewer: Viewer, default: RunRequest, gaps: list[float], out_dir: Path
) -> list[Any]:
    """Write one reception model per max gap; return their ids, positionally by gap.

    Keyed by the gap itself rather than by the gap's *run* equivalence class, which is a
    distinction with a difference: past a certain width every observed interval is covered, so
    two gaps can classify every detection identically — and share a run — while still being
    two different gaps. The model says which one it is, and a viewer that reported the other
    would be quietly wrong about the search it is describing. Identical models still collapse
    onto one file, so nothing is written twice.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ids: dict[str, int] = {}
    index: list[Any] = []
    for gap in gaps:
        model = viewer.run(replace(default, max_gap_minutes=gap, register_xy=[]))["reception"]
        if model is None:
            index.append(None)
            continue
        text = json.dumps(model, separators=(",", ":"))
        digest = hashlib.sha1(text.encode()).hexdigest()
        if digest not in ids:
            ids[digest] = len(ids)
            (out_dir / f"{ids[digest]}.json").write_text(text)
        index.append(ids[digest])
    return index


def _with(values: list[float], value: float) -> list[float]:
    """The grid always contains the configured value, so the default position is exact."""
    return sorted({*values, float(value)})


def _representatives(values: list[float], key) -> dict[float, float]:
    first: dict[str, float] = {}
    return {value: first.setdefault(key(value), value) for value in values}
