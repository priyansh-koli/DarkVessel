"""Benchmark a detector on LS-SSDD: pick its operating point on validation, report on test.

The protocol, so a number here means one thing:

1. Score every validation sub-image (held-out training scenes) and choose the threshold that
   maximises F1 there.
2. Apply that threshold, unchanged, to the official test scenes, and report precision, recall
   and F1 overall and split into inshore and offshore, plus threshold-free AP.

The inshore split matters most for this project: harbour walls, cranes and moored hulls are
where a detector's false alarms concentrate, and where a dark-vessel claim is least safe.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from darkvessel.detect import evaluate
from darkvessel.detect import lsssdd as L


def benchmark(scorer, root: Path, name: str, log=print) -> dict:
    """`scorer(image) -> (n, 3)` scored points. Returns validation and test results."""
    audit = L.Audit()
    val = L.load(root, L.split_names(root, "val"), audit)
    test = L.load(root, L.split_names(root, "test"), audit)
    inshore = set(L.split_names(root, "test_inshore"))

    started = time.time()
    val_points = [(scorer(L.read_image(s.image_path)), s.boxes) for s in val]
    chosen = evaluate.best_f1(*evaluate.pooled(val_points))
    log(f"{name}: validation best F1 {chosen.f1:.3f} at threshold {chosen.threshold:.3f}")

    test_points = [(scorer(L.read_image(s.image_path)), s.boxes) for s in test]
    seconds = time.time() - started
    per_image_ms = 1000 * seconds / max(len(val) + len(test), 1)

    report = {
        "detector": name,
        "threshold": round(chosen.threshold, 4),
        "validation": {**chosen.as_dict(), "ap": _ap(val_points)},
        "test": _report(test_points, chosen.threshold),
        "test_inshore": _report(
            [r for r, s in zip(test_points, test) if s.name in inshore], chosen.threshold
        ),
        "test_offshore": _report(
            [r for r, s in zip(test_points, test) if s.name not in inshore], chosen.threshold
        ),
        "ms_per_800px_image": round(per_image_ms, 1),
        "data_audit": dict(audit.counts),
    }
    for split in ("test", "test_inshore", "test_offshore"):
        r = report[split]
        log(
            f"  {split:14s} P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1']:.3f}  "
            f"AP {r['ap']:.3f}  ({r['true_positives']} hit, {r['false_positives']} false, "
            f"{r['false_negatives']} missed)"
        )
    return report


def _report(points: list[tuple[np.ndarray, np.ndarray]], threshold: float) -> dict:
    scores, hits, ships = evaluate.pooled(points)
    return {
        **evaluate.at_threshold(scores, hits, ships, threshold).as_dict(),
        "ap": _ap(points),
    }


def _ap(points) -> float:
    return round(evaluate.average_precision(*evaluate.pooled(points)), 4)
