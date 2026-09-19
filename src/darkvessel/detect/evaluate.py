"""Scoring a point detector against boxed ground truth: precision, recall, F1, AP.

The `Detector` protocol returns ship *centres*, not boxes, so a prediction is correct when it
lands inside a ground-truth box (grown by a small margin: many LS-SSDD ships are under ten
pixels, and a centre one pixel off the box edge is still that ship). Predictions and ships are
paired one-to-one by optimal assignment — the same reasoning as `fusion.match`: two
predictions on one ship are one hit and one false alarm, never two hits.

Thresholds are chosen on the validation scenes and only then applied to test, so a reported
test number is never tuned on the data it describes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

MARGIN_PX = 3.0
_INFEASIBLE = 1.0e9


@dataclass(frozen=True)
class Scores:
    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        found = self.true_positives + self.false_positives
        return self.true_positives / found if found else 1.0

    @property
    def recall(self) -> float:
        ships = self.true_positives + self.false_negatives
        return self.true_positives / ships if ships else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "threshold": round(float(self.threshold), 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


def match_image(points: np.ndarray, boxes: np.ndarray, margin: float = MARGIN_PX) -> np.ndarray:
    """For each scored point (row, col, score), whether it is paired with a ship.

    Pairing maximises the number of hits first, then prefers higher-scoring points, so the
    result does not depend on the order the detector listed its points in.
    """
    hit = np.zeros(len(points), dtype=bool)
    if len(points) == 0 or len(boxes) == 0:
        return hit
    rows, cols = points[:, 0][:, None], points[:, 1][:, None]
    inside = (
        (cols >= boxes[:, 0] - margin)
        & (cols <= boxes[:, 2] + margin)
        & (rows >= boxes[:, 1] - margin)
        & (rows <= boxes[:, 3] + margin)
    )
    cost = np.where(inside, -points[:, 2][:, None], _INFEASIBLE)
    pred_idx, box_idx = linear_sum_assignment(cost)
    feasible = inside[pred_idx, box_idx]
    hit[pred_idx[feasible]] = True
    return hit


def pooled(results: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, int]:
    """All images' point scores, whether each was a hit, and the total ship count.

    Matching is done once, at the lowest score the detector produced: raising the threshold
    only removes points, and a point that was a hit stays one — assignment prefers higher
    scores, so the pairing among the points above any threshold is the same.
    """
    scores, hits, ships = [], [], 0
    for points, boxes in results:
        ships += len(boxes)
        if len(points):
            scores.append(points[:, 2])
            hits.append(match_image(points, boxes))
    if not scores:
        return np.zeros(0), np.zeros(0, dtype=bool), ships
    return np.concatenate(scores), np.concatenate(hits), ships


def at_threshold(scores: np.ndarray, hits: np.ndarray, ships: int, threshold: float) -> Scores:
    kept = scores >= threshold
    tp = int(hits[kept].sum())
    return Scores(threshold, tp, int(kept.sum()) - tp, ships - tp)


def best_f1(scores: np.ndarray, hits: np.ndarray, ships: int) -> Scores:
    """The threshold that maximises F1 — used on validation only."""
    if len(scores) == 0:
        return Scores(float("inf"), 0, 0, ships)
    order = np.argsort(-scores)
    tp = np.cumsum(hits[order])
    fp = np.cumsum(~hits[order])
    f1 = 2 * tp / (tp + fp + ships)
    i = int(np.argmax(f1))
    return Scores(float(scores[order][i]), int(tp[i]), int(fp[i]), ships - int(tp[i]))


def average_precision(scores: np.ndarray, hits: np.ndarray, ships: int) -> float:
    """Area under the precision-recall curve (all-point interpolation, as in VOC 2010+)."""
    if len(scores) == 0 or ships == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = np.cumsum(hits[order])
    fp = np.cumsum(~hits[order])
    recall = np.concatenate([[0.0], tp / ships])
    precision = np.concatenate([[1.0], tp / (tp + fp)])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall) * precision[1:]))
