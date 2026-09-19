"""Build the detector a run configuration names — the one place that knows all three."""

from __future__ import annotations

from pathlib import Path

from darkvessel.detect.detector import Detector

KINDS = ("stub", "cfar", "cnn")


def make_detector(
    kind: str, threshold: float | None = None, weights: Path | None = None
) -> Detector:
    """`stub` for the synthetic fixture, `cfar` for the classical baseline, `cnn` for the
    trained model (needs `weights` and the `detector` extra)."""
    if kind == "stub":
        from darkvessel.detect.stub import BrightPixelDetector

        return BrightPixelDetector(threshold=0.5 if threshold is None else threshold)
    if kind == "cfar":
        from darkvessel.detect.cfar import CFARDetector

        return CFARDetector() if threshold is None else CFARDetector(threshold=threshold)
    if kind == "cnn":
        if weights is None:
            raise ValueError("the cnn detector needs `detector_weights`")
        try:
            from darkvessel.detect.cnn import CNNDetector
        except ImportError as error:  # torch is an optional extra
            message = 'the cnn detector needs torch: pip install -e ".[detector]"'
            raise ImportError(message) from error
        return CNNDetector(weights, threshold=threshold)
    raise ValueError(f"unknown detector {kind!r}; expected one of {', '.join(KINDS)}")
