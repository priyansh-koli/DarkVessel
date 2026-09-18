"""SAR pixels to PNG, for the viewer and for a static build.

Rendering is kept at native resolution and upscaled by the browser with nearest-neighbour:
a 5 m pixel is the unit the whole chain reasons in, so blurring it away in a resampled
image would hide exactly the thing an analyst is looking at.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pandas as pd
from PIL import Image

from darkvessel.embed.crops import crops_for


def stretch(image: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5) -> np.ndarray:
    """Map SAR amplitudes onto 0..255 by percentile, falling back to the full range.

    A percentile stretch is what makes real SAR readable — a handful of very bright returns
    otherwise push every vessel down into the noise floor. On a synthetic scene of mostly
    zeros both percentiles collapse to the same value, so min/max is used instead.
    """
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    low, high = float(np.percentile(finite, low_pct)), float(np.percentile(finite, high_pct))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = (np.clip(image, low, high) - low) / (high - low)
    return np.round(scaled * 255).astype(np.uint8)


def scene_png(image: np.ndarray) -> bytes:
    """The whole scene as an 8-bit greyscale PNG, at native pixel resolution."""
    return _png_bytes(Image.fromarray(stretch(image)))


def crop_data_uris(
    image: np.ndarray, found: pd.DataFrame, crop_px: int = 32, margin_px: int = 8
) -> list[str]:
    """One `data:` PNG per detection, in row order, ready to drop into an `<img src>`.

    Inlined rather than served as files so a live re-run and a pre-rendered static build hand
    the frontend exactly the same payload — see `darkvessel render`.
    """
    if found.empty:
        return []

    crops = crops_for(image, found, crop_px=crop_px, margin_px=margin_px)
    # Stretched over the whole scene, not per crop: a crop normalised to itself would make
    # empty water look as bright as a hull.
    low, high = _limits(image)
    uris = []
    for crop in crops:
        if high <= low:
            scaled = np.zeros(crop.shape, dtype=np.uint8)
        else:
            normalised = (np.clip(crop, low, high) - low) / (high - low)
            scaled = np.round(normalised * 255).astype(np.uint8)
        uris.append(_data_uri(_png_bytes(Image.fromarray(scaled))))
    return uris


def _limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 0.0
    low, high = float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.5))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    return low, high


def _png_bytes(pil_image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
