"""The embedder contract: crops in, one embedding vector per crop out.

Optional everywhere it is used — see `pipeline.run`. Contrastive embeddings are listed in the
README's Status as not yet built; this is the seam a trained one plugs into.
"""

from typing import Protocol

import geopandas as gpd
import numpy as np


class Embedder(Protocol):
    crop_px: int
    margin_px: int

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        """Return one embedding row per crop, in the same order."""
        ...


def attach(detections: gpd.GeoDataFrame, embeddings: np.ndarray) -> gpd.GeoDataFrame:
    """Attach one embedding per detection, in row order."""
    attached = detections.copy()
    attached["embedding"] = list(embeddings)
    return attached
