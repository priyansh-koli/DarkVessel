"""The structure register: detections explained by recurrence rather than by AIS.

Built from `embed.structures.standing` — see there for what counts as a "position" and why
recurrence, not appearance, is the signal. This module only turns that register into a column
on classified detections. Structure exclusion happens strictly after matching: a detection
AIS explains is a match whatever else stands at that coordinate, never the reverse.
"""

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd

from darkvessel.embed.structures import SAME_POSITION_M, Standing
from darkvessel.fusion.match import DARK

STRUCTURE = "structure"


@dataclass(frozen=True)
class Register:
    """Fixed positions recurrence has established, ready to mark unexplained detections."""

    positions: pd.DataFrame  # x, y, acquisitions, crops
    tolerance_m: float = SAME_POSITION_M

    @classmethod
    def from_standing(
        cls, standing: Standing, floor: int = 2, tolerance_m: float = SAME_POSITION_M
    ) -> "Register":
        return cls(positions=standing.seen_in(floor), tolerance_m=tolerance_m)

    def mark(self, detections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Reclassify dark detections that stand at a registered position as `STRUCTURE`.

        Only `DARK` rows are touched — a match AIS already explains is never overridden.
        """
        marked = detections.copy()
        if self.positions.empty:
            return marked

        px = self.positions["x"].to_numpy(dtype=float)
        py = self.positions["y"].to_numpy(dtype=float)
        for idx in marked.index[marked["status"] == DARK]:
            point = marked.geometry.loc[idx]
            distance = np.hypot(px - point.x, py - point.y)
            if distance.min() <= self.tolerance_m:
                marked.loc[idx, "status"] = STRUCTURE
        return marked


def without_a_register(detections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Pass detections through unchanged: the explicit no-op used when no register is supplied."""
    return detections.copy()
