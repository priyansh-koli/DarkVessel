"""AIS cleaning: every rule applied to raw archive rows before they are trusted, counted.

Auditable because a study that reports a vessel dark had better say what its AIS search
actually searched — a silent cleaning rule that happens to drop the one report that would
have explained a detection is indistinguishable, from the outside, from a genuinely dark
vessel.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd

from darkvessel.data.study_area import StudyArea

# ~50 knots; faster than any merchant or fishing vessel. Public because `fusion.reception`
# applies the same cap to the speed a pair of reports *implies*, and the two must not drift.
MAX_PLAUSIBLE_SPEED_MS = 25.7


@dataclass(frozen=True)
class CleaningReport:
    """How many rows each rule removed, in the order the rules ran."""

    starting_rows: int
    removed: dict[str, int]

    def line(self) -> str:
        removed_total = sum(self.removed.values())
        parts = ", ".join(f"{name}: {count}" for name, count in self.removed.items())
        kept = self.starting_rows - removed_total
        return f"{self.starting_rows} rows in, {removed_total} removed ({parts}), {kept} kept"


def clean(
    raw: gpd.GeoDataFrame, area: StudyArea | None = None
) -> tuple[gpd.GeoDataFrame, CleaningReport]:
    """Apply every cleaning rule in sequence, counting what each one removes."""
    removed: dict[str, int] = {}
    working = raw

    before = len(working)
    working = working.dropna(subset=["mmsi", "timestamp"])
    working = working[working.geometry.notna()]
    removed["missing_identity_time_or_position"] = before - len(working)

    before = len(working)
    working = working.drop_duplicates(subset=["mmsi", "timestamp"])
    removed["duplicate_mmsi_timestamp"] = before - len(working)

    if "speed_ms" in working.columns:
        before = len(working)
        working = working[
            working["speed_ms"].isna() | (working["speed_ms"] <= MAX_PLAUSIBLE_SPEED_MS)
        ]
        removed["implausible_speed"] = before - len(working)

    if area is not None:
        before = len(working)
        working = working[
            working.geometry.x.between(area.minx, area.maxx)
            & working.geometry.y.between(area.miny, area.maxy)
        ]
        removed["outside_study_area"] = before - len(working)

    return working.reset_index(drop=True), CleaningReport(starting_rows=len(raw), removed=removed)
