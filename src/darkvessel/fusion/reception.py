"""AIS reception: whether the archive could have placed a vessel here at all.

A dark detection says the AIS search ran and found nothing. That claim is only as good as the
search's reach, and an archive's reach is not uniform. Terrestrial AIS receivers hear a
transponder for some tens of kilometres and no further, satellite passes are intermittent, and
published archives are often decimated before release. Reception therefore falls away with
distance from shore — along the very gradient a dark-vessel study cares about — so a map of
unexplained detections drawn without it is partly a map of where the receivers are.

**What is estimated.** Not the radio link, but the thing the pipeline actually depends on:
*had a transmitting vessel been here at the acquisition instant, would this archive have
placed it?* `fusion.interpolate` places a vessel when a report falls within `max_gap` of the
instant, so that is the question asked, with the same `max_gap`. Decimation, receiver gaps and
satellite revisit all fold into the answer correctly, because all three change the archive
rather than the radio.

**How.** Every vessel the archive already holds is its own probe. For one vessel's
consecutive reports `t_i` and `t_i+1` a gap of `g` seconds, an instant drawn uniformly from
that gap is within `w = max_gap` of a report over `min(g, 2w)` of it — `w` at each end, capped
at the whole gap. Summing over the gaps observed in a neighbourhood gives

    reception = sum(min(g, 2w)) / sum(g)

which is 1 where reports come faster than the gap allows for, and falls towards zero where
they do not. Each gap is attributed to its own midpoint, because the uncovered part of a long
gap is the middle of it. Gaps whose straight-line speed is implausible are dropped and
counted: the vessel did not travel that line, so its midpoint is not where it was.

Evidence is pooled over a cell and the eight cells touching it, so an estimate is not an
artefact of which side of a grid line a detection fell. A cell backed by fewer than
`min_intervals` gaps returns no estimate at all rather than a confident one.

**What a low estimate does.** It moves a `dark` detection to `shadowed`: the search could not
have seen this vessel here, so its silence is not evidence. What a *missing* estimate does is
nothing — a detection whose neighbourhood carries too little evidence stays `dark`, with the
reason on the row. Reclassifying on an absence of evidence would be the same mistake as
reporting a dark vessel on one, pointed the other way.

**The caveat that matters.** A vessel switching its transponder off looks exactly like a
receiver that cannot hear it. Where many vessels go dark, reception is underestimated and
genuine findings are downgraded — which is the wrong direction for enforcement and the right
one for a published claim. The estimate is on the row either way, so a reader can take it
back off.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from darkvessel.data.ais import MAX_PLAUSIBLE_SPEED_MS
from darkvessel.fusion.match import DARK

SHADOWED = "shadowed"

ESTIMATED = "estimated"
INSUFFICIENT = "insufficient_evidence"

# The side of one grid cell, in metres. Evidence is pooled over a cell and its eight
# neighbours, so this is a third of the scale reception is smoothed over: 1 km cells smooth
# over 3 km, which is fine against the tens of kilometres a receiver's range varies on.
RECEPTION_CELL_M = 1000.0

# How many observed gaps a neighbourhood needs before its reception is reported at all. One
# vessel passing once says almost nothing about whether a second would have been heard.
MIN_INTERVALS = 5

# The default floor is *no* floor: reception is reported on every row, and where the bar for
# a dark claim sits is a decision for a run configuration to state, not for this module to
# make silently. See `configs/pipeline.yaml`.
RECEPTION_FLOOR = 0.0

# Declared once so a column's presence and dtype never depend on whether a coverage model was
# supplied, exactly as `context.schema` does for the contextual layers.
RECEPTION_COLUMNS = {
    "reception_p": "float64",
    "reception_basis": "string",
    "reception_intervals": "float64",
}

# Cell indices are packed into one int64 so a lookup is a binary search rather than a hash per
# row. Safe while |index| stays under the stride's half-width, which every projected CRS does.
_STRIDE = 1 << 32
_MAX_INDEX = _STRIDE // 2

_NEIGHBOURHOOD = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]


@dataclass(frozen=True)
class CoverageReport:
    """What the estimate was built from, and what it had to throw away."""

    intervals: int
    used: int
    dropped: dict
    cells: int
    max_gap_s: float

    def line(self) -> str:
        dropped_total = sum(self.dropped.values())
        parts = ", ".join(f"{name}: {count}" for name, count in self.dropped.items())
        return (
            f"{self.intervals} report intervals in, {dropped_total} dropped ({parts}), "
            f"{self.used} attributed to {self.cells} cells "
            f"at a max gap of {self.max_gap_s / 60:g} min"
        )


@dataclass(frozen=True)
class Coverage:
    """A reception estimate over the study area, ready to qualify dark detections.

    Built once per archive and `max_gap`. Held sorted by packed cell index so `mark` is one
    binary search over every detection at once rather than a dictionary lookup per row.
    """

    keys: np.ndarray  # packed cell index, sorted
    covered_s: np.ndarray
    total_s: np.ndarray
    intervals: np.ndarray
    cell_m: float
    max_gap: timedelta
    min_intervals: int
    report: CoverageReport

    @classmethod
    def from_archive(
        cls,
        ais: gpd.GeoDataFrame,
        max_gap: timedelta,
        cell_m: float = RECEPTION_CELL_M,
        min_intervals: int = MIN_INTERVALS,
    ) -> Coverage:
        """Estimate reception from the archive's own report intervals. See the module docstring.

        `ais` is the *cleaned* archive, in a projected CRS — the same rows the match searched,
        so the reception reported beside a dark claim describes the search that made it.
        """
        window_s = float(max_gap.total_seconds())
        gaps, midpoints, dropped = _intervals(ais)
        report_kwargs = dict(
            intervals=int(len(gaps) + sum(dropped.values())),
            used=int(len(gaps)),
            dropped=dropped,
            max_gap_s=window_s,
        )

        if len(gaps) == 0:
            return cls(
                keys=np.zeros(0, dtype=np.int64),
                covered_s=np.zeros(0),
                total_s=np.zeros(0),
                intervals=np.zeros(0),
                cell_m=float(cell_m),
                max_gap=max_gap,
                min_intervals=int(min_intervals),
                report=CoverageReport(cells=0, **report_kwargs),
            )

        covered = np.minimum(gaps, 2.0 * window_s)
        ix, iy = _cell_of(midpoints[:, 0], midpoints[:, 1], cell_m)
        ix, iy, covered_s, total_s, counts = _pooled(ix, iy, covered, gaps)

        keys = _pack(ix, iy)
        order = np.argsort(keys, kind="stable")
        return cls(
            keys=keys[order],
            covered_s=covered_s[order],
            total_s=total_s[order],
            intervals=counts[order],
            cell_m=float(cell_m),
            max_gap=max_gap,
            min_intervals=int(min_intervals),
            report=CoverageReport(cells=int(len(keys)), **report_kwargs),
        )

    def at(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reception and the number of gaps behind it, at each of these ground coordinates.

        Reception is `nan` where the neighbourhood holds fewer than `min_intervals` gaps: an
        estimate nothing supports is worse than none, because it reads like one that does.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        probability = np.full(len(x), np.nan)
        intervals = np.zeros(len(x), dtype=float)
        if len(self.keys) == 0 or len(x) == 0:
            return probability, intervals

        ix, iy = _cell_of(x, y, self.cell_m)
        position = np.searchsorted(self.keys, _pack(ix, iy))
        position = np.clip(position, 0, len(self.keys) - 1)
        found = self.keys[position] == _pack(ix, iy)

        intervals[found] = self.intervals[position[found]]
        enough = found & (intervals >= self.min_intervals)
        total = self.total_s[position[enough]]
        probability[enough] = np.where(total > 0, self.covered_s[position[enough]] / total, np.nan)
        return probability, intervals

    def mark(
        self, detections: gpd.GeoDataFrame, floor: float = RECEPTION_FLOOR
    ) -> gpd.GeoDataFrame:
        """Attach reception to every detection, and move the unsupportable dark ones aside.

        Reception goes on *every* row, matched included: it describes the place, and a match
        made where the archive barely reaches is worth knowing about too. Only `DARK` rows
        change status, and only on a measured estimate below `floor` — never on a missing one.
        """
        marked = detections.copy()
        if marked.empty:
            return without_coverage(marked)

        xy = shapely.get_coordinates(marked.geometry.values)
        probability, intervals = self.at(xy[:, 0], xy[:, 1])

        marked["reception_p"] = probability
        marked["reception_basis"] = pd.Series(
            np.where(np.isfinite(probability), ESTIMATED, INSUFFICIENT),
            index=marked.index,
            dtype="string",
        )
        marked["reception_intervals"] = intervals

        shadowed = (
            (marked["status"] == DARK).to_numpy()
            & np.isfinite(probability)
            & (probability < float(floor))
        )
        marked.loc[shadowed, "status"] = SHADOWED
        return marked

    def cells(self) -> pd.DataFrame:
        """One row per cell — bounds, reception, evidence — for drawing or for a join."""
        ix, iy = _unpack(self.keys)
        total = self.total_s
        with np.errstate(invalid="ignore", divide="ignore"):
            probability = np.where(total > 0, self.covered_s / total, np.nan)
        probability = np.where(self.intervals >= self.min_intervals, probability, np.nan)
        return pd.DataFrame(
            {
                "x0": ix * self.cell_m,
                "y0": iy * self.cell_m,
                "x1": (ix + 1) * self.cell_m,
                "y1": (iy + 1) * self.cell_m,
                "reception_p": probability,
                "intervals": self.intervals.astype(float),
                "observed_s": total,
            }
        )


def without_coverage(detections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fill the reception schema with absent values: the no-op used when nothing estimated it.

    The explicit counterpart of `fusion.register.without_a_register` — a column's presence must
    never depend on whether an optional stage ran.
    """
    filled = detections.copy()
    for column, dtype in RECEPTION_COLUMNS.items():
        empty = pd.NA if dtype == "string" else np.nan
        filled[column] = pd.Series([empty] * len(filled), index=filled.index, dtype=dtype)
    return filled


def coverage_for(
    ais: gpd.GeoDataFrame | None,
    max_gap: timedelta,
    cell_m: float = RECEPTION_CELL_M,
    min_intervals: int = MIN_INTERVALS,
) -> Coverage | None:
    """A coverage model, or `None` where there is no archive to estimate one from."""
    if ais is None or ais.empty:
        return None
    return Coverage.from_archive(ais, max_gap, cell_m=cell_m, min_intervals=min_intervals)


def _intervals(ais: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Every usable gap between consecutive reports of one vessel, and where its middle was."""
    dropped = {"not_a_pair": 0, "no_elapsed_time": 0, "implausible_speed": 0}
    if ais is None or len(ais) < 2:
        return np.zeros(0), np.zeros((0, 2)), dropped

    xy = shapely.get_coordinates(ais.geometry.values)
    # `as_unit("ns")` before `asi8`, never after: pandas keeps a datetime column in whatever
    # resolution it arrived at, and a GeoPackage hands back milliseconds. Reading the raw
    # integers without pinning the unit makes every gap a thousand times too short, which
    # reads as a vessel travelling at a thousand times its speed and throws the lot away.
    frame = pd.DataFrame(
        {
            "mmsi": ais["mmsi"].to_numpy(),
            "t": pd.DatetimeIndex(ais["timestamp"]).as_unit("ns").asi8,
        }
    )
    frame["x"], frame["y"] = xy[:, 0], xy[:, 1]
    frame = frame.sort_values(["mmsi", "t"], kind="stable")

    mmsi = frame["mmsi"].to_numpy()
    seconds = frame["t"].to_numpy(dtype=np.float64) / 1e9
    x, y = frame["x"].to_numpy(), frame["y"].to_numpy()

    same_vessel = mmsi[1:] == mmsi[:-1]
    gaps = seconds[1:] - seconds[:-1]
    dx, dy = x[1:] - x[:-1], y[1:] - y[:-1]

    dropped["not_a_pair"] = int((~same_vessel).sum())
    elapsed = same_vessel & (gaps > 0)
    dropped["no_elapsed_time"] = int((same_vessel & (gaps <= 0)).sum())

    # The same cap `data.ais.clean` applies to a reported speed, applied here to an implied
    # one: above it the vessel cannot have travelled this straight line, so its midpoint is
    # not a place the gap can be attributed to.
    with np.errstate(invalid="ignore", divide="ignore"):
        implied = np.where(elapsed, np.hypot(dx, dy) / np.where(gaps > 0, gaps, 1.0), 0.0)
    keep = elapsed & (implied <= MAX_PLAUSIBLE_SPEED_MS)
    dropped["implausible_speed"] = int((elapsed & (implied > MAX_PLAUSIBLE_SPEED_MS)).sum())

    midpoints = np.column_stack([(x[:-1] + x[1:])[keep] / 2.0, (y[:-1] + y[1:])[keep] / 2.0])
    return gaps[keep], midpoints, dropped


def _cell_of(x: np.ndarray, y: np.ndarray, cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    ix = np.floor(np.asarray(x, dtype=float) / cell_m).astype(np.int64)
    iy = np.floor(np.asarray(y, dtype=float) / cell_m).astype(np.int64)
    return ix, iy


def _pooled(
    ix: np.ndarray, iy: np.ndarray, covered: np.ndarray, gaps: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Total each cell, then lend every cell's totals to the eight cells touching it.

    Two passes rather than a neighbour search: the first collapses gaps onto cells, the second
    spreads those (far fewer) cell totals over the neighbourhood. Both are one `np.unique`.
    """
    ix, iy, covered_s, total_s, counts = _totals(ix, iy, covered, gaps, np.ones(len(gaps)))
    spread_ix = np.concatenate([ix + dx for dx, _ in _NEIGHBOURHOOD])
    spread_iy = np.concatenate([iy + dy for _, dy in _NEIGHBOURHOOD])
    repeats = len(_NEIGHBOURHOOD)
    return _totals(
        spread_ix,
        spread_iy,
        np.tile(covered_s, repeats),
        np.tile(total_s, repeats),
        np.tile(counts, repeats),
    )


def _totals(
    ix: np.ndarray, iy: np.ndarray, covered: np.ndarray, gaps: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, ...]:
    keys, inverse = np.unique(_pack(ix, iy), return_inverse=True)
    size = len(keys)
    cell_ix, cell_iy = _unpack(keys)
    return (
        cell_ix,
        cell_iy,
        np.bincount(inverse, weights=covered, minlength=size),
        np.bincount(inverse, weights=gaps, minlength=size),
        np.bincount(inverse, weights=counts, minlength=size),
    )


def _pack(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """One int64 per cell. `iy` is biased into the non-negative half so a southern cell packs
    and unpacks exactly as a northern one does."""
    if len(ix) and (np.abs(ix).max() >= _MAX_INDEX or np.abs(iy).max() >= _MAX_INDEX):
        raise ValueError(
            "cell indices overflow the packed key; the cell size is too small for this CRS"
        )
    return ix.astype(np.int64) * _STRIDE + (iy.astype(np.int64) + _MAX_INDEX)


def _unpack(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ix = np.floor_divide(keys, _STRIDE)
    return ix, keys - ix * _STRIDE - _MAX_INDEX
