"""AIS interpolation to the acquisition instant.

A vessel at 12 knots covers ~370 m a minute — more than any sane match tolerance — so every
declared position is placed at the acquisition timestamp before anything is compared. Nothing
is ever extrapolated past the end of a track: where the acquisition falls outside a vessel's
bracketing reports, the nearest single report is used instead, and the row says so in
`position_basis`, so a match built on it can be told apart from one built on a real bracket.

`max_gap` bounds how stale a *single* report may be, and a bracket is not bounded by it at
all: two reports an hour apart still interpolate, and report `position_age_s` of zero, because
the acquisition falls between them. The width of that bracket is therefore recorded as
`position_span_s`, so a position resting on two reports a minute apart can be told from one
resting on two an hour apart. `fusion.reception` asks the same question of the archive as a
whole.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

INTERPOLATED = "interpolated"
NEAREST = "nearest"

_EMPTY_COLUMNS = {
    "mmsi": "string",
    "length_m": "float64",
    "position_basis": "string",
    "position_age_s": "float64",
    "position_span_s": "float64",
    "velocity_east_ms": "float64",
    "velocity_north_ms": "float64",
}


def positions_at(
    ais: gpd.GeoDataFrame, acquired_at: datetime, max_gap: timedelta
) -> gpd.GeoDataFrame:
    """One declared position per MMSI at `acquired_at`, or none where nothing is close enough.

    `ais` carries `mmsi`, `timestamp`, `length_m` and point geometry, one row per report.
    Rows come back in the order each MMSI first appears in `ais`.

    Vectorised over the whole archive rather than looped per vessel: a day of a national
    archive holds thousands of vessels, and a Python loop per vessel dominated the fusion.
    Reports are sorted once by vessel and time, and the two either side of the instant are
    found by counting, per vessel, how many fall at or before it.
    """
    acquired = pd.Timestamp(acquired_at)
    timestamps = pd.DatetimeIndex(ais["timestamp"])
    if len(ais) and (timestamps.tz is None) != (acquired.tz is None):
        # The per-vessel comparisons this replaced raised here too. Comparing a local clock
        # with UTC silently would place every vessel at the wrong instant.
        raise TypeError("cannot compare tz-naive and tz-aware timestamps")

    # A report with no identity or no time places nothing.
    usable = (ais["mmsi"].notna().to_numpy()) & ~np.asarray(timestamps.isna())
    if not usable.any():
        return _empty(ais.crs)

    # Numbered after the filter, so every code in 0..n-1 has at least one report.
    codes = pd.factorize(ais["mmsi"][usable], sort=False)[0]
    times = timestamps[usable].as_unit("ns").asi8
    instant = acquired.as_unit("ns").value
    xy = shapely.get_coordinates(ais.geometry.values[usable])
    lengths = ais["length_m"].to_numpy(dtype=float)[usable]
    identities = ais["mmsi"].to_numpy()[usable]

    # By vessel, then by time; `lexsort` is stable, so equal times keep archive order.
    order = np.lexsort((times, codes))
    codes, times, xy = codes[order], times[order], xy[order]
    lengths, identities = lengths[order], identities[order]

    # `factorize` numbers vessels by first appearance, so `vessels` is in that order too.
    _, starts, counts = np.unique(codes, return_index=True, return_counts=True)
    last = starts + counts - 1
    before = np.bincount(codes, weights=times <= instant).astype(np.int64)

    bracketed = (before > 0) & (before < counts)
    left = (starts + before - 1)[bracketed]
    right = left + 1

    # Outside a bracket the nearest single report stands in: the track's last report if the
    # instant is after it, its first if the instant is before it.
    nearest = np.where(before > 0, last, starts)
    age_s = np.abs(instant - times[nearest]) / 1e9
    placed = bracketed | (age_s <= max_gap.total_seconds())
    if not placed.any():
        return _empty(ais.crs)

    span_s = (times[right] - times[left]) / 1e9
    fraction = (instant - times[left]) / 1e9 / span_s
    step = xy[right] - xy[left]

    position = xy[nearest].copy()
    position[bracketed] = xy[left] + fraction[:, None] * step

    # A lone report's velocity comes from the track's last two reports, whichever side of the
    # instant the track lies. `docs/progress.md` lists that as an open question for a track
    # that starts after the instant; it is kept exactly here, not changed.
    velocity = np.full((len(starts), 2), np.nan)
    prior = np.maximum(last - 1, starts)
    history_s = (times[last] - times[prior]) / 1e9
    has_history = ~bracketed & (counts >= 2) & (history_s > 0)
    velocity[has_history] = (xy[last] - xy[prior])[has_history] / history_s[has_history, None]
    velocity[bracketed] = step / span_s[:, None]

    position_span_s = np.full(len(starts), np.nan)
    # Zero age, but not zero uncertainty: this is how wide the bracket behind it was. A lone
    # report has no bracket, and its `position_age_s` is the whole story.
    position_span_s[bracketed] = span_s

    return gpd.GeoDataFrame(
        {
            "mmsi": pd.array(identities[starts][placed], dtype=ais["mmsi"].dtype),
            "length_m": lengths[last][placed],
            "position_basis": np.where(bracketed, INTERPOLATED, NEAREST)[placed],
            "position_age_s": np.where(bracketed, 0.0, age_s)[placed],
            "position_span_s": position_span_s[placed],
            "velocity_east_ms": velocity[placed, 0],
            "velocity_north_ms": velocity[placed, 1],
        },
        geometry=shapely.points(position[placed]),
        crs=ais.crs,
    )


def _empty(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {name: pd.array([], dtype=dtype) for name, dtype in _EMPTY_COLUMNS.items()},
        geometry=[],
        crs=crs,
    )
