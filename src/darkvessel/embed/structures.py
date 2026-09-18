"""Which detections are fixed structures, found by recurrence rather than by appearance.

An offshore wind turbine is a bright point scatterer on water, which is what a ship is, and
a ship detector returns turbines happily. Every one that reaches the fusion unexplained
becomes a "dark vessel" — a claim someone may be sent out on.

The signal used is **recurrence**: a position carrying a detection acquisition after
acquisition is not a ship. A vessel under way is somewhere else a week later; a mast is not.
It asks nothing of the pixels and nothing of a label — only of provenance the archive
already keeps.

**Why not greedy.** A greedy seed-and-claim
grouping ("every unclaimed crop within tolerance joins the current seed") is *order
dependent*: for three collinear points A-B=90m, B-C=90m, A-C=180m at a 100m tolerance,
processing in order A,B,C yields two groups (best case 2 acquisitions), while B,A,C yields
one group of three. Same points, same tolerance, different answer — so whether a structure
clears an exclusion floor could depend on the order rows happened to land in the archive.
Union-find over the mutual-tolerance graph removes the dependency: the grouping is the
graph's connected components, which are a property of the points alone.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# How close two detections must be to count as the same standing object. Turbines stand
# some 600 m apart and a detection wobbles by a pixel or two, so 100 m separates neighbouring
# masts while holding successive sightings of one mast together.
#
# Deliberately NOT the fusion's 200 m match tolerance, which answers a different question:
# that one asks "could this declaration explain this detection", this one asks "is this the
# same standing object".
SAME_POSITION_M = 100.0


class _Components:
    """Union-find over detection indices."""

    def __init__(self, count: int) -> None:
        self._parent = list(range(count))

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


@dataclass(frozen=True)
class Standing:
    """Every distinct position the archive holds a detection at, and how often it recurred.

    `positions` is one row per position: where it is, how many *distinct acquisitions* saw
    it, and how many crops it accounts for. `of_crop` maps each crop to its position by
    index, so a count over crops and a count over positions can never drift apart.
    """

    positions: pd.DataFrame
    of_crop: np.ndarray

    def acquisitions_of_crop(self) -> np.ndarray:
        return self.positions["acquisitions"].to_numpy()[self.of_crop]

    def seen_in(self, floor: int) -> pd.DataFrame:
        """Positions seen in at least `floor` distinct acquisitions, most persistent first."""
        kept = self.positions[self.positions["acquisitions"] >= floor]
        return kept.sort_values("acquisitions", ascending=False).reset_index(drop=True)


def standing(provenance: pd.DataFrame, tolerance_m: float = SAME_POSITION_M) -> Standing:
    """Group the archive's detections into standing positions by connected components.

    `provenance` carries `x`, `y` (in a projected CRS) and `scene`. Two detections are
    linked when they are within `tolerance_m`; a standing position is a connected component
    of that graph. Order-independent by construction — see the module docstring.
    """
    x = provenance["x"].to_numpy(dtype=float)
    y = provenance["y"].to_numpy(dtype=float)
    scenes = provenance["scene"].to_numpy()
    count = len(x)

    if count == 0:
        return Standing(
            positions=pd.DataFrame(columns=["x", "y", "acquisitions", "crops"]).astype(
                {"x": float, "y": float, "acquisitions": int, "crops": int}
            ),
            of_crop=np.zeros(0, dtype=np.int64),
        )

    components = _Components(count)
    for index in range(count):
        near = np.nonzero(np.hypot(x - x[index], y - y[index]) <= tolerance_m)[0]
        for neighbour in near:
            components.union(index, int(neighbour))

    # Relabel roots to dense 0..n-1 ids, in order of first appearance, so the output is
    # deterministic and independent of which index happened to become a component's root.
    of_crop = np.full(count, -1, dtype=np.int64)
    order: dict[int, int] = {}
    for index in range(count):
        root = components.find(index)
        if root not in order:
            order[root] = len(order)
        of_crop[index] = order[root]

    rows = []
    for position_id in range(len(order)):
        members = of_crop == position_id
        rows.append(
            {
                # The centre of the sightings, not the first of them: a mast detected forty
                # times is located better by forty detections than by any one of them.
                "x": float(x[members].mean()),
                "y": float(y[members].mean()),
                "acquisitions": len(set(scenes[members].tolist())),
                "crops": int(members.sum()),
            }
        )

    return Standing(
        positions=pd.DataFrame(rows, columns=["x", "y", "acquisitions", "crops"]).astype(
            {"x": float, "y": float, "acquisitions": int, "crops": int}
        ),
        of_crop=of_crop,
    )


@dataclass(frozen=True)
class Verified:
    """A register of fixed positions, checked against coordinates somebody else published.

    Both directions, because either alone can be made to look good. `found` says the register
    did not miss the farm; `unpublished` says the register is not full of things nobody has
    recorded — and where it is, that is a finding rather than an error, since the sea holds
    fixed structures no turbine list mentions.
    """

    known: int
    registered: int
    found: int
    unpublished: int
    median_m: float
    tolerance_m: float

    def line(self) -> str:
        return (
            f"{self.found} of {self.known} published positions carry a registered structure, "
            f"{self.registered - self.unpublished} of {self.registered} registered structures "
            f"stand at a published position, {self.median_m:.1f} m apart at the median, "
            f"within {self.tolerance_m:g} m"
        )


def verify(registered: pd.DataFrame, known: pd.DataFrame, tolerance_m: float) -> Verified:
    """Check a register of fixed positions against published coordinates for the same water.

    Both frames carry `x`/`y` in the *same* CRS; nothing is reprojected here, because a
    silent reprojection is how two sets of coordinates come to be compared in two metres.
    """
    if known.empty:
        raise ValueError(
            "a verification against no published positions is not a verification; this is a "
            "statement about the reference rather than a score the register can be given"
        )
    if registered.empty:
        # Finding none of the farm is the failure this check exists to catch, so it comes back
        # as a number. `median_m` is infinite, not zero — zero would read as perfect agreement.
        return Verified(
            known=len(known),
            registered=0,
            found=0,
            unpublished=0,
            median_m=float("inf"),
            tolerance_m=float(tolerance_m),
        )

    apart = np.hypot(
        registered["x"].to_numpy(dtype=float)[:, None] - known["x"].to_numpy(dtype=float)[None, :],
        registered["y"].to_numpy(dtype=float)[:, None] - known["y"].to_numpy(dtype=float)[None, :],
    )
    return Verified(
        known=len(known),
        registered=len(registered),
        found=int((apart.min(axis=0) <= tolerance_m).sum()),
        unpublished=int((apart.min(axis=1) > tolerance_m).sum()),
        median_m=float(np.median(apart.min(axis=1))),
        tolerance_m=float(tolerance_m),
    )
