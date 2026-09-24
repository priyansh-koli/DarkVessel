"""Scene loading: pixel-space SAR imagery paired with the geometry to place it on the ground."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from affine import Affine
from shapely import Polygon


@dataclass(frozen=True)
class Scene:
    """One SAR acquisition: pixels, the geometry to place them on the ground, and its identity.

    `id` is the provenance key structures are grouped by — see `data.provenance` and
    `embed.structures.standing`. It must be stable and unique per acquisition.
    """

    id: str
    image: np.ndarray
    transform: Affine
    crs: str
    acquired_at: datetime
    heading_deg: float
    incidence_deg: float

    @property
    def footprint(self) -> Polygon:
        """The ground polygon these pixels cover, in `crs`: what a search of this scene searched.

        Built from the four image corners rather than from a bounding box, so a transform with
        any rotation in it still describes the ground the radar actually looked at.
        """
        height, width = self.image.shape
        corners = ((0, 0), (width, 0), (width, height), (0, height))
        return Polygon([self.transform @ corner for corner in corners])


def read_scene(directory: str | Path) -> Scene:
    """Load a scene written by `write_scene` (directly, or via `data.synthetic.write_synthetic`)."""
    directory = Path(directory)
    meta = json.loads((directory / "scene.json").read_text())
    image = np.load(directory / "image.npy")
    return Scene(
        id=meta["id"],
        image=image,
        transform=Affine(*meta["transform"]),
        crs=meta["crs"],
        acquired_at=datetime.fromisoformat(meta["acquired_at"]),
        heading_deg=meta["heading_deg"],
        incidence_deg=meta["incidence_deg"],
    )


def write_scene(directory: str | Path, scene: Scene) -> None:
    """Write a scene in the layout `read_scene` expects."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "image.npy", scene.image)
    meta = {
        "id": scene.id,
        "transform": list(scene.transform)[:6],
        "crs": scene.crs,
        "acquired_at": scene.acquired_at.isoformat(),
        "heading_deg": scene.heading_deg,
        "incidence_deg": scene.incidence_deg,
    }
    (directory / "scene.json").write_text(json.dumps(meta, indent=2))
