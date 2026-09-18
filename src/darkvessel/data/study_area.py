"""The study area: the CRS and bounds every layer in this project is expressed in."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StudyArea:
    crs: str
    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)

    def contains(self, x: float, y: float) -> bool:
        return self.minx <= x <= self.maxx and self.miny <= y <= self.maxy
