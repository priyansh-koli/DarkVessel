"""The synthetic fixture: no credentials, no weights, no network.

Built so every branch in the pipeline gets exercised on purpose. See the README's Quick start
for the numbers this produces: 5 detections, 4 matched, 1 dark, 0 at a fixed structure.

Five targets, five stories, all bright enough to detect at the default threshold:
  - `stationary_match`     a plain match: one AIS report, close to the detection.
  - `interpolated_match`   the failure case the README calls out. Its two AIS reports bracket
                            the acquisition and interpolate exactly onto the target, but
                            *either report alone* sits 900 m away — outside any sane
                            tolerance. Matched against a report taken as it stands, this
                            vessel would come back dark.
  - `azimuth_corrected_match`  a fast vessel whose *declared* position is hundreds of metres
                            from the detection until the azimuth-shift correction is applied,
                            after which it lands on it exactly.
  - `simple_match`         another plain match, for variety.
  - `dark_vessel`          no AIS report at all: a genuine dark candidate.

The interpolated vessel's velocity is placed along the satellite's flight direction on
purpose, so its own azimuth-shift correction is zero — its story stays about interpolation
alone, uncomplicated by the correction the other vessel is there to demonstrate.

Every viewer control should visibly change something, so the scene holds more than the five
stories, arranged so that none of it shows at the default settings:
  - targets differ in brightness, so raising the detector threshold drops them one by one;
  - `faint_trawler` and `faint_dark` sit below the default threshold, and appear only when
    it is lowered;
  - the trawler's only AIS report is 14 minutes old, and a sixth vessel (`219000006`) last
    reported 25 minutes before the pass, near the dark vessel. Both sit beyond the default
    10-minute max gap, so the defaults declare exactly the four story vessels; lengthening the
    gap brings them in, and with a wide enough tolerance the stale report "explains" the dark
    vessel — which is the trap a long max gap sets;
  - the plain matches sit 140 m and 70 m from their detections, so tightening the tolerance
    below those distances turns them dark.

Four more vessels exist only in the AIS, and carry the two questions the radar side cannot
answer on its own:
  - `219100001` runs a lane 245 m east of `dark_vessel`, reporting every two minutes. Nothing
    in the radar stands where its 90 m hull declares itself, so it comes back **undetected** —
    the mirror of a dark vessel — and its dense reporting is what tells the reception estimate
    that the archive hears this water well, which is what keeps `dark_vessel`'s darkness
    meaning something.
  - `219100002` drifts beside `faint_dark`, heard once an hour and never within the default
    max gap. Its neighbourhood's reception comes out at 1/3, so once the threshold is lowered
    far enough to detect `faint_dark`, that detection is reported **shadowed** rather than
    dark: the search could not have heard a vessel there, so its silence is not evidence.
    Widen the max gap past the hour and the same reports cover the instant, reception rises to
    1, and the shadow lifts — the one control moving both halves of the claim at once.
  - `219100003` is 12 m long, under the detector's floor, so its absence from the radar is
    expected and it is reported **below_detectable** rather than as a finding.
  - `219100004` declares itself outside the image, so nothing about it was searched at all:
    **outside_scene**, the declaration side's `unsearched`.

`faint_trawler` sits beside neither lane, so at a lowered threshold it is reported dark with
no reception estimate behind it — `insufficient_evidence`, next to `faint_dark`'s measured
shadow. The two look alike and are not: one search came back empty, the other never reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import cos, radians, sin
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from shapely import Point

from darkvessel.data.scene import Scene, write_scene
from darkvessel.fusion.azimuth import Geometry

CRS = "EPSG:25832"
PIXEL_SIZE_M = 5.0
IMAGE_SHAPE = (320, 320)  # rows, cols
ORIGIN = (499900.0, 6101100.0)  # ground (x, y) of the top-left pixel corner
ACQUIRED_AT = datetime(2026, 8, 9, 5, 31, 24, tzinfo=timezone.utc)
HEADING_DEG = 350.0
INCIDENCE_DEG = 35.0
TOLERANCE_M = 200.0

# The fixture's intended reception settings. A 200 m cell pools evidence over the 600 m
# neighbourhood around it, which is the right scale for a scene 1600 m across; a real
# Sentinel-1 scene is hundreds of times wider and wants `reception.RECEPTION_CELL_M`.
RECEPTION_CELL_M = 200.0
RECEPTION_FLOOR = 0.5

TRANSFORM = Affine(PIXEL_SIZE_M, 0.0, ORIGIN[0], 0.0, -PIXEL_SIZE_M, ORIGIN[1])


@dataclass(frozen=True)
class _Target:
    name: str
    x: float
    y: float
    brightness: float = 1.0


# Positions sit on the 5 m pixel grid, so a blob's centre lands a fixed half-pixel off its
# target. `dark_vessel` is placed 200 m from `simple_match` at a bearing of 60 degrees, and
# `simple_match`'s declaration 70 m along the same line (see `_ais_rows`).
_TARGETS = [
    _Target("stationary_match", 500180.0, 6100870.0, 0.9),
    _Target("interpolated_match", 500420.0, 6100120.0, 0.75),
    _Target("azimuth_corrected_match", 500980.0, 6100640.0, 0.65),
    _Target("simple_match", 500760.0, 6099800.0, 0.85),
    _Target("dark_vessel", 500935.0, 6099900.0, 0.6),
    # Below the default threshold of 0.5: invisible until the threshold is lowered.
    _Target("faint_trawler", 501290.0, 6100950.0, 0.35),
    _Target("faint_dark", 500245.0, 6099700.0, 0.4),
]


def _by_name(name: str) -> _Target:
    return next(t for t in _TARGETS if t.name == name)


def _blob(image: np.ndarray, x: float, y: float, value: float = 1.0) -> None:
    """Paint a 2x2 block of equal-value pixels — the flat plateau `detect.stub` must collapse
    to one detection rather than reporting per pixel."""
    col, row = ~TRANSFORM * (x, y)
    r, c = int(row), int(col)
    image[r : r + 2, c : c + 2] = value


def _scene() -> Scene:
    image = np.zeros(IMAGE_SHAPE, dtype=np.float32)
    for target in _TARGETS:
        _blob(image, target.x, target.y, target.brightness)
    return Scene(
        id="synthetic-scene-1",
        image=image,
        transform=TRANSFORM,
        crs=CRS,
        acquired_at=ACQUIRED_AT,
        heading_deg=HEADING_DEG,
        incidence_deg=INCIDENCE_DEG,
    )


def _flight_direction() -> tuple[float, float]:
    heading = radians(HEADING_DEG)
    return sin(heading), cos(heading)


def _ais_rows() -> list[dict]:
    rows = []

    stationary = _by_name("stationary_match")
    rows.append(
        dict(
            mmsi="219000001",
            timestamp=ACQUIRED_AT,
            length_m=50.0,
            # 139 m off: inside the default tolerance, outside one tightened below 140 m.
            geometry=Point(stationary.x + 110.0, stationary.y - 85.0),
        )
    )

    # Bracketing reports 5 minutes either side, offset 900 m along the flight direction so
    # velocity has no line-of-sight component: this vessel's azimuth shift is exactly zero,
    # keeping its story about interpolation alone. Either single report sits 900 m from the
    # target; the interpolated midpoint sits exactly on it.
    interpolated = _by_name("interpolated_match")
    flight_east, flight_north = _flight_direction()
    offset = 900.0
    rows.append(
        dict(
            mmsi="219000002",
            timestamp=ACQUIRED_AT - timedelta(minutes=5),
            length_m=80.0,
            geometry=Point(
                interpolated.x - offset * flight_east, interpolated.y - offset * flight_north
            ),
        )
    )
    rows.append(
        dict(
            mmsi="219000002",
            timestamp=ACQUIRED_AT + timedelta(minutes=5),
            length_m=80.0,
            geometry=Point(
                interpolated.x + offset * flight_east, interpolated.y + offset * flight_north
            ),
        )
    )

    # One report exactly at the acquisition, and one 10 minutes earlier so a velocity can be
    # derived from history — the same way a real archive would give one. The *raw* declared
    # position is computed by inverting the same `fusion.azimuth.Geometry` the pipeline
    # applies, so this fixture's correctness doesn't depend on hand-computed offsets tracking
    # that formula.
    azimuth = _by_name("azimuth_corrected_match")
    velocity_east, velocity_north = 8.0, 0.0
    geometry = Geometry(heading_deg=HEADING_DEG, incidence_deg=INCIDENCE_DEG)
    shift_east, shift_north = geometry.displacement(velocity_east, velocity_north, latitude=55.7)
    raw_at_acquisition = Point(azimuth.x - shift_east, azimuth.y - shift_north)
    raw_before = Point(
        raw_at_acquisition.x - velocity_east * 600.0,
        raw_at_acquisition.y - velocity_north * 600.0,
    )
    rows.append(
        dict(
            mmsi="219000003",
            timestamp=ACQUIRED_AT - timedelta(minutes=10),
            length_m=140.0,
            geometry=raw_before,
        )
    )
    rows.append(
        dict(
            mmsi="219000003",
            timestamp=ACQUIRED_AT,
            length_m=140.0,
            geometry=raw_at_acquisition,
        )
    )

    # 70 m from its own detection towards the dark vessel, so 130 m from that one: inside the
    # tolerance of both, and the one-to-one assignment must award it to the nearer.
    simple = _by_name("simple_match")
    towards_dark = sin(radians(60.0)), cos(radians(60.0))
    rows.append(
        dict(
            mmsi="219000004",
            timestamp=ACQUIRED_AT,
            length_m=60.0,
            geometry=Point(simple.x + 70.0 * towards_dark[0], simple.y + 70.0 * towards_dark[1]),
        )
    )

    # "dark_vessel" gets no AIS row at all — nothing declares it, so it stays dark.

    # A busy lane 245 m east of the dark vessel, reporting every two minutes for twenty
    # minutes either side of the pass. It runs along the flight direction, so its own azimuth
    # shift is zero and the 245 m gap does not depend on whether the correction is applied.
    # Two jobs: nothing is detected where it declares, so it is the `undetected` story; and
    # its dense reporting is the evidence that the archive hears this water well.
    rows.extend(
        _lane(
            mmsi="219100001",
            length_m=90.0,
            centre=Point(501180.0, 6099900.0),
            speed_ms=0.5,
            offsets_minutes=[m for m in range(-20, 21, 2)],
        )
    )

    # A drifter beside the faint dark target, heard once an hour and never within the default
    # max gap. Its hourly gaps are what put that neighbourhood's reception at 1/3.
    rows.extend(
        _lane(
            mmsi="219100002",
            length_m=30.0,
            centre=Point(500520.0, 6099700.0),
            speed_ms=0.005,
            offsets_minutes=[-270, -210, -150, -90, -30, 30, 90, 150, 210, 270],
        )
    )

    # 12 m, under the detector's floor: a miss here says nothing about this vessel.
    rows.append(
        dict(
            mmsi="219100003",
            timestamp=ACQUIRED_AT,
            length_m=12.0,
            geometry=Point(500500.0, 6100500.0),
        )
    )

    # Declared 300 m east of the image edge: outside the scene, so nothing was searched.
    rows.append(
        dict(
            mmsi="219100004",
            timestamp=ACQUIRED_AT,
            length_m=60.0,
            geometry=Point(501800.0, 6100300.0),
        )
    )

    # Stale reports, both beyond the default max gap. The trawler's lone report is 14 minutes
    # old and 60 m from it; the sixth vessel last reported 25 minutes before the pass, 320 m
    # from the dark vessel — close enough to "explain" it once both gap and tolerance are
    # loosened far enough.
    trawler = _by_name("faint_trawler")
    rows.append(
        dict(
            mmsi="219000005",
            timestamp=ACQUIRED_AT - timedelta(minutes=14),
            length_m=24.0,
            geometry=Point(trawler.x - 36.0, trawler.y - 48.0),
        )
    )
    dark = _by_name("dark_vessel")
    rows.append(
        dict(
            mmsi="219000006",
            timestamp=ACQUIRED_AT - timedelta(minutes=25),
            length_m=110.0,
            geometry=Point(dark.x + 192.0, dark.y - 256.0),
        )
    )
    return rows


def _lane(
    mmsi: str,
    length_m: float,
    centre: Point,
    speed_ms: float,
    offsets_minutes: list[float],
) -> list[dict]:
    """One vessel running along the satellite's ground track, reporting at these offsets.

    Along the flight direction on purpose: the line-of-sight component of that velocity is
    zero, so the azimuth correction moves the vessel nowhere and its distance from everything
    else in the scene is the same whether the correction is on or off.
    """
    flight_east, flight_north = _flight_direction()
    return [
        dict(
            mmsi=mmsi,
            timestamp=ACQUIRED_AT + timedelta(minutes=offset),
            length_m=length_m,
            geometry=Point(
                centre.x + speed_ms * offset * 60.0 * flight_east,
                centre.y + speed_ms * offset * 60.0 * flight_north,
            ),
        )
        for offset in offsets_minutes
    ]


def _ais() -> gpd.GeoDataFrame:
    frame = pd.DataFrame(_ais_rows())
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["mmsi"] = frame["mmsi"].astype("string")
    return gpd.GeoDataFrame(frame.drop(columns=["geometry"]), geometry=frame["geometry"], crs=CRS)


def write_synthetic(out_dir: str | Path) -> Path:
    """Write a scene and its AIS archive that exercise every branch in the pipeline on purpose.

    Returns the directory written to.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_scene(out_dir / "scene", _scene())
    _ais().to_file(out_dir / "ais.gpkg", driver="GPKG")
    return out_dir
