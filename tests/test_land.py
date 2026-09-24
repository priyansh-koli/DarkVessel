"""The land mask: what is not searched, and what the declaration side makes of that."""

from datetime import timedelta

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely import box

from darkvessel.data.land import LandMask
from darkvessel.data.synthetic import (
    HEADING_DEG,
    INCIDENCE_DEG,
    RECEPTION_CELL_M,
    RECEPTION_FLOOR,
    TOLERANCE_M,
    _ais,
    _scene,
)
from darkvessel.data.tiling import Tiling
from darkvessel.detect.infer import detect_scene
from darkvessel.detect.stub import BrightPixelDetector
from darkvessel.fusion.azimuth import Geometry
from darkvessel.fusion.declarations import MASKED, UNDETECTED
from darkvessel.fusion.match import DARK
from darkvessel.fusion.reception import coverage_for
from darkvessel.pipeline import run

_DETECTOR = BrightPixelDetector(threshold=0.5)
_TILING = Tiling(tile_px=128, overlap_px=32)


def _fuse(land=None):
    ais = _ais()
    max_gap = timedelta(minutes=10)
    return run(
        scene=_scene(),
        ais=ais,
        detector=_DETECTOR,
        tiling=_TILING,
        tolerance_m=TOLERANCE_M,
        max_gap=max_gap,
        geometry=Geometry(heading_deg=HEADING_DEG, incidence_deg=INCIDENCE_DEG),
        coverage=coverage_for(ais, max_gap, cell_m=RECEPTION_CELL_M),
        reception_floor=RECEPTION_FLOOR,
        land=land,
    )


class _CountingImage:
    """An image that records which windows were read, as a GeoTIFF-backed scene would."""

    def __init__(self, array):
        self.array, self.shape, self.reads = array, array.shape, []

    def __getitem__(self, key):
        self.reads.append(key)
        return self.array[key]


def _pixel_box(scene, row0, col0, row1, col1):
    """The ground polygon covering pixel rows row0..row1 and columns col0..col1."""
    x0, y0 = scene.transform @ (col0, row0)
    x1, y1 = scene.transform @ (col1, row1)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def test_a_target_on_land_is_not_reported_and_one_at_sea_is():
    scene = _scene()
    image = np.zeros((256, 256), dtype=np.float32)
    image[40, 40] = 1.0  # on land
    image[200, 200] = 1.0  # at sea
    land = LandMask.from_geometries([_pixel_box(scene, 0, 0, 100, 100)])
    found = detect_scene(image, _DETECTOR, _TILING, mask=land.mask_for(scene))
    assert list(zip(found["row"], found["col"])) == [(200.0, 200.0)]


def test_a_tile_whose_core_is_all_land_is_never_read():
    scene = _scene()
    image = _CountingImage(np.zeros((256, 256), dtype=np.float32))
    land = LandMask.from_geometries([_pixel_box(scene, 0, 0, 256, 256)])
    detect_scene(image, _DETECTOR, _TILING, mask=land.mask_for(scene))
    assert image.reads == []


def test_no_land_changes_nothing():
    plain, masked = _fuse(), _fuse(land=LandMask.from_geometries([]))
    assert plain.detections["status"].tolist() == masked.detections["status"].tolist()
    assert plain.declarations["status"].tolist() == masked.declarations["status"].tolist()


def test_a_declaration_on_masked_land_is_masked_not_undetected():
    """The lane vessel is `undetected` in the fixture. Put land under it and it was never
    searched for, so it must not count against the detector."""
    baseline = _fuse().declarations
    lane = baseline[baseline["mmsi"] == "219100001"]
    assert lane["status"].iloc[0] == UNDETECTED
    where = lane.geometry.iloc[0]

    fusion = _fuse(land=LandMask.from_geometries([where.buffer(30.0)]))
    after = fusion.declarations.set_index("mmsi")["status"]
    assert after["219100001"] == MASKED
    assert fusion.agreement().masked == 1
    assert fusion.agreement().ais_only == baseline["status"].eq(UNDETECTED).sum() - 1
    # The dark vessel 245 m away is untouched by 30 m of land.
    assert fusion.detections["status"].eq(DARK).sum() == 1


def test_a_buffer_pushes_the_mask_out_to_sea():
    land = LandMask.from_geometries([box(0, 0, 100, 100)], buffer_m=50.0)
    assert land.geometry.contains(shapely.Point(140, 50))
    assert not land.geometry.contains(shapely.Point(160, 50))


def test_read_takes_only_nearby_land_and_reprojects_it(tmp_path):
    """Land is stored in lon/lat; the scene is in UTM. Far-away land is not read at all."""
    scene = _scene()
    near = _pixel_box(scene, 0, 0, 50, 50)
    far = near.buffer(50.0).centroid.buffer(10.0)
    far = shapely.affinity.translate(far, 200_000, 0)
    land = gpd.GeoDataFrame(geometry=[near, far], crs=scene.crs).to_crs("EPSG:4326")
    path = tmp_path / "land.gpkg"
    land.to_file(path)

    mask = LandMask.read(path, scene, buffer_m=0.0)
    assert mask.geometry.area == pytest.approx(near.area, rel=1e-3)


def test_read_with_no_land_nearby_gives_an_empty_mask(tmp_path):
    scene = _scene()
    far = shapely.affinity.translate(scene.footprint, 500_000, 0)
    path = tmp_path / "land.gpkg"
    gpd.GeoDataFrame(geometry=[far], crs=scene.crs).to_file(path)
    assert LandMask.read(path, scene).empty
