"""The CFAR baseline, detector scoring, the LS-SSDD audit, and the detector factory.

None of these need the real dataset or torch: each is exercised on a constructed case whose
right answer is known.
"""

from pathlib import Path

import numpy as np
import pytest

from darkvessel.detect import evaluate
from darkvessel.detect import lsssdd as L
from darkvessel.detect.cfar import CFARDetector
from darkvessel.detect.factory import make_detector


def _sea(shape=(200, 200), seed=0):
    rng = np.random.default_rng(seed)
    return rng.gamma(4.0, 0.05, size=shape).astype(np.float32) + 0.01


# ───────────────────────────── CFAR ─────────────────────────────


def test_cfar_finds_a_bright_hull_in_clutter():
    sea = _sea()
    sea[100:103, 60:66] = 3.0
    points = CFARDetector()(sea)
    assert len(points) == 1
    row, col = points[0]
    assert abs(row - 101) < 1.5 and abs(col - 62.5) < 1.5


def test_cfar_judges_each_pixel_against_its_own_clutter():
    """The same absolute brightness is a ship on calm sea and nothing on rough sea."""
    sea = _sea()
    sea[:, 100:] *= 8.0  # the right half is rough
    sea[50:53, 40:43] = 1.2  # stands far out of the calm half
    sea[150:153, 150:153] = 1.2  # buried in the rough half
    points = CFARDetector()(sea)
    assert len(points) == 1
    assert points[0][1] < 100


def test_the_swath_edge_is_not_a_line_of_ships():
    """A cliff from no-data (exactly zero) to sea must not read as targets."""
    sea = _sea()
    sea[:, :80] = 0.0
    assert CFARDetector()(sea) == []


def test_cfar_never_detects_in_no_data():
    sea = _sea()
    sea[:, :80] = 0.0
    z = CFARDetector().statistic(sea)
    assert (z[:, :80] == 0).all()


# ───────────────────────────── scoring ─────────────────────────────


def test_a_point_inside_a_box_is_a_hit_and_outside_is_not():
    boxes = np.array([[10, 10, 20, 20]], dtype=np.float32)
    points = np.array([[15, 15, 0.9], [50, 50, 0.8]], dtype=np.float32)
    assert evaluate.match_image(points, boxes).tolist() == [True, False]


def test_two_points_on_one_ship_are_one_hit_and_one_false_alarm():
    boxes = np.array([[10, 10, 20, 20]], dtype=np.float32)
    points = np.array([[14, 14, 0.6], [16, 16, 0.9]], dtype=np.float32)
    hit = evaluate.match_image(points, boxes)
    assert hit.tolist() == [False, True]  # the higher-scoring point takes the ship


def test_matching_maximises_hits_rather_than_taking_the_best_score_first():
    """The strongest point sits in both boxes; greedy would strand the weaker one."""
    boxes = np.array([[0, 0, 10, 10], [8, 0, 18, 10]], dtype=np.float32)
    points = np.array([[5, 9, 0.9], [5, 2, 0.5]], dtype=np.float32)
    assert evaluate.match_image(points, boxes).sum() == 2


def test_best_f1_and_ap_on_a_known_ranking():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    hits = np.array([True, True, False, True])
    best = evaluate.best_f1(scores, hits, ships=4)
    assert best.true_positives == 3 and best.false_positives == 1
    assert evaluate.average_precision(scores, hits, ships=4) == pytest.approx(
        0.25 * 1 + 0.25 * 1 + 0.25 * 0.75
    )


# ───────────────────────────── the LS-SSDD audit ─────────────────────────────


def _voc(path: Path, boxes):
    objects = "".join(
        f"<object><name>ship</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
        f"<xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
        for a, b, c, d in boxes
    )
    path.write_text(f"<annotation>{objects}</annotation>")


@pytest.fixture
def tiny(tmp_path):
    from PIL import Image

    root = tmp_path / "LS-SSDD"
    for folder in ("JPEGImages_sub", "Annotations_sub", "ImageSets/Main"):
        (root / folder).mkdir(parents=True)
    image = (np.ones((800, 800)) * 60).astype(np.uint8)
    for name in ("01_1_1", "11_1_1"):
        Image.fromarray(image).save(root / "JPEGImages_sub" / f"{name}.jpg")
    _voc(
        root / "Annotations_sub" / "01_1_1.xml",
        [
            (10, 10, 20, 20),
            (10, 10, 20, 20),  # exact duplicate
            (11, 11, 21, 21),  # the same ship boxed again, IoU 0.68 -> kept
            (30, 30, 40, 40),
            (31, 30, 41, 40),  # the same ship boxed again, IoU 0.82 -> merged
            (790, 790, 810, 812),  # runs past the edge
            (50, 60, 40, 70),  # corners swapped
            (100, 100, 100.5, 110),  # thinner than a pixel
        ],
    )
    (root / "ImageSets/Main/train.txt").write_text("01_1_1\n")
    (root / "ImageSets/Main/test.txt").write_text("11_1_1\n")
    (root / "ImageSets/Main/test_inshore.txt").write_text("")
    (root / "ImageSets/Main/test_offshore.txt").write_text("11_1_1\n")
    return root


def test_the_audit_fixes_and_counts_every_label_problem(tiny):
    audit = L.Audit()
    [sample] = L.load(tiny, ["01_1_1"], audit)
    counts = audit.counts
    assert counts["exact duplicate box (dropped)"] == 1
    assert counts["same ship boxed twice, IoU > 0.7 (merged)"] == 1
    assert counts["box past the image edge (clipped)"] == 1
    assert counts["box corners swapped (fixed)"] == 1
    assert counts["box under one pixel (dropped)"] == 1
    assert len(sample.boxes) == 5
    assert sample.boxes.max() <= 800


def test_a_missing_label_file_means_no_ships_and_is_counted(tiny):
    audit = L.Audit()
    [sample] = L.load(tiny, ["11_1_1"], audit)
    assert len(sample.boxes) == 0
    assert audit.counts["label file missing (kept as no ships)"] == 1


def test_an_unreadable_image_is_dropped_and_counted(tiny):
    (tiny / "JPEGImages_sub" / "11_1_1.jpg").write_bytes(b"not a jpeg")
    audit = L.Audit()
    assert L.load(tiny, ["11_1_1"], audit) == []
    assert audit.counts["image unreadable (dropped)"] == 1


def test_validation_is_split_by_scene_not_by_sub_image(tmp_path):
    main = tmp_path / "ImageSets" / "Main"
    main.mkdir(parents=True)
    main.joinpath("train.txt").write_text("01_1_1\n05_1_1\n05_1_2\n10_3_3\n")
    assert L.split_names(tmp_path, "val") == ["05_1_1", "05_1_2", "10_3_3"]
    assert L.split_names(tmp_path, "train") == ["01_1_1"]


def test_the_split_audit_catches_a_scene_on_both_sides(tiny):
    (tiny / "ImageSets/Main/test.txt").write_text("11_1_1\n01_2_2\n")
    (tiny / "ImageSets/Main/test_offshore.txt").write_text("11_1_1\n01_2_2\n")
    assert L.audit_splits(tiny).counts["scene in both train and test"] == 1


def test_centres_are_row_col():
    sample = L.Sample("x", Path("x"), np.array([[10, 20, 30, 60]], dtype=np.float32))
    assert sample.centres.tolist() == [[40.0, 20.0]]


# ───────────────────────────── factory ─────────────────────────────


def test_the_factory_builds_each_kind():
    from darkvessel.detect.stub import BrightPixelDetector

    assert isinstance(make_detector("stub"), BrightPixelDetector)
    assert isinstance(make_detector("cfar", 6.0), CFARDetector)
    assert make_detector("cfar", 6.0).threshold == 6.0
    with pytest.raises(ValueError, match="detector_weights"):
        make_detector("cnn")
    with pytest.raises(ValueError, match="unknown detector"):
        make_detector("yolo")


def test_a_config_without_a_detector_is_the_stub_at_half_brightness(tmp_path):
    from darkvessel import config

    path = tmp_path / "c.yaml"
    path.write_text("scene_dir: s\ntolerance_m: 200\n")
    cfg = config.load(path)
    assert cfg.detector == "stub" and cfg.detector_threshold == 0.5
    path.write_text("scene_dir: s\ntolerance_m: 200\ndetector: cnn\ndetector_weights: m.pt\n")
    cfg = config.load(path)
    assert cfg.detector == "cnn" and cfg.detector_threshold is None
    assert cfg.detector_weights == Path("m.pt")


# ───────────────────────────── the CNN (needs torch) ─────────────────────────────


def test_prepare_is_scale_invariant():
    """8-bit chips and calibrated backscatter must look the same to the network."""
    pytest.importorskip("torch")
    from darkvessel.detect.cnn import prepare

    sea = _sea()
    a, b = prepare(sea), prepare(sea * 250.0)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_prepare_marks_no_data():
    pytest.importorskip("torch")
    from darkvessel.detect.cnn import prepare

    sea = _sea()
    sea[:, :50] = 0.0
    x = prepare(sea)
    assert (x[1][:, :50] == 0).all() and (x[1][:, 50:] == 1).all()
    assert (x[0][:, :50] == 0).all()


def test_peaks_are_local_maxima_above_the_floor_and_in_swath():
    torch = pytest.importorskip("torch")
    from darkvessel.detect.cnn import peaks

    heat = torch.zeros(20, 20)
    heat[5, 5], heat[5, 6], heat[15, 15], heat[10, 2] = 0.9, 0.5, 0.3, 0.95
    valid = torch.ones(20, 20)
    valid[:, :3] = 0
    found = peaks(heat, valid, floor=0.25)
    assert sorted(map(tuple, found[:, :2].tolist())) == [(5.0, 5.0), (15.0, 15.0)]


def test_the_heatmap_peaks_at_exactly_one_on_each_ship():
    pytest.importorskip("torch")
    from darkvessel.detect.train import render_heatmap

    heat = render_heatmap(np.array([[10, 10, 20, 20], [40, 40, 44, 44]], np.float32), (64, 64))
    assert heat[15, 15] == 1.0 and heat[42, 42] == 1.0
    assert (heat == 1.0).sum() == 2
    assert heat[0, 0] < 1e-3


def test_the_shipped_weights_find_a_hull_and_ignore_the_swath_edge():
    """The committed model, end to end: loads, finds one ship in clutter, nothing else."""
    pytest.importorskip("torch")
    weights = Path(__file__).parent.parent / "models" / "ship_centrenet.pt"
    if not weights.exists():
        pytest.skip("no trained weights in this checkout")
    detector = make_detector("cnn", weights=weights)
    sea = _sea((256, 256))
    assert detector(sea) == []
    hull = sea.copy()
    hull[120:124, 80:92] = 1.5
    [(row, col)] = detector(hull)
    assert 119 <= row <= 125 and 79 <= col <= 92
    edge = sea.copy()
    edge[:, :100] = 0.0
    assert detector(edge) == []
