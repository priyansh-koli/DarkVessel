"""The viewer's payload and HTTP surface.

The payload is the single contract the frontend draws from — a live run and a pre-rendered
static build hand it exactly the same shape, so both modes are exercised here.
"""

import json

import pytest
from fastapi.testclient import TestClient

from darkvessel.data.synthetic import (
    RECEPTION_CELL_M,
    RECEPTION_FLOOR,
    TOLERANCE_M,
    _ais,
    _scene,
)
from darkvessel.web.app import create_app
from darkvessel.web.payload import RunRequest, Viewer, build


def _viewer(ais=None):
    """A viewer at the fixture's intended reception settings — see `data.synthetic`."""
    return Viewer(_scene(), _ais() if ais is None else ais, reception_cell_m=RECEPTION_CELL_M)


def _request(**values):
    values.setdefault("tolerance_m", TOLERANCE_M)
    values.setdefault("reception_floor", RECEPTION_FLOOR)
    return RunRequest(**values)


@pytest.fixture(scope="module")
def payload():
    return _viewer().run(_request())


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A live app over a freshly written synthetic fixture."""
    from darkvessel.data.synthetic import write_synthetic

    root = tmp_path_factory.mktemp("fixture")
    write_synthetic(root / "synthetic")
    config = root / "pipeline.yaml"
    config.write_text(
        f"scene_dir: {root / 'synthetic' / 'scene'}\n"
        f"ais_path: {root / 'synthetic' / 'ais.gpkg'}\n"
        "tolerance_m: 200\nmax_gap_minutes: 10\n"
        "reception_cell_m: 200\nreception_floor: 0.5\nsmallest_detectable_m: 20\n"
        "geometry:\n  heading_deg: 350.0\n  incidence_deg: 35.0\n"
    )
    return TestClient(create_app(config))


def test_the_payload_reproduces_the_readme_counts(payload):
    assert payload["counts"] == {
        "total": 5, "matched": 4, "dark": 1, "shadowed": 0, "structure": 0, "unsearched": 0
    }


def test_the_payload_is_json_serialisable(payload):
    """numpy scalars and pandas NA reach the frontend as plain JSON or not at all."""
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["counts"] == payload["counts"]


def test_every_detection_carries_pixel_coordinates_and_a_crop(payload):
    for detection in payload["detections"]:
        assert 0 <= detection["px"] <= payload["scene"]["width"]
        assert 0 <= detection["py"] <= payload["scene"]["height"]
        assert detection["crop"].startswith("data:image/png;base64,")


def test_absent_values_arrive_as_null_not_nan(payload):
    """`NaN` is not valid JSON, and a dark row is mostly absent values."""
    dark = next(d for d in payload["detections"] if d["status"] == "dark")
    assert dark["mmsi"] is None
    assert dark["match_distance_m"] is None
    assert "NaN" not in json.dumps(payload)


def test_declarations_carry_both_the_raw_and_the_drawn_position(payload):
    """The gap between the two is the azimuth correction the viewer draws as a dashed line."""
    moving = next(d for d in payload["declarations"] if (d["azimuth_shift_m"] or 0) > 1.0)
    assert moving["mmsi"] == "219000003"
    assert moving["raw"]["px"] != moving["drawn"]["px"]
    assert moving["matched_index"] is not None


def test_a_declaration_without_velocity_is_drawn_where_it_declared(payload):
    """No velocity means no shift was computed: it is drawn in place, and its shift is
    unknown (None), not a measured 0 m."""
    still = next(d for d in payload["declarations"] if d["mmsi"] == "219000001")
    assert still["azimuth_shift_m"] is None
    assert still["raw"]["px"] == pytest.approx(still["drawn"]["px"])


def test_turning_off_the_correction_reports_the_fast_vessel_dark():
    off = _viewer().run(_request(apply_azimuth=False))
    assert off["counts"]["matched"] == 3
    assert off["counts"]["dark"] == 2


def test_registering_the_dark_position_reclassifies_it():
    from darkvessel.data.synthetic import _by_name

    dark = _by_name("dark_vessel")
    marked = _viewer().run(_request(register_xy=[(dark.x, dark.y)]))
    assert marked["counts"]["structure"] == 1
    assert marked["counts"]["dark"] == 0
    assert marked["counts"]["matched"] == 4  # matches are never overridden


def test_the_ais_summary_reports_the_cleaning_rules(payload):
    assert payload["ais"]["rows_in"] == 41
    assert payload["ais"]["vessels"] == 10
    assert "duplicate_mmsi_timestamp" in payload["ais"]["removed"]


def test_no_ais_means_unsearched_and_no_declarations():
    nothing = build(_scene(), None, RunRequest(tolerance_m=TOLERANCE_M))
    assert nothing["counts"]["unsearched"] == 5
    assert nothing["declarations"] == []
    assert nothing["ais"] is None
    # There is nothing to estimate reception from either, and the payload says so rather
    # than shipping an empty model that would read as "heard nowhere".
    assert nothing["reception"] is None
    assert nothing["declaration_counts"]["total"] == 0
    assert nothing["agreement"]["apparent_recall"] is None


def test_the_scene_transform_is_shipped_for_the_viewer(payload):
    """The frontend turns a click into a coordinate with this rather than its own affine."""
    a, b, c, d, e, f = payload["scene"]["transform"]
    detection = payload["detections"][0]
    assert a * detection["px"] + b * detection["py"] + c == pytest.approx(detection["x"])
    assert d * detection["px"] + e * detection["py"] + f == pytest.approx(detection["y"])


# ───────────────────────────── HTTP surface ─────────────────────────────


def test_health_reports_the_loaded_scene(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["scene"] == "synthetic-scene-1"


def test_the_scene_png_is_served(client):
    response = client.get("/api/scene.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_run_honours_the_tolerance_query(client):
    """At 5 m every declaration falls outside tolerance, so nothing is explained. Four of the
    five read dark; the fifth stands where the archive is heard a third of the time, so this
    configuration's floor reports it shadowed instead. Both are unexplained by AIS — which is
    what the tolerance moved — and the split between them is the reception claim, not this one."""
    tight = client.get("/api/run", params={"tolerance_m": 5}).json()["counts"]
    assert tight["matched"] == 0
    assert tight["dark"] + tight["shadowed"] == 5
    assert client.get("/api/run", params={"tolerance_m": 200}).json()["counts"]["matched"] == 4


def test_run_honours_the_azimuth_toggle(client):
    off = client.get("/api/run", params={"apply_azimuth": False}).json()
    assert off["counts"]["dark"] == 2


def test_run_accepts_registered_positions(client):
    response = client.get("/api/run", params={"register": ["500935,6099900"]})
    assert response.json()["counts"]["structure"] == 1


def test_a_malformed_registered_position_is_refused(client):
    assert client.get("/api/run", params={"register": ["not-a-point"]}).status_code == 400
    assert client.get("/api/run", params={"register": ["1,2,3"]}).status_code == 400


def test_an_out_of_range_tolerance_is_refused(client):
    assert client.get("/api/run", params={"tolerance_m": -5}).status_code == 422


def test_the_frontend_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "darkvessel" in response.text
    assert "app.js" in response.text


def test_health_reports_the_configured_defaults(client):
    """Reset returns the controls here, so it must be the run configuration."""
    defaults = client.get("/api/health").json()["defaults"]
    assert defaults["tolerance_m"] == 200
    assert defaults["apply_azimuth"] is True


def test_the_run_payload_is_compressed(client):
    response = client.get("/api/run", headers={"Accept-Encoding": "gzip"})
    assert response.headers.get("content-encoding") == "gzip"


def test_the_guide_and_how_it_works_pages_are_served(client):
    for page in ("guide.html", "about.html", "pipeline.svg", "docs.js"):
        assert client.get(f"/{page}").status_code == 200


def test_the_document_pages_load_their_script_and_list_their_sections(client):
    """Both pages enhance themselves with docs.js, and every contents entry must point at a
    heading that exists — a stale entry is a dead link in the only navigation they have."""
    import re

    for page in ("guide.html", "about.html"):
        html = client.get(f"/{page}").text
        assert 'src="docs.js"' in html, page
        targets = set(re.findall(r'\sid="([^"]+)"', html))
        for href in re.findall(r'<a href="#([^"]+)">', html):
            assert href in targets, f"{page}: contents entry #{href} has no heading"


def test_the_overlay_does_not_hide_its_own_focusable_detections(client):
    """The detection marks are focusable buttons built into #overlay. aria-hidden on the
    overlay would take every one of them out of the accessibility tree while leaving them in
    the tab order, which is the worst of both."""
    html = client.get("/").text
    overlay = html[html.index('<svg id="overlay"') :].split(">", 1)[0]
    assert "aria-hidden" not in overlay


def test_the_page_does_not_make_itself_a_scroll_container(client):
    """`overflow-x: hidden` on html/body turns them into scroll containers, which silently
    disables every `position: sticky` on the page — the control column, the inspector column
    and the contents rail all depend on it."""
    css = client.get("/styles.css").text
    assert "overflow-x: clip" in css
    assert "overflow-x: hidden" not in css


# ───────────────────────────── the cached viewer ─────────────────────────────


def test_a_held_viewer_gives_the_same_payload_as_a_one_off_build(payload):
    viewer = _viewer()
    viewer.run(_request(tolerance_m=5))  # warm the caches with a different request
    assert viewer.run(_request()) == payload


def test_detection_runs_once_per_detector_setting(monkeypatch):
    """Moving a fusion control must not re-read every pixel of the scene."""
    import darkvessel.web.payload as payload_module

    calls = []
    real = payload_module.detect_scene
    monkeypatch.setattr(
        payload_module, "detect_scene", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    viewer = _viewer()
    for tolerance in (50, 100, 200):
        for azimuth in (True, False):
            viewer.run(RunRequest(tolerance_m=tolerance, apply_azimuth=azimuth))
    assert len(calls) == 1


def test_reception_is_estimated_once_per_max_gap(monkeypatch):
    """It depends on the archive and the gap alone, so moving the tolerance must not rebuild
    it — the whole reason `fuse` takes a pre-built model."""
    import darkvessel.web.payload as payload_module

    calls = []
    real = payload_module.coverage_for
    monkeypatch.setattr(
        payload_module, "coverage_for", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    viewer = _viewer()
    for tolerance in (50, 100, 200):
        for floor in (0.0, 0.5):
            viewer.run(_request(tolerance_m=tolerance, reception_floor=floor))
    viewer.run(_request(max_gap_minutes=90))
    assert len(calls) == 2  # one per distinct max gap, not one per run


def test_the_archive_is_cleaned_before_it_is_matched():
    """The cleaning report on screen must describe the archive the match searched. A report
    cleaning rejects (here, a 40 m/s vessel) sitting right on the dark vessel must not
    explain it."""
    import numpy as np
    import pandas as pd
    from shapely import Point

    from darkvessel.data.synthetic import ACQUIRED_AT, _by_name

    raw = _ais()
    raw["speed_ms"] = np.nan
    dark = _by_name("dark_vessel")
    bogus = raw.iloc[[0]].copy()
    bogus["mmsi"] = "219999999"
    bogus["timestamp"] = pd.Timestamp(ACQUIRED_AT)
    bogus["speed_ms"] = 40.0
    bogus = bogus.set_geometry([Point(dark.x, dark.y)], crs=raw.crs)
    dirty = pd.concat([raw, bogus], ignore_index=True)

    shown = build(_scene(), dirty, RunRequest(tolerance_m=TOLERANCE_M))
    assert shown["ais"]["removed"]["implausible_speed"] == 1
    assert shown["counts"]["dark"] == 1


# ───────────────────────────── the static bake ─────────────────────────────


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    from darkvessel.web.bake import bake

    out = tmp_path_factory.mktemp("bake")
    default = _request()
    viewer = _viewer()
    return viewer, default, bake(viewer, default, out), out


def _baked_run(baked, **values):
    viewer, default, manifest, out = baked
    flat = 0
    for key in manifest["order"]:
        grid = manifest["grid"][key]
        flat = flat * len(grid) + grid.index(values.get(key, manifest["defaults"][key]))
    return json.loads((out / "runs" / f"{manifest['runs'][flat]}.json").read_text())


def test_the_bake_covers_every_control_position(baked):
    _, _, manifest, _ = baked
    expected = 1
    for key in manifest["order"]:
        expected *= len(manifest["grid"][key])
    assert len(manifest["runs"]) == expected
    assert manifest["distinct_runs"] < len(manifest["runs"])  # identical outcomes collapse


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"apply_azimuth": False},
        {"tolerance_m": 25.0},
        {"max_gap_minutes": 1.0, "detector_threshold": 0.95},
        {"max_gap_minutes": 120.0, "tolerance_m": 1000.0, "apply_azimuth": False},
    ],
)
def test_a_baked_run_is_exactly_the_live_run(baked, values):
    """Shared equivalence classes must never change an answer, only save the work."""
    from dataclasses import replace

    viewer, default, manifest, out = baked
    live = viewer.run(replace(default, **values))
    live.pop("config")
    # The reception model is written once per max-gap class rather than copied into every run
    # that shares one, so it is compared from its own file.
    reception = live.pop("reception")
    assert _baked_run(baked, **values) == json.loads(json.dumps(live))

    gap = values.get("max_gap_minutes", manifest["defaults"]["max_gap_minutes"])
    at = manifest["grid"]["max_gap_minutes"].index(gap)
    hoisted = json.loads((out / "reception" / f"{manifest['reception'][at]}.json").read_text())
    assert hoisted == json.loads(json.dumps(reception))


def test_the_reception_model_is_baked_once_per_max_gap(baked):
    """Two gaps that declare the same vessels can still disagree about reception, so the gap
    equivalence class has to account for both — otherwise a static build would serve one
    gap's answer to the other's question."""
    _, _, manifest, out = baked
    gaps = manifest["grid"]["max_gap_minutes"]
    # One entry per gap, positionally: the frontend looks it up by index, because a gap of
    # 10.0 reaches JavaScript as the number 10 and would never match the key "10.0".
    assert len(manifest["reception"]) == len(gaps)
    ids = set(manifest["reception"])
    assert len(ids) > 1, "reception must move with the max gap"
    assert len(ids) == len(list((out / "reception").glob("*.json")))


def test_the_bake_reproduces_the_azimuth_story(baked):
    assert _baked_run(baked)["counts"]["dark"] == 1
    assert _baked_run(baked, apply_azimuth=False)["counts"]["dark"] == 2


def test_declarations_carry_course_and_speed_for_the_radar_view(payload):
    """A lone report has no velocity, so it has no course — absent, not zero."""
    by_mmsi = {d["mmsi"]: d for d in payload["declarations"]}
    fast = by_mmsi["219000003"]
    assert fast["course_deg"] == pytest.approx(90.0, abs=0.5)
    assert fast["speed_kn"] > 10
    assert by_mmsi["219000001"]["course_deg"] is None
    assert by_mmsi["219000001"]["speed_kn"] is None


def test_every_fusion_and_detector_control_changes_the_result():
    """The fixture is built so that no viewer control is inert: each one, moved away from its
    default, changes the counts."""
    viewer = _viewer()
    default = viewer.run(_request())["counts"]
    for changed in (
        _request(tolerance_m=100),
        _request(detector_threshold=0.3),
        _request(detector_threshold=0.7),
        _request(max_gap_minutes=30, tolerance_m=350),
    ):
        assert viewer.run(changed)["counts"] != default, changed
    stale_gap = viewer.run(_request(detector_threshold=0.3, max_gap_minutes=15))["counts"]
    assert stale_gap != viewer.run(_request(detector_threshold=0.3))["counts"]


# ───────────────────────── reception and the other side ─────────────────────────


def test_every_detection_carries_its_reception_estimate(payload):
    for detection in payload["detections"]:
        assert "reception_p" in detection
        assert detection["reception_basis"] in ("estimated", "insufficient_evidence")


def test_the_reception_model_ships_cells_the_viewer_can_draw(payload):
    reception = payload["reception"]
    assert reception["cell_m"] == RECEPTION_CELL_M
    assert reception["floor"] == RECEPTION_FLOOR
    assert reception["cells"], "nothing to draw"
    scene = payload["scene"]
    for cell in reception["cells"]:
        # Clipped to the scene: a cell nobody can see is payload nobody needs.
        assert cell["px"] < scene["width"] and cell["px"] + cell["pw"] > 0
        assert cell["py"] < scene["height"] and cell["py"] + cell["ph"] > 0


def test_raising_the_floor_moves_dark_detections_into_the_shadow():
    viewer = _viewer()
    at_zero = viewer.run(_request(detector_threshold=0.3, reception_floor=0.0))["counts"]
    at_half = viewer.run(_request(detector_threshold=0.3, reception_floor=0.5))["counts"]
    assert at_zero["shadowed"] == 0
    assert at_half["shadowed"] == 1
    assert at_zero["dark"] == at_half["dark"] + at_half["shadowed"]


def test_the_declaration_side_is_counted_and_listed(payload):
    assert payload["declaration_counts"] == {
        "total": 8,
        "explained": 4,
        "undetected": 2,
        "below_detectable": 1,
        "outside_scene": 1,
        "masked": 0,
    }
    undetected = [d for d in payload["declarations"] if d["status"] == "undetected"]
    assert {d["mmsi"] for d in undetected} == {"219100001", "219100002"}
    assert all(d["nearest_detection_m"] > TOLERANCE_M for d in undetected)


def test_the_agreement_gives_a_recall_estimate_without_labels(payload):
    agreement = payload["agreement"]
    assert agreement["both"] == 4
    assert agreement["ais_only"] == 2
    assert agreement["apparent_recall"] == pytest.approx(4 / 6)
    assert "apparent recall" in agreement["line"]


def test_run_honours_the_reception_floor_query(client):
    low = client.get("/api/run", params={"detector_threshold": 0.3, "reception_floor": 0}).json()
    high = client.get("/api/run", params={"detector_threshold": 0.3, "reception_floor": 1}).json()
    assert low["counts"]["shadowed"] == 0
    assert high["counts"]["shadowed"] > 0


def test_an_out_of_range_reception_floor_is_refused(client):
    assert client.get("/api/run", params={"reception_floor": 1.5}).status_code == 422


def test_the_configured_geometry_drives_the_azimuth_correction():
    """The viewer corrects with the configured geometry, as `darkvessel run` does, not the
    scene's defaults. A shorter `shift_per_mps` shrinks every shift in proportion."""
    from darkvessel.fusion.azimuth import Geometry

    scene = _scene()
    default = _viewer().run(_request())
    shorter = Viewer(
        scene,
        _ais(),
        reception_cell_m=RECEPTION_CELL_M,
        geometry=Geometry(scene.heading_deg, scene.incidence_deg, shift_per_mps=60.0),
    ).run(_request())

    shifted = [
        (d["azimuth_shift_m"], s["azimuth_shift_m"])
        for d, s in zip(default["declarations"], shorter["declarations"])
        if (d["azimuth_shift_m"] or 0) > 1
    ]
    assert shifted
    for full, short in shifted:
        assert short == pytest.approx(full * 60.0 / 113.0, rel=1e-6)
    # The fast vessel's drawn position no longer reaches its detection, as on the command line.
    assert (shorter["counts"]["matched"], shorter["counts"]["dark"]) == (3, 2)
