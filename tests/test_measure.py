"""Scale and measurements (core/measure): no GUI."""
import csv
import math
import threading

import numpy as np
import pytest
from PIL import Image

from core.database.connection import DatabaseConnection
from core.measure import morphometrics as m
from core.measure import scale as sc
from core.measure.export import ExportCancelled, columns, export_measurements
from services import pil_limits


def rect(length, width, deg, cx=500.0, cy=500.0):
    """Corners of a length x width rectangle, long side at `deg` (counter-clockwise on screen)."""
    a = math.radians(deg)
    u = np.array([math.cos(a), -math.sin(a)])
    v = np.array([math.sin(a), math.cos(a)])
    return np.array([[cx, cy] + su * length / 2 * u + sv * width / 2 * v
                     for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))], np.float32)


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------
@pytest.mark.parametrize("deg", [0, 30, 90, 135])
def test_rotated_rectangle(deg):
    r = m.measure(rect(100, 20, deg))
    assert r["length"] == pytest.approx(100, abs=1e-3)
    assert r["width"] == pytest.approx(20, abs=1e-3)
    assert r["area"] == pytest.approx(2000, rel=1e-4)
    assert r["perimeter"] == pytest.approx(240, rel=1e-4)
    assert r["orientation_deg"] == pytest.approx(deg % 180, abs=1e-3) or \
        r["orientation_deg"] == pytest.approx(180, abs=1e-3)
    assert r["aspect_ratio"] == pytest.approx(5, rel=1e-4)
    assert r["solidity"] == pytest.approx(1)
    assert (r["centroid_x"], r["centroid_y"]) == (pytest.approx(500), pytest.approx(500))


def test_circle_and_concave_shape():
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    c = m.measure(np.c_[50 * np.cos(t), 50 * np.sin(t)] + 100)
    assert c["circularity"] == pytest.approx(1, abs=1e-3)
    assert c["area"] == pytest.approx(math.pi * 2500, rel=1e-3)
    ell = m.measure([[0, 0], [10, 0], [10, 2], [2, 2], [2, 10], [0, 10]])   # an L
    assert ell["area"] == pytest.approx(36) and ell["solidity"] == pytest.approx(36 / 68)


def test_orientation_follows_the_screen():
    # long side going up-right on screen (y decreases) is 45 degrees
    assert m.measure(rect(100, 10, 45))["orientation_deg"] == pytest.approx(45, abs=1e-3)
    assert m.measure([[0, 0], [100, 0], [100, 10], [0, 10]])["orientation_deg"] in (
        pytest.approx(0, abs=1e-6), pytest.approx(180, abs=1e-6))


def test_degenerate_outlines():
    assert m.measure([[0, 0], [1, 1]]) is None
    assert m.measure([[0, 0], [1, 1], [2, 2]]) is None


def test_to_units():
    px = m.measure(rect(100, 20, 0))
    mm = m.to_units(px, um_per_px=10.0, unit_um=1000.0)      # 10 µm per px
    assert mm["length"] == pytest.approx(1.0) and mm["area"] == pytest.approx(0.2)
    assert mm["orientation_deg"] == px["orientation_deg"] and mm["solidity"] == px["solidity"]
    none = m.to_units(px, None, 1000.0)
    assert none["length"] is None and none["circularity"] == px["circularity"]


@pytest.mark.parametrize("um_per_screen_px, text", [(10.58, "1 mm"), (2.6, "200 µm"),
                                                     (84.0, "10 mm"), (0.5, "50 µm")])
def test_scale_bar_lengths_are_round(um_per_screen_px, text):
    um = m.nice_bar_length(um_per_screen_px)
    assert m.format_length(um) == text
    assert 50 <= um / um_per_screen_px <= 250


# --------------------------------------------------------------------------
# Scale from files and project
# --------------------------------------------------------------------------
def test_dpi_from_files(tmp_path):
    rgb = Image.new("RGB", (20, 10))
    rgb.save(tmp_path / "scan.tif", dpi=(2400, 2400))
    rgb.save(tmp_path / "photo.jpg", dpi=(300, 300))
    rgb.save(tmp_path / "default72.jpg", dpi=(72, 72))
    rgb.save(tmp_path / "default96.png", dpi=(96, 96))
    rgb.save(tmp_path / "none.png")
    assert sc.file_um_per_px(tmp_path / "scan.tif") == pytest.approx(25400 / 2400)
    assert sc.file_um_per_px(tmp_path / "photo.jpg") == pytest.approx(25400 / 300, rel=1e-3)
    for name in ("default72.jpg", "default96.png", "none.png", "missing.png"):
        assert sc.file_um_per_px(tmp_path / name) is None


def test_dpi_in_centimetres(tmp_path):
    from PIL import TiffImagePlugin
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[282] = 1000.0     # pixels per cm
    info[283] = 1000.0
    info[296] = 3          # unit: cm
    Image.new("L", (20, 10)).save(tmp_path / "cm.tif", tiffinfo=info)
    assert sc.file_um_per_px(tmp_path / "cm.tif") == pytest.approx(10.0)


def test_big_scan_header_is_read(tmp_path, monkeypatch):
    path = tmp_path / "big.tif"
    Image.new("L", (64, 64)).save(path, dpi=(2400, 2400))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)     # "too big" for PIL's default check
    assert sc.file_um_per_px(path) == pytest.approx(25400 / 2400)
    assert Image.MAX_IMAGE_PIXELS == 10


def test_project_scale_falls_back_and_round_trips(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "scan.tif", dpi=(2400, 2400))
    Image.new("RGB", (8, 8)).save(tmp_path / "crop.png")
    s = sc.ProjectScale(um_per_px=5.0, source="ruler", unit="µm", show_bar=False)
    assert s.for_image(tmp_path / "scan.tif") == (pytest.approx(10.5833, rel=1e-4), sc.FROM_FILE)
    assert s.for_image(tmp_path / "crop.png") == (5.0, sc.FROM_PROJECT)
    s.use_file_dpi = False
    assert s.for_image(tmp_path / "scan.tif") == (5.0, sc.FROM_PROJECT)
    assert sc.ProjectScale().for_image(tmp_path / "crop.png") == (None, "")

    (tmp_path / "p.SEproj").write_text('{"name": "p", "images": ["scan.tif"]}', encoding="utf-8")
    s.save(tmp_path)
    loaded = sc.ProjectScale.load(tmp_path)
    assert loaded.to_dict() == s.to_dict()
    import json
    assert json.loads((tmp_path / "p.SEproj").read_text(encoding="utf-8"))["images"] == ["scan.tif"]


def test_bad_scale_settings_are_ignored():
    s = sc.ProjectScale.from_dict({"um_per_px": "abc", "unit": "furlong"})
    assert s.um_per_px is None and s.unit == "mm"
    assert sc.ProjectScale.from_dict(None).use_file_dpi


def test_pil_limit_switch_is_thread_safe():
    before = Image.MAX_IMAGE_PIXELS
    seen = []
    barrier = threading.Barrier(8)

    def worker():
        with pil_limits.large_images_allowed():
            barrier.wait()
            seen.append(Image.MAX_IMAGE_PIXELS)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == [None] * 8 and Image.MAX_IMAGE_PIXELS == before


# --------------------------------------------------------------------------
# CSV export
# --------------------------------------------------------------------------
@pytest.fixture
def project(tmp_path):
    imgs = tmp_path / "images"
    imgs.mkdir()
    Image.new("RGB", (1000, 1000)).save(imgs / "scan.tif", dpi=(2540, 2540))   # 10 µm per px
    Image.new("RGB", (1000, 1000)).save(imgs / "crop.png")                     # no resolution
    db = DatabaseConnection(str(tmp_path / "masks.db"))
    for image, cls, shape in (("scan", "daphnia", rect(100, 20, 0)),
                              ("scan", "copepod", rect(50, 10, 90)),
                              ("crop", "daphnia", rect(200, 40, 0)),
                              ("scan", "daphnia", np.array([[0, 0], [1, 1], [2, 2]], np.float32))):
        db.execute_query("INSERT INTO masks (image_name, mask_data, class_name, surface_area) "
                         "VALUES (?, ?, ?, 0)", (image, shape.tobytes(), cls))
    db.conn.close()
    paths = {"scan": str(imgs / "scan.tif"), "crop": str(imgs / "crop.png")}
    return tmp_path, paths


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_export_mm_with_file_dpi_and_pixels_without_scale(project):
    root, paths = project
    r = export_measurements(root / "masks.db", root / "out.csv", paths, sc.ProjectScale(), unit="mm")
    assert (r.rows, r.skipped, r.without_scale) == (3, 1, 1)
    rows = read(root / "out.csv")
    assert list(rows[0]) == columns("mm")
    scan = [x for x in rows if x["image_name"] == "scan"]
    daphnia = next(x for x in scan if x["class_name"] == "daphnia")
    assert float(daphnia["length_mm"]) == pytest.approx(1.0)
    assert float(daphnia["area_mm2"]) == pytest.approx(0.2)
    assert daphnia["scale_source"] == "file DPI" and float(daphnia["um_per_px"]) == pytest.approx(10)
    crop = next(x for x in rows if x["image_name"] == "crop")
    assert crop["length_mm"] == "" and float(crop["length_px"]) == pytest.approx(200)


def test_export_um_project_scale_filters_and_per_class(project):
    root, paths = project
    s = sc.ProjectScale(um_per_px=2.0)
    r = export_measurements(root / "masks.db", root / "res.csv", paths, s, unit="µm",
                            classes=["daphnia"], per_class=True)
    assert [p.name for p in r.files] == ["res_daphnia.csv"]
    rows = read(r.files[0])
    crop = next(x for x in rows if x["image_name"] == "crop")
    assert float(crop["length_µm"]) == pytest.approx(400) and crop["scale_source"] == "project"
    only_scan = export_measurements(root / "masks.db", root / "one.csv", paths, s,
                                    image_names=["scan"])
    assert {x["image_name"] for x in read(root / "one.csv")} == {"scan"} and only_scan.rows == 2


def test_export_cancel_leaves_no_file(project):
    root, paths = project
    with pytest.raises(ExportCancelled):
        export_measurements(root / "masks.db", root / "c.csv", paths, sc.ProjectScale(),
                            cancelled=lambda: True)
    assert not (root / "c.csv").exists()


# --------------------------------------------------------------------------
# Centimetres
# --------------------------------------------------------------------------
def test_cm_unit(project):
    root, paths = project
    assert sc.ProjectScale.from_dict({"unit": "cm"}).unit == "cm"
    r = export_measurements(root / "masks.db", root / "cm.csv", paths, sc.ProjectScale(), unit="cm")
    rows = read(root / "cm.csv")
    assert list(rows[0]) == columns("cm") and "area_cm2" in rows[0]
    daphnia = next(x for x in rows if x["image_name"] == "scan" and x["class_name"] == "daphnia")
    assert float(daphnia["length_cm"]) == pytest.approx(0.1)        # 100 px * 10 µm = 1 mm
    assert float(daphnia["area_cm2"]) == pytest.approx(0.002)       # 0.2 mm²
    assert r.rows == 3


# --------------------------------------------------------------------------
# Logging: the export says what it did, and which images had no scale
# --------------------------------------------------------------------------
@pytest.fixture
def app_log(caplog):
    import logging
    app = logging.getLogger("segmentme")       # does not propagate to the root logger
    app.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="segmentme")
    yield caplog
    app.removeHandler(caplog.handler)


def test_export_logs_a_summary_and_images_without_scale(project, app_log):
    root, paths = project
    export_measurements(root / "masks.db", root / "log.csv", paths, sc.ProjectScale())
    text = app_log.text
    assert "Measuring 4 mask(s) on 2 image(s)" in text
    assert "1 from file DPI, 0 project scale, 1 none" in text
    assert "No scale for 1 image(s), sizes in pixels only: crop" in text
    assert "Measured 3 mask(s)" in text and "1 without area left out" in text


def test_ignored_default_dpi_is_logged(tmp_path, app_log):
    Image.new("RGB", (8, 8)).save(tmp_path / "x.jpg", dpi=(72, 72))
    assert sc.file_um_per_px(tmp_path / "x.jpg") is None
    assert "Ignoring 72 DPI" in app_log.text


def test_read_only_connection_handles_odd_folder_names(tmp_path):
    import sqlite3
    from core.database.connection import connect_read_only
    folder = tmp_path / "week 3 #1 - 26-08-'25 100%"
    folder.mkdir()
    db = folder / "masks.db"
    sqlite3.connect(db).execute("CREATE TABLE t (x)").connection.commit()
    conn = connect_read_only(db)
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone() == (0,)
    import pytest
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO t VALUES (1)")     # read-only
    conn.close()
