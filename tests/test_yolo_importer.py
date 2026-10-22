"""File > Import Annotations: the engine (core/importers/yolo_importer.py)."""
import os

import numpy as np
import pytest

from core.database.connection import DatabaseConnection
from core.importers import yolo_importer as yi
from tests.conftest import make_image

NAMES = {0: "copepod", 1: "daphnia"}
TRI = "0.1 0.1 0.5 0.1 0.3 0.6"          # a triangle in normalised coordinates


# --------------------------------------------------------------------------
# .txt parsing
# --------------------------------------------------------------------------
def parse(tmp_path, text, names=NAMES):
    p = tmp_path / "x.txt"
    p.write_bytes(text.encode() if isinstance(text, str) else text)
    return yi.parse_label_file(p, names)


def test_parse_polygon(tmp_path):
    r = parse(tmp_path, f"0 {TRI}\n1 {TRI}\n")
    assert r.error is None and [c for c, _ in r.polygons] == [0, 1]
    assert np.allclose(r.polygons[0][1], [[0.1, 0.1], [0.5, 0.1], [0.3, 0.6]])


def test_parse_ignores_trailing_confidence(tmp_path):
    r = parse(tmp_path, f"0 {TRI} 0.87\n")
    assert r.error is None and len(r.polygons) == 1 and r.confidence_ignored == 1


def test_parse_windows_line_endings_and_bom(tmp_path):
    r = parse(tmp_path, ("﻿" + f"0 {TRI}\r\n1 {TRI}\r\n").encode("utf-8"))
    assert r.error is None and len(r.polygons) == 2


def test_parse_empty_file_means_no_objects(tmp_path):
    r = parse(tmp_path, "")
    assert r.error is None and r.polygons == []


@pytest.mark.parametrize("line, why", [
    ("0 0.5 0.5 0.2 0.2", "box"),
    ("0 0.1 0.1 0.2", "values"),
    ("7 " + TRI, "class"),
    ("-1 " + TRI, "class"),
    ("0 0.1 abc 0.5 0.1 0.3 0.6", "number"),
    ("0 0.1 0.1 1.5 0.1 0.3 0.6", "outside"),
    ("0 nan 0.1 0.5 0.1 0.3 0.6", "number"),
])
def test_parse_broken_line_skips_the_whole_file(tmp_path, line, why):
    r = parse(tmp_path, f"0 {TRI}\n{line}\n")
    assert r.error and r.polygons == []


def test_parse_binary_file_is_broken(tmp_path):
    assert parse(tmp_path, b"\x00\x01\x02garbage").error


def test_parse_drops_polygons_without_area(tmp_path):
    r = parse(tmp_path, f"0 {TRI}\n0 0.1 0.1 0.2 0.2 0.3 0.3\n")
    assert r.error is None and len(r.polygons) == 1 and r.dropped_empty == 1


def test_parse_drops_repeated_closing_point(tmp_path):
    r = parse(tmp_path, "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.1 0.1\n")
    assert len(r.polygons[0][1]) == 4


# --------------------------------------------------------------------------
# data.yaml and layout
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["names: [copepod, daphnia]\n",
                                  "names:\n  0: copepod\n  1: daphnia\n"])
def test_read_class_names(tmp_path, text):
    y = tmp_path / "data.yaml"
    y.write_text(text, encoding="utf-8")
    assert yi.read_class_names(y) == NAMES


@pytest.mark.parametrize("text", ["nc: 2\n", "names: []\n", "names: {a: x}\n", ": : :\n"])
def test_read_class_names_rejects_bad_yaml(tmp_path, text):
    y = tmp_path / "data.yaml"
    y.write_text(text, encoding="utf-8")
    with pytest.raises(yi.ImportFailed):
        yi.read_class_names(y)


def test_clean_class_name():
    assert yi.clean_class_name("Small fish (adult)", 3) == "Small_fish_adult"
    assert yi.clean_class_name("!!!", 3) == "class_3"


def test_find_yaml(tmp_path):
    root = tmp_path / "ds"
    (root / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text("names: [a]\n")
    assert yi.find_yaml(root) == root / "data.yaml"
    assert yi.find_yaml(root / "labels") == root / "data.yaml"   # labels/ chosen: one level up


def test_find_yaml_ignores_unrelated_yaml_further_up(tmp_path):
    (tmp_path / "data.yaml").write_text("names: [wrong]\n")
    ds = tmp_path / "some" / "annotations"
    ds.mkdir(parents=True)
    assert yi.find_yaml(ds) is None


# --------------------------------------------------------------------------
# Scan + import: annotations for images already in the project
# --------------------------------------------------------------------------
@pytest.fixture
def project(tmp_path):
    """Project images a.png (200x100, has 1 mask), b.jpg, c.png, d.png, E.png, x.png + x.jpg."""
    root = tmp_path / "project"
    (root / ".segmentme").mkdir(parents=True)
    imgs = root / "images"
    make_image(imgs / "a.png", (200, 100))
    for n in ("b.jpg", "c.png", "d.png", "E.png", "x.png", "x.jpg"):
        make_image(imgs / n, (100, 100))
    db = DatabaseConnection(str(root / ".segmentme" / "masks.db"))
    db.execute_query("INSERT INTO classes (class_name, color) VALUES ('Copepod', '#ff0000')")
    tri = np.array([[1, 1], [9, 1], [5, 9]], np.float32).tobytes()
    db.execute_query("INSERT INTO masks (image_name, mask_data, class_name, surface_area) "
                     "VALUES ('a', ?, 'Copepod', 32)", (tri,))
    yield root, db
    db.conn.close()


@pytest.fixture
def labels(tmp_path):
    """Annotations in nested folders, as a YOLO export with splits."""
    root = tmp_path / "export" / "labels"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    (root / "train" / "a.txt").write_text(f"0 {TRI}\n1 {TRI}\n")      # 2 masks, image has 1 already
    (root / "val" / "b.txt").write_text(f"1 {TRI} 0.9\n")             # conf ignored
    (root / "train" / "c.txt").write_text("0 0.5 0.5 0.1 0.1\n")      # box: broken
    (root / "train" / "d.txt").write_text(f"0 {TRI}\n")
    (root / "val" / "d.txt").write_text(f"0 {TRI}\n")                 # d twice: skip both
    (root / "e.txt").write_text(f"0 {TRI}\n")                         # E.png: case differs, no match
    (root / "x.txt").write_text(f"0 {TRI}\n")                         # x.png and x.jpg
    (root / "zzz.txt").write_text(f"0 {TRI}\n")                       # no such image
    (root / "empty.txt").write_text("")
    (root / "classes.txt").write_text("copepod\ndaphnia\n")          # not a label file
    (root / ".hidden").mkdir()
    (root / ".hidden" / "a.txt").write_text(f"0 {TRI}\n")
    (root.parent / "data.yaml").write_text("names: [copepod, daphnia]\n")
    return root


def scan(labels, project_root, db=None):
    images = sorted((project_root / "images").iterdir())
    existing = dict(db.fetch_all("SELECT image_name, COUNT(*) FROM masks GROUP BY image_name")) if db else {}
    return yi.scan_annotations(labels, NAMES, images, existing)


def test_scan_reads_subfolders_and_matches_exact_names(labels, project):
    root, db = project
    s = scan(labels, root, db)
    status = {str(e.label_path.relative_to(labels)).replace(os.sep, "/"): e.status for e in s.entries}
    assert status == {
        "train/a.txt": yi.IMPORT, "val/b.txt": yi.IMPORT, "train/c.txt": yi.IMPORT,
        "train/d.txt": yi.DUPLICATE, "val/d.txt": yi.DUPLICATE,
        "e.txt": yi.NO_IMAGE, "x.txt": yi.AMBIGUOUS, "zzz.txt": yi.NO_IMAGE,
        "empty.txt": yi.NO_IMAGE,
    }
    assert [e.stem for e in s.broken_labels] == ["c"]
    assert sorted(e.stem for e in s.to_import) == ["a", "b"]
    assert s.mask_count == 3 and s.images_with_masks == 2
    assert [e.stem for e in s.with_existing_masks] == ["a"]
    assert s.with_existing_masks[0].existing_masks == 1
    assert s.to_import[0].size == (200, 100)


def test_import_adds_next_to_existing_masks(labels, project):
    root, db = project
    s = scan(labels, root, db)
    before_files = sorted(os.listdir(root / "images"))
    result = yi.run_import(s, {0: "copepod", 1: "daphnia"}, {"Copepod": "#ff0000"}, db)
    assert (result.images, result.masks) == (2, 3)
    assert result.classes_added == ["daphnia"]
    assert result.classes_merged == [("copepod", "Copepod")]
    assert db.fetch_one("SELECT COUNT(*) FROM masks WHERE image_name='a'")[0] == 3   # 1 kept + 2
    assert db.fetch_one("SELECT COUNT(*) FROM masks WHERE image_name='d'")[0] == 0   # duplicate .txt
    rows = db.fetch_all("SELECT mask_data, class_name FROM masks WHERE image_name='a' ORDER BY id")
    pts = np.frombuffer(rows[1][0], np.float32).reshape(-1, 2)
    assert np.allclose(pts, [[20, 10], [100, 10], [60, 60]]) and rows[1][1] == "Copepod"
    assert sorted(os.listdir(root / "images")) == before_files        # no files touched


def test_failed_import_changes_nothing(labels, project, monkeypatch):
    root, db = project
    s = scan(labels, root, db)
    real = db.executemany
    calls = []

    def fail_on_masks(query, rows):
        calls.append(query)
        if "masks" in query:
            raise RuntimeError("database is locked")
        return real(query, rows)
    monkeypatch.setattr(db, "executemany", fail_on_masks)
    with pytest.raises(RuntimeError):
        yi.run_import(s, {0: "copepod", 1: "daphnia"}, {"Copepod": "#ff0000"}, db)
    assert db.fetch_one("SELECT COUNT(*) FROM masks")[0] == 1
    assert db.fetch_all("SELECT class_name FROM classes") == [("Copepod",)]   # new class rolled back


def test_cancel_changes_nothing(labels, project):
    root, db = project
    s = scan(labels, root, db)
    with pytest.raises(yi.ImportCancelled):
        yi.run_import(s, {0: "copepod", 1: "daphnia"}, {}, db,
                      progress=lambda done, total, name: done < 1)
    assert db.fetch_one("SELECT COUNT(*) FROM masks")[0] == 1


def test_scan_can_be_cancelled(labels, project):
    root, _ = project
    with pytest.raises(yi.ImportCancelled):
        yi.scan_annotations(labels, NAMES, sorted((root / "images").iterdir()),
                            progress=lambda done, total: done < 2)


def test_unreadable_project_image_is_reported(labels, project):
    root, db = project
    (root / "images" / "b.jpg").write_bytes(b"not an image")
    s = scan(labels, root, db)
    assert [e.stem for e in s.with_status(yi.UNREADABLE)] == ["b"]


def test_invalid_or_missing_class_target(labels, project):
    root, db = project
    s = scan(labels, root, db)
    with pytest.raises(yi.ImportFailed):
        yi.run_import(s, {0: "copepod"}, {}, db)
    with pytest.raises(yi.ImportFailed):
        yi.run_import(s, {0: "cope pod", 1: "daphnia"}, {}, db)


def test_report_mentions_every_problem(labels, project):
    root, db = project
    s = scan(labels, root, db)
    result = yi.run_import(s, {0: "copepod", 1: "daphnia"}, {"Copepod": "#ff0000"}, db)
    text = yi.describe(s, result)
    for expected in ("Imported 3 mask(s) for 2 image(s).",
                     "Skipped - no project image with this name (3):",
                     "Skipped - same name as another .txt in the folder (2):",
                     "Skipped - more than one project image with this name (1):",
                     "Skipped - broken .txt (1):",
                     "Confidence values ignored in 1 file(s)",
                     "Added next to masks the image already had (1 image(s)):",
                     "project image(s) had no .txt in the folder"):
        assert expected in text, expected


def test_pick_new_colors_are_distinct():
    colors = yi.pick_new_colors(5, ["#ff0000"])
    assert len(set(colors)) == 5 and all(c.startswith("#") and len(c) == 7 for c in colors)


def test_image_size_honours_exif_rotation(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (60, 20))
    exif = img.getexif()
    exif[0x0112] = 6  # rotated 90 degrees
    p = tmp_path / "rot.jpg"
    img.save(p, exif=exif)
    assert yi.image_size(p) == (20, 60)


def test_scan_leaves_pil_settings_as_they_were(labels, project):
    from PIL import Image
    root, db = project
    before = Image.MAX_IMAGE_PIXELS
    scan(labels, root, db)
    assert Image.MAX_IMAGE_PIXELS == before


def test_scan_reads_many_images_in_parallel_with_the_same_result(tmp_path, monkeypatch):
    imgs = [make_image(tmp_path / "images" / f"img_{i:03d}.png", (40 + i, 30)) for i in range(60)]
    lab = tmp_path / "labels"
    lab.mkdir()
    for i in range(60):
        (lab / f"img_{i:03d}.txt").write_text(f"0 {TRI}\n")
    results = []
    for workers in (1, 16):
        monkeypatch.setattr(yi, "_READ_WORKERS", workers)
        s = yi.scan_annotations(lab, NAMES, imgs)
        results.append([(e.stem, e.status, e.size, len(e.label.polygons)) for e in s.entries])
    assert results[0] == results[1]
    assert [r[2] for r in results[1]] == [(40 + i, 30) for i in range(60)]
