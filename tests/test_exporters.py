"""YOLO and COCO export of a small project: the files written."""
import json

import numpy as np
import pytest
import yaml

from tests.conftest import make_image

pytest.importorskip("sam2", reason="the main window needs SAM2 installed")


@pytest.fixture
def project(qapp, tmp_path, no_popups):
    from ui.main_window import MainApp
    from services.recent_projects import initialize_project
    from tests.conftest import wait_for_image
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    make_image(root / "images" / "a.png", (200, 100))
    make_image(root / "images" / "b.png", (100, 50))
    make_image(root / "images" / "c.png", (80, 80))            # no masks
    (root / "proj.SEproj").write_text(json.dumps({"name": "proj", "images": ["a.png", "b.png", "c.png"]}))
    db = str(root / ".segmentme" / "masks.db")
    w = MainApp(db_path=db)
    initialize_project(w, db)
    wait_for_image(w)
    for name, color in (("leaf", "#ff0000"), ("seed", "#00ff00")):
        w.state_manager.db.execute_query("INSERT INTO classes (class_name, color) VALUES (?, ?)", (name, color))
    mm = w.state_manager.mask_manager
    mm.save_mask(np.array([[20, 10], [120, 10], [120, 60], [20, 60]], np.float32), "a", "leaf")
    mm.save_mask(np.array([[150, 50], [190, 50], [170, 90]], np.float32), "a", "seed")
    mm.save_mask(np.array([[10, 10], [50, 10], [50, 40]], np.float32), "b", "seed")
    qapp.processEvents()
    yield w, root
    w.close()
    qapp.processEvents()
    w.state_manager.db.conn.close()
    w.deleteLater()
    qapp.processEvents()


def _export(w, kind):
    from core.exporters.yolo_exporter import YOLOExporter
    from core.exporters.coco_exporter import COCOExporter
    cls = YOLOExporter if kind == "yolo" else COCOExporter
    exporter = cls(w.state_manager, w)
    exporter.export_all_annotations(70, 20, 10)


def _floats(line):
    parts = line.split()
    return int(parts[0]), np.array([float(v) for v in parts[1:]]).reshape(-1, 2)


def test_yolo_export(project, no_popups):
    w, root = project
    _export(w, "yolo")
    assert not [p for p in no_popups if "Overwrite" in p[1]]          # first export: no question
    labels = root / "labels"
    assert sorted(p.name for p in labels.iterdir()) == ["a.txt", "b.txt"]   # only images with masks
    a = [_floats(l) for l in (labels / "a.txt").read_text().splitlines()]
    assert [c for c, _ in a] == [0, 1]                                       # leaf = 0, seed = 1
    # the outline is written closed: the first point again at the end
    assert np.allclose(a[0][1], [[0.1, 0.1], [0.6, 0.1], [0.6, 0.6], [0.1, 0.6], [0.1, 0.1]], atol=1e-6)
    b = [_floats(l) for l in (labels / "b.txt").read_text().splitlines()]
    assert b[0][0] == 1 and np.allclose(b[0][1][:3], [[0.1, 0.2], [0.5, 0.2], [0.5, 0.8]], atol=1e-6)
    data = yaml.safe_load((root / "data.yaml").read_text())
    assert data["nc"] == 2 and data["names"] == {0: "leaf", 1: "seed"}
    split = set()
    for part in ("train", "val", "test"):
        f = root / f"autosplit_{part}.txt"
        if f.exists():
            split |= {line.strip().split("/")[-1] for line in f.read_text().splitlines()}
    assert split == {"a.png", "b.png", "c.png"}


def test_coco_export(project):
    w, root = project
    _export(w, "coco")
    docs = [json.loads(p.read_text()) for p in sorted((root / "labels_coco").glob("instances_*.json"))]
    images = {i["file_name"]: i for d in docs for i in d["images"]}
    assert set(images) == {"a.png", "b.png", "c.png"}
    assert (images["a.png"]["width"], images["a.png"]["height"]) == (200, 100)
    anns = [a for d in docs for a in d["annotations"]]
    assert len(anns) == 3
    cats = {c["name"]: c["id"] for c in docs[0]["categories"]}
    assert cats == {"leaf": 1, "seed": 2}
    by_image = {}
    for d in docs:
        names = {i["id"]: i["file_name"] for i in d["images"]}
        for a in d["annotations"]:
            by_image.setdefault(names[a["image_id"]], []).append(a["category_id"])
    assert sorted(by_image["a.png"]) == [1, 2] and by_image["b.png"] == [2]


@pytest.mark.parametrize("kind, folder", [("yolo", "labels"), ("coco", "labels_coco")])
def test_second_export_asks_before_replacing(project, monkeypatch, kind, folder):
    from PyQt6.QtWidgets import QMessageBox
    w, root = project
    _export(w, kind)
    asked = []
    for answer, expect_kept in ((QMessageBox.StandardButton.No, True), (QMessageBox.StandardButton.Yes, False)):
        marker = root / folder / "mine.txt"
        marker.write_text("keep?")
        monkeypatch.setattr(QMessageBox, "warning",
                            staticmethod(lambda *a, answer=answer, **k: asked.append(a[1]) or answer))
        _export(w, kind)
        assert marker.exists() == expect_kept
    assert len(asked) == 2 and all("Overwrite" in t for t in asked)
