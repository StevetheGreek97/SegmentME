"""Reopening a project returns to the image, tool and class used last."""
import json

import pytest

from tests.conftest import make_image, wait_for_image


def _project(tmp_path):
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    names = ["scan_1.png", "scan_2.png", "scan_10.png"]
    for n in names:
        make_image(root / "images" / n)
    (root / "proj.SEproj").write_text(json.dumps({"name": "proj", "images": names}))
    return root


def _open(root, qapp):
    from services.recent_projects import initialize_project
    from ui.main_window import MainApp
    db = str(root / ".segmentme" / "masks.db")
    win = MainApp(db_path=db)
    initialize_project(win, db)
    wait_for_image(win)
    return win


def _close(win):
    win.remember_place()
    win.state_manager.db.conn.close()
    win.deleteLater()


def test_reopens_at_the_last_image_with_tool_and_class(qapp, tmp_path, no_popups):
    from PyQt6.QtGui import QColor
    root = _project(tmp_path)
    win = _open(root, qapp)
    assert win.state_manager.current_image_index == 0
    win.state_manager.class_manager.add_class("Daphnia", QColor("#e6194b"))
    win.state_manager.class_manager.add_class("Copepod", QColor("#3cb44b"))
    qapp.processEvents()
    win.go_to_image(2)
    wait_for_image(win)
    win.sidebar.class_dropdown.setCurrentText("Copepod")
    win.sidebar.select_tool("intelligent_scissors")
    _close(win)
    saved = json.loads((root / ".segmentme" / "last_place.json").read_text())
    assert saved == {"image": "scan_10.png", "tool": "intelligent_scissors", "class": "Copepod"}

    again = _open(root, qapp)
    try:
        assert again.state_manager.current_image_path.endswith("scan_10.png")
        assert again.slider.slider.value() == 2
        assert again.sidebar.class_dropdown.currentText() == "Copepod"
        import time
        end = time.time() + 3
        while time.time() < end and again.tool_manager.current_tool is None:
            qapp.processEvents()
            time.sleep(0.02)
        assert again.sidebar.intelligent_scissors.isChecked()
    finally:
        _close(again)


def test_opening_does_not_overwrite_the_place_with_the_first_image(qapp, tmp_path, no_popups):
    root = _project(tmp_path)
    (root / ".segmentme" / "last_place.json").write_text(
        json.dumps({"image": "scan_2.png", "tool": None, "class": None}))
    win = _open(root, qapp)
    try:
        assert win.state_manager.current_image_path.endswith("scan_2.png")
        qapp.processEvents()
        win._place_timer.timeout.emit()        # what a pending save would write now
        saved = json.loads((root / ".segmentme" / "last_place.json").read_text())
        assert saved["image"] == "scan_2.png"
    finally:
        _close(win)


@pytest.mark.parametrize("content", [
    json.dumps({"image": "deleted.png", "tool": "manual_mask", "class": "Gone"}),
    "{ broken",
    json.dumps(["not", "a", "dict"]),
])
def test_unknown_or_damaged_place_starts_at_the_first_image(qapp, tmp_path, no_popups, content):
    root = _project(tmp_path)
    (root / ".segmentme" / "last_place.json").write_text(content)
    win = _open(root, qapp)
    try:
        assert win.state_manager.current_image_index == 0
    finally:
        _close(win)


def test_tool_without_its_model_is_not_restored(qapp, tmp_path, no_popups, monkeypatch):
    from services import model_store
    monkeypatch.setattr(model_store, "is_available", lambda spec: False)
    root = _project(tmp_path)
    (root / ".segmentme" / "last_place.json").write_text(
        json.dumps({"image": "scan_1.png", "tool": "sam", "class": None}))
    win = _open(root, qapp)
    try:
        import time
        end = time.time() + 0.6
        while time.time() < end:
            qapp.processEvents()
            time.sleep(0.02)
        assert win.tool_manager.current_tool is None     # no download question at start
        assert no_popups == []
    finally:
        _close(win)
