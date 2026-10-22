"""Main window: Import Images (copy / link), a linked image whose original is
gone, and deleting a project with linked images."""
import json
import os

import pytest

from services import image_files
from tests.conftest import make_image

pytest.importorskip("sam2", reason="the main window needs SAM2 installed")


def _linked(path, original) -> bool:
    return os.path.samefile(path, original)


@pytest.fixture
def originals(tmp_path):
    return [make_image(tmp_path / "originals" / n) for n in ("one.png", "two.jpg")]


@pytest.fixture
def window(qapp, tmp_path, no_popups):
    from ui.main_window import MainApp
    from services.recent_projects import initialize_project
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    make_image(root / "images" / "first.png", (120, 80))
    (root / "proj.SEproj").write_text(json.dumps({"name": "proj", "images": ["first.png"]}))
    db = str(root / ".segmentme" / "masks.db")
    win = MainApp(db_path=db)
    initialize_project(win, db)
    qapp.processEvents()
    yield win
    win.state_manager.db.conn.close()
    win.close()
    win.deleteLater()
    qapp.processEvents()


def _import(win, files, mode, monkeypatch):
    from PyQt6.QtWidgets import QDialog, QFileDialog
    import ui.main_window as mw
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([str(f) for f in files], "")))

    class Choose(mw.ImageModeDialog):
        def exec(self):
            self.choice.set_mode(mode)
            return QDialog.DialogCode.Accepted
    monkeypatch.setattr(mw, "ImageModeDialog", Choose)
    win.import_images()


@pytest.mark.parametrize("mode", [image_files.COPY, image_files.LINK])
def test_import_images(window, tmp_path, originals, monkeypatch, no_popups, mode):
    _import(window, originals, mode, monkeypatch)
    images = tmp_path / "proj" / "images"
    assert sorted(os.listdir(images)) == ["first.png", "one.png", "two.jpg"]
    for f in originals:
        assert _linked(images / f.name, f) == (mode == image_files.LINK)
    assert len(window.state_manager.image_paths) == 3
    seproj = json.loads((tmp_path / "proj" / "proj.SEproj").read_text(encoding="utf-8"))
    assert seproj["images"] == ["first.png", "one.png", "two.jpg"]
    assert "Imported 2 images" in no_popups[-1][2]


def test_import_images_cancel_does_nothing(window, tmp_path, originals, monkeypatch):
    from PyQt6.QtWidgets import QDialog, QFileDialog
    import ui.main_window as mw
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([str(f) for f in originals], "")))
    monkeypatch.setattr(mw.ImageModeDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    window.import_images()
    assert os.listdir(tmp_path / "proj" / "images") == ["first.png"]


def test_missing_original_shows_a_message_and_keeps_masks(window, tmp_path, originals, monkeypatch):
    from tests.conftest import can_symlink
    if not can_symlink(tmp_path):
        pytest.skip("symbolic links not allowed here")
    images = tmp_path / "proj" / "images"
    os.symlink(originals[0], images / "one.png")
    window.state_manager.set_image_paths(sorted(str(p) for p in images.iterdir()))
    db = window.state_manager.db
    import numpy as np
    poly = np.array([[1, 1], [20, 1], [10, 15]], np.float32).tobytes()
    db.execute_query("INSERT INTO masks (image_name, mask_data, class_name, surface_area) "
                     "VALUES ('one', ?, 'c', 1)", (poly,))
    originals[0].unlink()                      # the original goes away

    display = window.image_display
    index = window.state_manager.image_paths.index(str(images / "one.png"))
    window.state_manager.image_index = index
    display.display_image(str(images / "one.png"))
    assert display.image_missing
    assert "original of this linked image is missing" in display._missing_item.toPlainText()
    assert window.state_manager.current_image is None
    assert display.pixmap_item.pixmap().isNull()    # previous image no longer shown

    # Re-opening the project keeps the image in the list and its masks
    from services.recent_projects import _sync_seproj_images
    from services.file_handlers import loader
    assert _sync_seproj_images(window, str(tmp_path / "proj"), loader(str(images)))
    assert db.fetch_one("SELECT COUNT(*) FROM masks WHERE image_name='one'")[0] == 1

    # A readable image clears the message again
    display.display_image(str(images / "first.png"))
    assert not display.image_missing and window.state_manager.current_image is not None


def test_deleting_a_linked_project_keeps_the_originals(window, tmp_path, originals, monkeypatch):
    from services import recent_projects
    _import(window, originals, image_files.LINK, monkeypatch)
    window.state_manager.db.conn.close()
    monkeypatch.setattr(recent_projects, "remove_recent_project", lambda db_path: None)
    assert recent_projects.delete_project_and_remove_recent(
        str(tmp_path / "proj" / ".segmentme" / "masks.db"))
    assert not (tmp_path / "proj").exists()
    assert all(f.exists() and f.stat().st_size > 0 for f in originals)


def _menu_action(window, text):
    for action in window.menuBar().actions()[0].menu().actions():
        if action.text() == text:
            return action
    raise AssertionError(f"no File menu entry {text!r}")


def _run_import_annotations(window, folder, monkeypatch, qapp, review=None, via_menu=False):
    import time
    from PyQt6.QtWidgets import QDialog, QFileDialog
    import ui.main_window as mw
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder)))
    seen = {}

    def fake_exec(self):
        t = time.time()
        qapp.processEvents()   # the first scan starts once the window is up
        while self.scan_result is None and self._yaml_path is not None and time.time() - t < 30:
            qapp.processEvents()
        seen["summary"] = self.summary.text()
        seen["import_enabled"] = self.import_btn.isEnabled()
        if review:
            review(self)
        return QDialog.DialogCode.Accepted if self.import_btn.isEnabled() else QDialog.DialogCode.Rejected
    monkeypatch.setattr(mw.ImportAnnotationsDialog, "exec", fake_exec)
    reports = []
    monkeypatch.setattr(mw.ImportReportDialog, "exec",
                        lambda self: reports.append(self.text.toPlainText()) or 1)
    if via_menu:   # as the user does it: the action passes its "checked" flag along
        _menu_action(window, "Import Annotations…").trigger()
    else:
        window.import_annotations()
    return seen, reports


@pytest.mark.parametrize("via_menu", [False, True])
def test_import_annotations(window, tmp_path, monkeypatch, no_popups, qapp, via_menu):
    import numpy as np
    ds = tmp_path / "export"
    (ds / "labels" / "train").mkdir(parents=True)
    (ds / "labels" / "train" / "first.txt").write_text("0 0.1 0.1 0.5 0.1 0.3 0.6\n")
    (ds / "labels" / "other.txt").write_text("0 0.1 0.1 0.5 0.1 0.3 0.6\n")
    (ds / "data.yaml").write_text("names: [copepod]\n")
    images_before = sorted(os.listdir(tmp_path / "proj" / "images"))

    seen, reports = _run_import_annotations(window, ds / "labels", monkeypatch, qapp,
                                            via_menu=via_menu)
    assert seen["import_enabled"]
    assert "<b>1</b> mask(s) will be added to <b>1</b> project image(s)" in seen["summary"]
    db = window.state_manager.db
    poly = np.frombuffer(db.fetch_one("SELECT mask_data FROM masks WHERE image_name='first'")[0],
                         np.float32).reshape(-1, 2)
    assert np.allclose(poly, [[12, 8], [60, 8], [36, 48]])        # first.png is 120 x 80
    assert window.state_manager.class_manager.get_all_class_names() == ["copepod"]
    assert sorted(os.listdir(tmp_path / "proj" / "images")) == images_before   # no images added
    assert reports and "Imported 1 mask(s) for 1 image(s)." in reports[0]
    assert "Skipped - no project image with this name (1):" in reports[0]


def test_import_annotations_needs_data_yaml(window, tmp_path, monkeypatch, no_popups, qapp):
    ds = tmp_path / "noyaml"
    ds.mkdir()
    (ds / "first.txt").write_text("0 0.1 0.1 0.5 0.1 0.3 0.6\n")
    seen, reports = _run_import_annotations(window, ds, monkeypatch, qapp)
    assert not seen.get("import_enabled", False) and reports == []
    assert window.state_manager.db.fetch_one("SELECT COUNT(*) FROM masks")[0] == 0


def test_menu_has_import_images_and_import_annotations(window):
    texts = [a.text() for a in window.menuBar().actions()[0].menu().actions()]
    assert "Import Images" in texts and "Import Annotations…" in texts
    assert not any("with Annotations" in t for t in texts)


# --------------------------------------------------------------------------
# Image picker (searchable dropdown above the slider)
# --------------------------------------------------------------------------
def _wait(qapp, ms=400):
    import time
    end = time.time() + ms / 1000
    while time.time() < end:
        qapp.processEvents()


@pytest.fixture
def many(window, tmp_path, qapp):
    from services.file_handlers import loader
    images = tmp_path / "proj" / "images"
    for name in ("250812_F1.png", "250812_F2.png", "251014_H4.png", "other.png"):
        make_image(images / name)
    paths = loader(str(images))
    window.state_manager.set_image_paths(paths)
    window.slider.set_image_count(len(paths))
    qapp.processEvents()
    return window


def test_picker_lists_images_in_slider_order(many):
    picker = many.slider.image_picker
    names = [os.path.basename(p) for p in many.state_manager.image_paths]
    assert [picker.itemText(i) for i in range(picker.count())] == names
    assert many.slider.label.text() == f"{many.slider.slider.value() + 1} / {len(names)}"
    assert picker.currentText() == os.path.basename(many.state_manager.current_image_path)


def test_search_filters_by_any_part_of_the_name(many):
    completer = many.slider.image_picker.completer()
    completer.setCompletionPrefix("2508")
    assert completer.completionCount() == 2
    completer.setCompletionPrefix("h4")          # case does not matter
    assert completer.completionCount() == 1
    assert completer.currentCompletion() == "251014_H4.png"


def test_picking_a_name_shows_that_image(many, qapp):
    many.slider._go_to_name("251014_H4.png")
    _wait(qapp)
    assert os.path.basename(many.state_manager.current_image_path) == "251014_H4.png"
    assert many.slider.image_picker.currentText() == "251014_H4.png"
    index = many.state_manager.image_paths.index(many.state_manager.current_image_path)
    assert many.slider.slider.value() == index
    assert many.state_manager.current_image is not None


def test_text_that_is_no_image_name_is_put_back(many, qapp):
    picker = many.slider.image_picker
    current = picker.currentText()
    picker.setEditText("no such image")
    picker.lineEdit().editingFinished.emit()
    assert picker.currentText() == current


def test_picker_follows_next_image_and_imports(many, qapp):
    many.next_image()
    qapp.processEvents()
    assert many.slider.image_picker.currentText() == os.path.basename(many.state_manager.current_image_path)
    make_image(many.state_manager.image_paths[0].rsplit(os.sep, 1)[0] + os.sep + "zzz_new.png")
    from services.file_handlers import loader
    paths = loader(os.path.dirname(many.state_manager.image_paths[0]))
    many.state_manager.set_image_paths(paths)
    many.slider.set_image_count(len(paths))
    assert many.slider.image_picker.findText("zzz_new.png") >= 0


def test_names_are_rebuilt_only_when_the_image_list_changes(many, qapp):
    resets = []
    many.slider._names.modelReset.connect(lambda: resets.append(1))
    for i in range(len(many.state_manager.image_paths)):
        many.slider.slider.setValue(i)
    many.next_image()
    assert resets == []
    paths = list(many.state_manager.image_paths)   # a new list, as after an import
    many.state_manager.set_image_paths(paths)
    many.slider.set_image_count(len(paths))
    assert resets == [1]
