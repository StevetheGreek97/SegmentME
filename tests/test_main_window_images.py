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
    import ui.flows.data_io as mw
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
    import ui.flows.data_io as mw
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
    assert display.pixmap_item.isNull()    # previous image no longer shown

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
    import ui.flows.data_io as mw
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder)))
    seen = {}

    def fake_exec(self):
        seen["opened_empty"] = self._folder is None and not self.import_btn.isEnabled()
        self._choose_folder()  # the user presses Browse… in the dialog
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
    assert seen["opened_empty"]           # no folder picker before the dialog
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


# --------------------------------------------------------------------------
# Copying many / big images: window stays responsive, Cancel, disk space
# --------------------------------------------------------------------------
def _slow_place(monkeypatch, delay=0.15):
    import time
    real = image_files.place_image

    def slow(src, dest, mode, cancelled=None, on_bytes=None):
        time.sleep(delay)
        return real(src, dest, mode, cancelled, on_bytes)
    monkeypatch.setattr(image_files, "place_image", slow)


def _many_originals(tmp_path, n=8):
    return [make_image(tmp_path / "originals" / f"scan_{i}.png") for i in range(n)]


def test_import_keeps_the_window_responsive(window, tmp_path, monkeypatch, no_popups, qapp):
    from PyQt6.QtCore import QTimer
    _slow_place(monkeypatch)
    files = _many_originals(tmp_path)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(50)
    _import(window, files, image_files.COPY, monkeypatch)
    timer.stop()
    assert len(os.listdir(tmp_path / "proj" / "images")) == 9
    assert len(ticks) >= 10        # events kept flowing for the ~1.2 s of copying


def test_import_cancel_removes_what_was_copied(window, tmp_path, monkeypatch, no_popups, qapp):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QProgressDialog
    _slow_place(monkeypatch)
    files = _many_originals(tmp_path)

    def press_cancel():
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QProgressDialog) and w.isVisible():
                w.cancel()
                return
        QTimer.singleShot(50, press_cancel)
    QTimer.singleShot(600, press_cancel)
    _import(window, files, image_files.COPY, monkeypatch)
    assert os.listdir(tmp_path / "proj" / "images") == ["first.png"]
    assert all(f.exists() for f in files)
    assert no_popups[-1][1] == "Import cancelled"
    assert len(window.state_manager.image_paths) == 1


def test_import_refuses_copies_that_do_not_fit(window, tmp_path, monkeypatch, no_popups):
    from collections import namedtuple
    from ui.dialogs import place_progress
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(place_progress.shutil, "disk_usage", lambda p: Usage(10, 10, 1000))
    files = _many_originals(tmp_path, 3)
    _import(window, files, image_files.COPY, monkeypatch)
    assert os.listdir(tmp_path / "proj" / "images") == ["first.png"]
    assert "Link to the original files" in no_popups[-1][2]



def test_import_annotations_opens_the_dialog_first(window, monkeypatch, qapp):
    """No folder picker before the dialog: it opens empty, with a hint, and
    stays empty if Browse… is cancelled."""
    from PyQt6.QtWidgets import QDialog, QFileDialog
    import ui.flows.data_io as mw
    pickers = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: pickers.append(1) or ""))
    seen = {}

    def fake_exec(self):
        seen["pickers_before"] = len(pickers)
        seen["hint"] = self.summary.text()
        self._choose_folder()                   # Browse…, then Cancel in the picker
        qapp.processEvents()
        seen["after"] = (self._folder, self.import_btn.isEnabled())
        return QDialog.DialogCode.Rejected
    monkeypatch.setattr(mw.ImportAnnotationsDialog, "exec", fake_exec)
    window.import_annotations()
    assert seen["pickers_before"] == 0
    assert "Browse" in seen["hint"]
    assert seen["after"] == (None, False)


# --------------------------------------------------------------------------
# Scale, scale bar, measurements in the table / statistics, Save Results
# --------------------------------------------------------------------------
def _scan_window(window, tmp_path, qapp):
    """Add a 2540 DPI image (10 µm per pixel) with two masks and show it."""
    import numpy as np
    window.resize(1200, 800)
    window.show()          # docks and the scale bar only update while visible
    from PIL import Image
    from services.file_handlers import loader
    images = tmp_path / "proj" / "images"
    Image.new("RGB", (600, 400), (200, 160, 60)).save(images / "scan.tif", dpi=(2540, 2540))
    db = window.state_manager.db
    for poly in ([[100, 100], [300, 100], [300, 140], [100, 140]],      # 200 x 40 px
                 [[400, 200], [450, 200], [450, 220], [400, 220]]):     # 50 x 20 px
        db.execute_query("INSERT INTO masks (image_name, mask_data, class_name, surface_area) "
                         "VALUES ('scan', ?, 'daphnia', 0)", (np.array(poly, np.float32).tobytes(),))
    db.execute_query("INSERT OR IGNORE INTO classes (class_name, color) VALUES ('daphnia', '#00aaff')")
    paths = loader(str(images))
    window.state_manager.set_image_paths(paths)
    window.slider.set_image_count(len(paths))
    window.slider._go_to_name("scan.tif")
    _wait(qapp)
    assert window.state_manager.current_image_name == "scan"
    return window


def test_scale_bar_follows_the_image_scale(window, tmp_path, qapp):
    display = window.image_display
    window.show()
    assert not display.scale_bar.isVisible()            # first.png has no resolution
    _scan_window(window, tmp_path, qapp)
    assert display.scale_bar.isVisible()
    assert display.scale_bar.text.endswith(("mm", "µm"))
    old = display.scale_bar.text
    for _ in range(12):                                   # zoom in: shorter bar
        display.scale(1.25, 1.25)
        display._refresh_scale_bar()
    assert display.scale_bar.text != old
    window.menu_bar.scale_bar_action.setChecked(False)   # View > Show Scale Bar
    assert not display.scale_bar.isVisible() and window.project_scale.show_bar is False
    window.menu_bar.scale_bar_action.setChecked(True)
    assert display.scale_bar.isVisible()


def test_table_shows_mm_and_sorts_numbers(window, tmp_path, qapp):
    from PyQt6.QtCore import Qt
    _scan_window(window, tmp_path, qapp)
    window.show_results()
    _wait(qapp, 200)
    dock = window.annotations
    m = dock._model
    headers = [m.headerData(i, Qt.Orientation.Horizontal) for i in range(m.columnCount())]
    assert headers[2:] == ["Area (mm²)", "Class", "Length (mm)", "Width (mm)"]
    lengths = sorted(float(m.data(m.index(r, m.COL_LENGTH))) for r in range(m.rowCount()))
    assert lengths == [pytest.approx(0.5), pytest.approx(2.0)]
    proxy = dock._proxy
    proxy.sort(m.COL_AREA, Qt.SortOrder.AscendingOrder)
    areas = [float(proxy.data(proxy.index(r, m.COL_AREA))) for r in range(proxy.rowCount())]
    assert areas == sorted(areas) and areas[0] == pytest.approx(0.1)   # 50x20 px = 0.1 mm²
    window.slider._go_to_name("first.png")                 # no resolution: pixels
    _wait(qapp)
    assert m.headerData(m.COL_LENGTH, Qt.Orientation.Horizontal) == "Length (px)"


def test_set_scale_by_typing_and_by_ruler(window, tmp_path, qapp, monkeypatch):
    """The real flow, through Actions > Set Scale… (the window must survive
    being hidden while the ruler is drawn: it used to be closed by that)."""
    import json
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDialog
    import ui.dialogs.scale_dialog as sd
    from ui.elements.image_display.ruler_mode import RulerMode
    window.show()
    proj = tmp_path / "proj" / "proj.SEproj"

    # pixels per cm is offered too
    d = window.set_scale()
    d.how.setCurrentText("pixels per cm")
    d.value.setValue(400)                               # 400 px per cm = 25 µm per px
    assert d._um == pytest.approx(25.0)
    d.reject()
    qapp.processEvents()

    # Typed: 2400 DPI for images without their own resolution, then OK
    d = window.set_scale()
    assert d.isVisible()
    d.how.setCurrentText("DPI (pixels per inch)")
    d.value.setValue(2400)
    d.accept()
    qapp.processEvents()
    saved = json.loads(proj.read_text(encoding="utf-8"))["scale"]
    assert saved["um_per_px"] == pytest.approx(25400 / 2400) and saved["source"] == "typed"

    # Ruler: drag 200 px on the image, say it is 5 mm, then OK
    monkeypatch.setattr(sd.KnownLengthDialog, "exec",
                        lambda self: (self.length.setValue(5), self.unit.setCurrentText("mm"),
                                      QDialog.DialogCode.Accepted)[-1])
    d = window.set_scale()
    d._draw()
    qapp.processEvents()
    assert not d.isVisible() and isinstance(window.image_display._edit_mode, RulerMode)
    assert window._scale_dialog is d               # hiding did not close it
    mode = window.image_display._edit_mode
    mode.mouse_press((10, 10), Qt.MouseButton.LeftButton)
    mode.mouse_move((210, 10))
    mode.mouse_release(Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert d.isVisible() and window.image_display._edit_mode is None
    assert d._um == pytest.approx(25.0) and d._source == "ruler"     # 5000 µm / 200 px
    d.accept()
    qapp.processEvents()
    saved = json.loads(proj.read_text(encoding="utf-8"))["scale"]
    assert saved["um_per_px"] == pytest.approx(25.0) and saved["source"] == "ruler"
    assert window.current_image_scale()[0] == pytest.approx(25.0)    # first.png has no DPI
    assert window.image_display.scale_bar.isVisible()

    # Cancel changes nothing
    d = window.set_scale()
    d.value.setValue(1.0)
    d.reject()
    qapp.processEvents()
    assert window.project_scale.um_per_px == pytest.approx(25.0)


def test_ruler_escape_changes_nothing(window, qapp):
    from PyQt6.QtCore import Qt
    from ui.elements.image_display.ruler_mode import RulerMode
    results = []
    window.image_display.start_edit_mode(RulerMode(window.image_display, results.append))
    window.image_display.cancel_edit_mode()                 # Esc
    assert results == [None]
    window.image_display.start_edit_mode(RulerMode(window.image_display, results.append))
    mode = window.image_display._edit_mode
    mode.mouse_press((5, 5), Qt.MouseButton.LeftButton)     # click without dragging
    mode.mouse_release(Qt.MouseButton.LeftButton)
    assert results == [None, None]


def test_save_results_writes_measurements(window, tmp_path, qapp, monkeypatch, no_popups):
    import csv
    from PyQt6.QtWidgets import QDialog
    import ui.main_window as mw
    _scan_window(window, tmp_path, qapp)
    out = tmp_path / "res.csv"

    def accept(self):
        self.path.setText(str(out))
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(mw.SaveResultsDialog, "exec", accept)
    for action in window.menuBar().actions()[0].menu().actions():
        if action.text() == "Save Results…":
            action.trigger()                                # as from the menu
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert sorted(float(r["length_mm"]) for r in rows) == [pytest.approx(0.5), pytest.approx(2.0)]
    assert "Saved 2 mask(s)" in no_popups[-1][2]


def test_sizes_tab_plots_lengths(window, tmp_path, qapp):
    _scan_window(window, tmp_path, qapp)
    window.show_statistics() if hasattr(window, "show_statistics") else None
    stats = window.statistics
    if stats is None:
        from ui.dialogs.instance_count import MaskStatisticsDock
        stats = MaskStatisticsDock(window)
    stats.tabs.setCurrentWidget(stats.sizes_plot_widget)
    qapp.processEvents()
    curves = stats.sizes_plot_widget.getPlotItem().listDataItems()
    assert len(curves) == 1 and "daphnia (n=2" in curves[0].name()
    assert stats.sizes_plot_widget.getPlotItem().getAxis("bottom").labelText == "Length (mm)"


def test_slider_jump_refreshes_the_table_like_next_image(window, tmp_path, qapp):
    """Jumping with the slider / search box must update the annotations
    table (it used to keep showing the previous image's masks)."""
    _scan_window(window, tmp_path, qapp)            # on scan.tif: 2 masks
    window.show_results()
    _wait(qapp, 200)
    assert window.annotations._model.rowCount() == 2
    window.slider._go_to_name("first.png")          # no masks
    _wait(qapp)
    assert window.annotations._model.rowCount() == 0
    window.slider._go_to_name("scan.tif")
    _wait(qapp)
    assert window.annotations._model.rowCount() == 2


def test_scale_bar_stays_in_its_corner_when_panning(window, tmp_path, qapp):
    _scan_window(window, tmp_path, qapp)
    display = window.image_display
    for _ in range(15):                                  # zoom in so there is room to pan
        display.scale(1.25, 1.25)
    display._refresh_scale_bar()
    qapp.processEvents()
    corner = display.scale_bar.pos()
    for dx, dy in ((200, 0), (0, 150), (-90, -40)):      # drag the image around
        h, v = display.horizontalScrollBar(), display.verticalScrollBar()
        h.setValue(h.value() + dx)
        v.setValue(v.value() + dy)
        qapp.processEvents()
        assert display.scale_bar.pos() == corner
    bottom = display.viewport().height() - display.scale_bar.height()
    assert corner.x() == 14 and corner.y() == bottom - 14


# --------------------------------------------------------------------------
# Selection: outlines follow the table's selection; select-all is cheap to draw
# --------------------------------------------------------------------------
def test_click_selection_and_select_all(window, tmp_path, qapp):
    _scan_window(window, tmp_path, qapp)            # scan.tif with 2 masks
    window.show_results()
    _wait(qapp, 200)
    d = window.image_display
    ids = sorted(d._mask_index, key=int)
    a, b = ids
    d._select_masks([a]); qapp.processEvents()
    assert set(d.highlighted_mask_ids) == {a} and d.selection_item.count() == 1
    d._select_masks([b]); qapp.processEvents()          # click a second mask: both
    assert set(d.highlighted_mask_ids) == {a, b} and d.selection_item.count() == 2
    d._select_masks([a]); qapp.processEvents()          # click the first again: off
    assert set(d.highlighted_mask_ids) == {b} and d.selection_item.count() == 1
    window.annotations.table.selectAll(); qapp.processEvents()
    assert set(d.highlighted_mask_ids) == {a, b} and d.selection_item.count() == 2
    window.slider._go_to_name("first.png"); _wait(qapp)  # another image: nothing selected
    assert d.highlighted_mask_ids == [] and d.selection_item.count() == 0


def test_many_selected_outlines_repaint_fast(qapp):
    """5,000 selected outlines on a big image: a repaint must stay well under a second
    (drawn as one 2 px path it took ~4 s)."""
    import time
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF
    from PyQt6.QtGui import QPolygonF
    from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView
    from ui.elements.image_display.selection_outlines import SelectionOutlines
    rng = np.random.default_rng(3)
    polys = []
    for _ in range(5000):
        cx, cy = rng.uniform(0, 7600), rng.uniform(0, 11800)
        t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        polys.append(QPolygonF([QPointF(cx + 30 * np.cos(a), cy + 15 * np.sin(a)) for a in t]))
    scene = QGraphicsScene(QRectF(0, 0, 7610, 11842))
    item = SelectionOutlines()
    item.set_polygons(polys)
    scene.addItem(item)
    view = QGraphicsView(scene)
    view.resize(900, 900)
    view.fitInView(scene.sceneRect())
    view.show()
    qapp.processEvents()
    t = time.perf_counter()
    view.viewport().grab()
    assert time.perf_counter() - t < 1.0
    assert item.count() == 5000
    item.set_polygons([])
    assert item.boundingRect().isEmpty()


# --------------------------------------------------------------------------
# Left/Right/Delete/H: app shortcuts, except while typing or in a dialog
# --------------------------------------------------------------------------
def _key(widget, key, qapp, text=""):
    from PyQt6.QtTest import QTest
    QTest.keyClick(widget, key)
    qapp.processEvents()


def _at(window, index, qapp):
    window.state_manager.image_index = index
    window.image_display.display_image(window.state_manager.image_paths[index])
    qapp.processEvents()


def test_arrows_change_image_from_the_view_and_the_table(many, qapp):
    from PyQt6.QtCore import Qt
    many.show()
    _at(many, 1, qapp)
    _key(many.image_display, Qt.Key.Key_Right, qapp)
    assert many.state_manager.current_image_index == 2
    many.show_results()
    qapp.processEvents()
    _key(many.annotations.table, Qt.Key.Key_Left, qapp)
    assert many.state_manager.current_image_index == 1


def test_typing_in_the_search_box_keeps_its_keys(many, qapp):
    from PyQt6.QtCore import Qt
    many.show()
    _at(many, 1, qapp)
    line = many.slider.image_picker.lineEdit()
    line.setFocus()
    line.setText("abc")
    line.setCursorPosition(3)
    _key(line, Qt.Key.Key_Left, qapp)
    assert line.cursorPosition() == 2                 # the cursor moved ...
    assert many.state_manager.current_image_index == 1   # ... the image did not
    _key(line, Qt.Key.Key_H, qapp)
    assert line.text() == "abhc"                      # "h" is typed, not "hide masks"
    assert not many.image_display._peeking


def test_delete_in_a_text_box_keeps_the_masks(many, qapp, monkeypatch):
    from PyQt6.QtCore import Qt
    many.show()
    deleted = []
    monkeypatch.setattr(many.image_display, "delete_selected_masks", lambda: deleted.append(1))
    line = many.slider.image_picker.lineEdit()
    line.setFocus()
    _key(line, Qt.Key.Key_Delete, qapp)
    assert deleted == []
    _key(many.image_display, Qt.Key.Key_Delete, qapp)
    assert deleted == [1]


# --------------------------------------------------------------------------
# Next / Previous / slider read the image on a worker thread
# --------------------------------------------------------------------------
@pytest.fixture
def sized(window, tmp_path, qapp):
    """Five images, each a different width (so the one shown is recognisable)."""
    from services.file_handlers import loader
    images = tmp_path / "proj" / "images"
    for i in range(5):
        make_image(images / f"img{i}.png", (100 + i, 60))
    paths = loader(str(images))
    window.state_manager.set_image_paths(paths)
    window.slider.set_image_count(len(paths))
    window.show()
    _at(window, 0, qapp)
    return window


def _slow_reads(window, monkeypatch, delay):
    """Make reading an image take `delay` s; returns the list of paths read."""
    import time
    loader = window.image_display._image_loader
    real, reads = loader._read, []

    def slow(path):
        reads.append(os.path.basename(path))
        time.sleep(delay)
        return real(path)
    monkeypatch.setattr(loader, "_read", slow)
    return reads


def _shown_width(window):
    return window.state_manager.current_image.shape[1]


def _expected_width(window):
    name = os.path.basename(window.state_manager.current_image_path)
    return 120 if name == "first.png" else 100 + int(name[3])


def test_next_reads_the_image_without_freezing(sized, qapp, monkeypatch):
    import time
    win = sized
    _slow_reads(win, monkeypatch, 0.5)
    t = time.perf_counter()
    win.next_image()
    assert time.perf_counter() - t < 0.2                    # returned at once
    assert win.image_display.loading()
    _wait(qapp, 250)
    assert win.image_display._loading_label.isVisible()     # "Loading img1.png…"
    assert "Loading" in win.image_display._loading_label.text()
    _wait(qapp, 600)
    assert not win.image_display.loading()
    assert not win.image_display._loading_label.isVisible()
    assert win.state_manager.current_image_index == 1
    assert _shown_width(win) == _expected_width(win)


def test_quick_nexts_show_only_the_last_image(sized, qapp, monkeypatch):
    win = sized
    reads = _slow_reads(win, monkeypatch, 0.3)
    for _ in range(4):
        win.next_image()
        qapp.processEvents()
    _wait(qapp, 1200)
    assert win.state_manager.current_image_index == 4
    assert _shown_width(win) == _expected_width(win)
    assert len(reads) <= 2              # the first one in flight, then the last; none between
    assert reads[-1] == os.path.basename(win.state_manager.current_image_path)


def test_slider_jump_reads_in_the_background(sized, qapp, monkeypatch):
    win = sized
    _slow_reads(win, monkeypatch, 0.3)
    win.slider.slider.setValue(3)
    _wait(qapp, 1000)
    assert win.state_manager.current_image_index == 3
    assert _shown_width(win) == _expected_width(win)


def test_no_drawing_while_loading(sized, qapp, monkeypatch):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    win = sized
    _slow_reads(win, monkeypatch, 0.5)
    calls = []

    from core.tools.base import MaskTool

    class Tool(MaskTool):
        def complete_mask(self):
            calls.append("S")
    monkeypatch.setattr(win.tool_manager, "current_tool", Tool(win.image_display.tool_host))
    win.next_image()
    QTest.keyClick(win.image_display, Qt.Key.Key_S)         # would save onto the old picture
    assert calls == []
    _wait(qapp, 800)
    QTest.keyClick(win.image_display, Qt.Key.Key_S)
    assert calls == ["S"]


def test_unreadable_image_shows_the_missing_message(sized, qapp, monkeypatch):
    win = sized
    _slow_reads(win, monkeypatch, 0.05)
    os.remove(win.state_manager.image_paths[1])
    win.next_image()
    _wait(qapp, 400)
    assert win.image_display.image_missing


def test_prefetcher_reads_neighbours_and_stops_when_moving_on(tmp_path):
    import time
    from ui.elements.image_display.image_loader import Prefetcher
    paths = []
    for i in range(5):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(os.urandom(1000 + i))
        paths.append(str(p))
    pre = Prefetcher()
    pre.around(paths, 2)
    end = time.time() + 5
    while time.time() < end and len(pre._done) < 3:
        time.sleep(0.01)
    assert pre._done == {paths[3], paths[4], paths[1]}
    pre.around(paths, 2)                # already read: nothing queued
    assert pre._queue == []
    # cancelling stops a read between pieces
    big = tmp_path / "big.bin"
    big.write_bytes(b"\0" * (Prefetcher.CHUNK * 3))
    pre._generation += 1
    pre._read_through(str(big), pre._generation - 1, bytearray(1024))
    assert str(big) not in pre._done


# --------------------------------------------------------------------------
# Opening a project / changing classes don't read the scan again
# --------------------------------------------------------------------------
def _count_reads(window, monkeypatch):
    reads = []
    real = window.image_display._read_rgb

    def counting(path):
        reads.append(os.path.basename(path))
        return real(path)
    monkeypatch.setattr(window.image_display, "_read_rgb", counting)
    monkeypatch.setattr(window.image_display._image_loader, "_read", counting)
    return reads


def test_opening_a_project_reads_the_first_image_in_the_background(qapp, tmp_path, no_popups, monkeypatch):
    import time
    from ui.main_window import MainApp
    from services.recent_projects import initialize_project
    from tests.conftest import wait_for_image
    root = tmp_path / "p2"
    (root / ".segmentme").mkdir(parents=True)
    make_image(root / "images" / "a.png", (90, 70))
    (root / "p2.SEproj").write_text(json.dumps({"name": "p2", "images": ["a.png"]}))
    db = str(root / ".segmentme" / "masks.db")
    win = MainApp(db_path=db)
    try:
        loader = win.image_display._image_loader
        real = loader._read
        monkeypatch.setattr(loader, "_read", lambda p: (time.sleep(0.4), real(p))[1])
        t = time.perf_counter()
        initialize_project(win, db)
        assert time.perf_counter() - t < 0.3                 # did not wait for the file
        assert win.image_display.loading()
        wait_for_image(win)
        assert win.state_manager.current_image.shape[:2] == (70, 90)
    finally:
        win.state_manager.db.conn.close()
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_class_changes_redraw_without_reading_the_file(window, qapp, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    win = window
    win.image_display.display_image(win.state_manager.current_image_path)
    reads = _count_reads(win, monkeypatch)
    win.state_manager.db.execute_query("INSERT INTO classes (class_name, color) VALUES ('leaf', '#ff0000')")
    win.state_manager.db.execute_query(
        "INSERT INTO masks (image_name, mask_data, class_name, surface_area) VALUES ('first', ?, 'leaf', 0)",
        (__import__("numpy").array([[5, 5], [60, 5], [60, 50], [5, 50]], "float32").tobytes(),))
    before = win.image_display.cached_image_with_masks.copy()
    win._refresh_after_annotation_import()                # class list changed
    qapp.processEvents()                                  # one redraw for the burst of changes
    after = win.image_display.cached_image_with_masks
    assert (before != after).any()                        # the new mask is drawn ...
    win.sidebar.populate_class_dropdown()
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Yes)
    win.sidebar.class_dropdown.setCurrentIndex(win.sidebar.class_dropdown.findText("leaf"))
    win.sidebar.remove_selected_class()                   # class removed
    qapp.processEvents()
    assert (win.image_display.cached_image_with_masks == before).all()
    assert reads == []                                    # ... and the scan was not read again


# --------------------------------------------------------------------------
# Import Annotations writes on a worker thread
# --------------------------------------------------------------------------
def _dataset(tmp_path):
    ds = tmp_path / "export"
    (ds / "labels").mkdir(parents=True)
    (ds / "labels" / "first.txt").write_text("0 0.1 0.1 0.5 0.1 0.3 0.6\n")
    (ds / "data.yaml").write_text("names: [copepod]\n")
    return ds


def test_import_annotations_runs_off_the_window_thread(window, tmp_path, monkeypatch, no_popups, qapp):
    import threading
    import time
    from PyQt6.QtCore import QTimer
    import ui.flows.data_io as mw
    real, seen = mw.run_import, {}

    def slow_import(*args, **kwargs):
        seen["thread"] = threading.current_thread() is not threading.main_thread()
        result = real(*args, **kwargs)
        time.sleep(0.6)                      # a big database write
        return result
    monkeypatch.setattr(mw, "run_import", slow_import)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(20)
    _run_import_annotations(window, _dataset(tmp_path) / "labels", monkeypatch, qapp)
    timer.stop()
    assert seen["thread"]
    assert len(ticks) > 10                   # the window kept handling events meanwhile
    assert window.state_manager.db.fetch_one("SELECT COUNT(*) FROM masks")[0] == 1


def test_import_annotations_cancel_imports_nothing(window, tmp_path, monkeypatch, no_popups, qapp):
    import time
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QProgressDialog
    import ui.flows.data_io as mw
    from core.importers.yolo_importer import ImportCancelled

    def endless(scan, targets, classes, db, report):
        while report(0, 1, "first.txt"):     # until Cancel is pressed
            time.sleep(0.01)
        raise ImportCancelled()
    monkeypatch.setattr(mw, "run_import", endless)

    def press_cancel():
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QProgressDialog) and w.isVisible():
                w.cancel()
                return
        QTimer.singleShot(50, press_cancel)
    QTimer.singleShot(400, press_cancel)
    _run_import_annotations(window, _dataset(tmp_path) / "labels", monkeypatch, qapp)
    assert window.state_manager.db.fetch_one("SELECT COUNT(*) FROM masks")[0] == 0
    assert window.state_manager.class_manager.get_all_class_names() == []


# --------------------------------------------------------------------------
# Masks are drawn on the worker thread; the view shows the array itself
# --------------------------------------------------------------------------
def _add_square(window, name, x0=5, y0=5, x1=60, y1=40):
    import numpy as np
    db = window.state_manager.db
    if not db.fetch_one("SELECT 1 FROM classes WHERE class_name='leaf'"):
        db.execute_query("INSERT INTO classes (class_name, color) VALUES ('leaf', '#ff0000')")
    db.execute_query("INSERT INTO masks (image_name, mask_data, class_name, surface_area) "
                     "VALUES (?, ?, 'leaf', 0)",
                     (name, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "float32").tobytes()))


def test_masks_are_drawn_on_the_worker_thread(sized, qapp, monkeypatch):
    import threading
    import numpy as np
    from ui.elements.image_display import mask_render
    win = sized
    nxt = os.path.splitext(os.path.basename(win.state_manager.image_paths[1]))[0]
    _add_square(win, nxt)
    threads = []
    real = mask_render.blend_masks

    def spy(*args, **kwargs):
        threads.append(threading.current_thread() is threading.main_thread())
        return real(*args, **kwargs)
    monkeypatch.setattr(mask_render, "blend_masks", spy)
    win.next_image()
    _wait(qapp, 500)
    assert win.state_manager.current_image_name == nxt
    assert threads == [False]                          # drawn once, off the window's thread
    d = win.image_display
    expected = real(win.state_manager.current_image, win.state_manager.mask_manager.load_masks(nxt),
                    d.renderer.class_colors())
    assert np.array_equal(d.cached_image_with_masks, expected)
    assert d.pixmap_item.array() is d.cached_image_with_masks   # shown as is: no extra copy


def test_masks_added_while_loading_are_drawn(sized, qapp, monkeypatch):
    import numpy as np
    win = sized
    nxt = os.path.splitext(os.path.basename(win.state_manager.image_paths[1]))[0]
    win.state_manager.db.execute_query("INSERT INTO classes (class_name, color) VALUES ('leaf', '#ff0000')")
    loader = win.image_display._image_loader
    real = loader._read

    def read_then_add(path):
        import sqlite3
        prepared = real(path)
        # e.g. inference finished meanwhile (another connection: this is a worker thread)
        conn = sqlite3.connect(win.state_manager.db.db_path)
        conn.execute("INSERT INTO masks (image_name, mask_data, class_name, surface_area) VALUES (?, ?, 'leaf', 0)",
                     (nxt, np.array([[30, 30], [50, 30], [50, 50], [30, 50]], "float32").tobytes()))
        conn.commit()
        conn.close()
        return prepared
    monkeypatch.setattr(loader, "_read", read_then_add)
    win.next_image()
    _wait(qapp, 500)
    d = win.image_display
    assert len(d._mask_index) == 1
    expected = d.overlay_base_masks(win.state_manager.current_image)
    assert np.array_equal(d.cached_image_with_masks, expected)


def test_array_item_shares_memory_and_repaints_in_place(qapp):
    import numpy as np
    from ui.elements.image_display.array_image_item import ArrayImageItem
    item = ArrayImageItem()
    assert item.isNull() and item.boundingRect().isEmpty()
    arr = np.zeros((40, 60, 3), np.uint8)
    item.set_array(arr)
    assert item.boundingRect().width() == 60 and item.boundingRect().height() == 40
    arr[10, 20] = (255, 0, 0)                          # changed in place ...
    assert item.image().pixelColor(20, 10).red() == 255   # ... seen without a copy
    item.changed([(20, 10, 21, 11)])
    item.set_array(None)
    assert item.isNull()
