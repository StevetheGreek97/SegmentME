"""Drawing tools driven the way a user drives them: mouse clicks and keys on
the image view of a real window, saving into the project database."""
import json

import numpy as np
import pytest


pytest.importorskip("sam2", reason="the main window needs SAM2 installed")


@pytest.fixture(autouse=True)
def no_errors(monkeypatch):
    """An error inside a mouse or key handler would close the real app
    (PyQt6 aborts on unhandled exceptions): fail the test on any."""
    import sys
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    yield
    assert not errors, f"error in an event handler: {errors[0][1]!r}"


@pytest.fixture
def win(qapp, tmp_path, no_popups):
    from PIL import Image, ImageDraw
    from ui.main_window import MainApp
    from services.recent_projects import initialize_project
    from tests.conftest import wait_for_image
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    (root / "images").mkdir()
    # a dark rectangle on a light background: clear edges for the scissors
    for name, box in (("a.png", (40, 30, 160, 110)), ("b.png", (20, 20, 90, 60))):
        im = Image.new("RGB", (200, 140), (230, 230, 230))
        ImageDraw.Draw(im).rectangle(box, fill=(20, 20, 20))
        im.save(root / "images" / name)
    (root / "proj.SEproj").write_text(json.dumps({"name": "proj", "images": ["a.png", "b.png"]}))
    db = str(root / ".segmentme" / "masks.db")
    w = MainApp(db_path=db)
    w.resize(900, 700)
    w.show()
    initialize_project(w, db)
    wait_for_image(w)
    w.image_display.fit_to_view()
    qapp.processEvents()
    yield w
    w.tool_manager.disable_tools()
    w.close()
    qapp.processEvents()             # pending redraws run while the database is open
    w.state_manager.db.conn.close()
    w.deleteLater()
    qapp.processEvents()


def add_class(w, name="leaf", color="#ff0000"):
    w.state_manager.db.execute_query("INSERT INTO classes (class_name, color) VALUES (?, ?)", (name, color))
    w.sidebar.populate_class_dropdown()
    w.sidebar.class_dropdown.setCurrentIndex(w.sidebar.class_dropdown.findText(name))


def view_pos(w, x, y):
    from PyQt6.QtCore import QPointF
    return w.image_display.mapFromScene(QPointF(x + 0.5, y + 0.5))


def click(w, x, y, button=None, ctrl=False):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    mods = Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier
    QTest.mouseClick(w.image_display.viewport(), button or Qt.MouseButton.LeftButton, mods, view_pos(w, x, y))


def key(w, k):
    from PyQt6.QtTest import QTest
    QTest.keyClick(w.image_display, k)


def saved(w):
    return [(name, np.frombuffer(data, np.float32).reshape(-1, 2), cls)
            for name, data, cls in w.state_manager.db.fetch_all(
                "SELECT image_name, mask_data, class_name FROM masks ORDER BY id")]


def near(polygon, xy, tol=3):
    return np.abs(np.asarray(polygon, float) - np.asarray(xy, float)).max() <= tol


# ------------------------------------------------------------------ manual
def test_manual_tool_draws_and_saves_a_polygon(win, qapp):
    from PyQt6.QtCore import Qt
    add_class(win)
    assert win.tool_manager.enable_tool("manual_mask")
    for x, y in ((50, 40), (150, 40), (150, 100), (50, 100)):
        click(win, x, y)
    key(win, Qt.Key.Key_S)
    [(name, poly, cls)] = saved(win)
    assert name == "a" and cls == "leaf"
    assert near(poly, [(50, 40), (150, 40), (150, 100), (50, 100)], tol=1)
    assert win.image_display.scene.items().__len__() == len(_base_items(win))   # points cleared


def _base_items(win):
    d = win.image_display
    return [i for i in d.scene.items() if i in (d.pixmap_item, d.highlight_item, d.selection_item)]


def test_manual_right_click_removes_the_last_point(win, qapp):
    from PyQt6.QtCore import Qt
    add_class(win)
    win.tool_manager.enable_tool("manual_mask")
    for x, y in ((50, 40), (150, 40), (150, 100), (60, 120)):
        click(win, x, y)
    click(win, 60, 120, Qt.MouseButton.RightButton)          # undo the last point
    click(win, 50, 100)
    key(win, Qt.Key.Key_S)
    [(_n, poly, _c)] = saved(win)
    assert len(poly) == 4 and near(poly[-1], (50, 100), tol=1)


def test_manual_needs_three_points(win, qapp):
    from PyQt6.QtCore import Qt
    add_class(win)
    win.tool_manager.enable_tool("manual_mask")
    click(win, 50, 40)
    click(win, 150, 40)
    key(win, Qt.Key.Key_S)
    assert saved(win) == []


def test_saving_without_a_class_saves_nothing(win, qapp, monkeypatch):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a[1])))
    win.tool_manager.enable_tool("manual_mask")
    for x, y in ((50, 40), (150, 40), (150, 100)):
        click(win, x, y)
    key(win, Qt.Key.Key_S)
    assert saved(win) == [] and warned


# ------------------------------------------------------------------ scissors
def test_scissors_follow_the_edges_and_save(win, qapp):
    from PyQt6.QtCore import Qt
    add_class(win)
    assert win.tool_manager.enable_tool("intelligent_scissors")
    for x, y in ((40, 30), (100, 30), (160, 30), (160, 70), (160, 110), (100, 110), (40, 110), (40, 70), (40, 30)):
        click(win, x, y)
    key(win, Qt.Key.Key_S)
    [(name, poly, cls)] = saved(win)
    assert name == "a" and cls == "leaf" and len(poly) > 20
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    assert near((x0, y0, x1, y1), (40, 30, 160, 110), tol=4)   # along the rectangle's edges


def test_tool_points_are_dropped_on_next_image(win, qapp):
    from PyQt6.QtCore import Qt
    from tests.conftest import wait_for_image
    add_class(win)
    win.tool_manager.enable_tool("manual_mask")
    for x, y in ((50, 40), (150, 40), (150, 100)):
        click(win, x, y)
    win.next_image()
    wait_for_image(win)
    key(win, Qt.Key.Key_S)                                   # nothing left to save
    assert saved(win) == []


def test_scissors_use_the_new_image_after_next(win, qapp):
    """b.png's rectangle is elsewhere: the path must follow b's edges, not a's."""
    from PyQt6.QtCore import Qt
    from tests.conftest import wait_for_image
    add_class(win)
    win.tool_manager.enable_tool("intelligent_scissors")
    win.next_image()
    wait_for_image(win)
    click(win, 20, 20)                                     # the first seed sets the tool up
    tool = win.tool_manager.current_tool
    fresh = type(tool)(tool.host if hasattr(tool, "host") else tool.parent)
    fresh.set_image(win.state_manager.current_image)
    assert np.array_equal(tool.edge_weights, fresh.edge_weights)   # edges of b, not a
    tool.clear_temp_items()
    for x, y in ((20, 20), (55, 20), (90, 20), (90, 60), (55, 60), (20, 60), (20, 20)):
        click(win, x, y)
    key(win, Qt.Key.Key_S)
    [(name, poly, _cls)] = saved(win)
    assert name == "b"
    assert near((*poly.min(0), *poly.max(0)), (20, 20, 90, 60), tol=4)


def test_scissors_without_points_save_nothing(win, qapp):
    from PyQt6.QtCore import Qt
    add_class(win)
    win.tool_manager.enable_tool("intelligent_scissors")
    key(win, Qt.Key.Key_S)
    assert saved(win) == []


# ------------------------------------------------------------------ base tool, fake host
class FakeHost:
    def __init__(self, class_name="leaf"):
        from PyQt6.QtWidgets import QGraphicsScene
        self.scene = QGraphicsScene()
        self.class_name = class_name
        self.saved = []

    def image(self):
        return np.zeros((50, 50, 3), np.uint8)

    def image_name(self):
        return "img"

    def choose_class(self):
        return self.class_name

    def save_mask(self, polygon, image_name, class_name):
        self.saved.append((polygon, image_name, class_name))


def test_base_tool_saves_through_the_host(qapp):
    from core.tools.manual_mask import ManualMask
    host = FakeHost()
    tool = ManualMask(host)
    added = []
    tool.mask_added.connect(lambda name, poly: added.append(name))
    for p in ((1, 1), (10, 1), (10, 10)):
        tool.left_click(p)
    tool.complete_mask()
    [(poly, name, cls)] = host.saved
    assert name == "img" and cls == "leaf" and poly.dtype == np.float32 and poly.shape == (3, 2)
    assert added == ["img"] and tool.current_polygon() is None and not host.scene.items()


def test_base_tool_without_a_class_saves_nothing(qapp):
    from core.tools.manual_mask import ManualMask
    host = FakeHost(class_name=None)
    tool = ManualMask(host)
    for p in ((1, 1), (10, 1), (10, 10)):
        tool.left_click(p)
    tool.complete_mask()
    assert host.saved == []


def test_view_passes_clicks_moves_and_releases_to_the_tool(win, qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from core.tools.base import MaskTool
    events = []

    class Recorder(MaskTool):
        def left_click(self, point, ctrl=False):
            events.append(("left", point, ctrl))

        def right_click(self, point):
            events.append(("right", point))

        def mouse_move(self, point):
            events.append(("move", point))

        def left_release(self, point):
            events.append(("release", point))
    win.tool_manager.current_tool = Recorder(win.image_display.tool_host)
    vp = win.image_display.viewport()
    click(win, 50, 40)
    click(win, 60, 50, ctrl=True)
    click(win, 70, 60, Qt.MouseButton.RightButton)
    QTest.mouseMove(vp, view_pos(win, 80, 70))
    kinds = [e[0] for e in events]
    assert ("left", (50, 40), False) in events and ("left", (60, 50), True) in events
    assert ("right", (70, 60)) in events and "release" in kinds and "move" in kinds
    win.tool_manager.current_tool = None


# ------------------------------------------------------------------ widgets follow the state's signals
def test_many_saved_masks_give_one_redraw(win, qapp, monkeypatch):
    """Inference saves masks one by one: the image is redrawn once."""
    add_class(win)
    d = win.image_display
    calls = []
    real = d.refresh_masks
    monkeypatch.setattr(d, "refresh_masks", lambda: (calls.append(1), real())[1])
    square = np.array([[50, 40], [90, 40], [90, 80], [50, 80]], np.float32)
    for i in range(30):
        win.state_manager.mask_manager.save_mask(square + i, "a", "leaf")
    assert calls == []                                   # nothing yet: coalesced
    qapp.processEvents()
    assert calls == [1]
    assert len(d._mask_index) == 30


def test_class_list_follows_class_changes_and_keeps_the_choice(win, qapp):
    from PyQt6.QtGui import QColor
    add_class(win, "leaf")
    add_class(win, "seed", "#00ff00")
    win.sidebar.class_dropdown.setCurrentIndex(win.sidebar.class_dropdown.findText("seed"))
    win.state_manager.class_manager.add_class("worm", QColor("#0000ff"))   # e.g. from inference
    items = [win.sidebar.class_dropdown.itemText(i) for i in range(win.sidebar.class_dropdown.count())]
    assert "worm" in items
    assert win.sidebar.class_dropdown.currentText() == "seed"


def test_assigning_a_class_from_the_menu_redraws_the_image(win, qapp):
    from ui.custom_components.mask_context_menu import MaskContextMenu
    add_class(win, "leaf", "#ff0000")
    add_class(win, "seed", "#00ff00")
    win.state_manager.mask_manager.save_mask(
        np.array([[50, 40], [90, 40], [90, 80], [50, 80]], np.float32), "a", "leaf")
    qapp.processEvents()
    d = win.image_display
    mask_id = int(next(iter(d._mask_index)))
    before = d.cached_image_with_masks[60, 70].copy()
    MaskContextMenu(win, [("a", mask_id)], parent=d)._assign_class("seed")
    qapp.processEvents()
    after = d.cached_image_with_masks[60, 70]
    assert after[1] > before[1] and after[0] < before[0]   # red tint became green
    assert saved(win)[0][2] == "seed"


def test_app_works_without_the_dextr_package(win, qapp, monkeypatch, no_popups):
    """DEXTR is installed separately: without it, only the DEXTR tool is
    unavailable (with a message), not the whole window."""
    import sys
    import core.managers.tool_manager as tm
    for name in [m for m in sys.modules if m == "DEXTR" or m.startswith("DEXTR.")]:
        monkeypatch.delitem(sys.modules, name)                       # loaded by earlier tests
    monkeypatch.setitem(sys.modules, "DEXTR", None)                 # "not installed"
    monkeypatch.delitem(sys.modules, "core.tools.dextr_mask", raising=False)
    monkeypatch.setattr(tm, "ensure_model_available", lambda *a, **k: True)
    assert win.tool_manager.enable_tool("dextr") is False
    assert any("could not be started" in p[2] for p in no_popups)
    assert win.tool_manager.enable_tool("manual_mask")              # other tools still work
