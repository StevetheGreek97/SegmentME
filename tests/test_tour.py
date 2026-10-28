"""Guided tours: every step points at a real, visible part of the window,
the bubble stays on screen, keys and buttons work, the window can't be
used underneath, and the first tour is offered once."""
import json

import pytest
from PyQt6.QtCore import QPoint, QRect, QSettings, Qt
from PyQt6.QtTest import QTest

from tests.conftest import make_image, wait_for_image


@pytest.fixture
def window(qapp, tmp_path, no_popups):
    from ui.main_window import MainApp
    from services.recent_projects import initialize_project
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    for n in ("a.png", "b.png"):
        make_image(root / "images" / n, (400, 300))
    (root / "proj.SEproj").write_text(json.dumps({"name": "proj", "images": ["a.png", "b.png"]}))
    db = str(root / ".segmentme" / "masks.db")
    win = MainApp(db_path=db)
    initialize_project(win, db)
    win.resize(1280, 800)
    win.show()
    wait_for_image(win)
    yield win
    if win.tour is not None:
        win.tour.close_tour(False)
    win.close()
    win.state_manager.db.conn.close()


@pytest.mark.parametrize("key", ["getting_started", "editing", "table", "inference", "training",
                                 "measuring", "import_export", "safety"])
def test_every_step_points_at_something_real(window, qapp, key):
    from ui import tour
    overlay = tour.start_tour(window, key)
    for i, step in enumerate(overlay.steps):
        overlay.show_step(i)
        for _ in range(3):
            qapp.processEvents()
        if step.target is not None:
            assert overlay.hole is not None, f"{key} step {i + 1} ({step.title}): target not found"
            assert not overlay.hole.isEmpty()
        bubble = overlay.bubble.geometry()
        assert overlay.rect().contains(bubble), f"{key} step {i + 1}: bubble off screen"
        if overlay.hole is not None and overlay.hole.width() < overlay.width() * 0.5:
            assert not bubble.intersects(overlay.hole), f"{key} step {i + 1}: bubble covers target"
        assert step.text and step.title
    assert overlay.next_button.text() == "Finish"
    overlay.next_button.click()
    assert window.tour is None


def test_keys_and_buttons(window, qapp):
    from ui import tour
    overlay = tour.start_tour(window, "getting_started")
    index = window.state_manager.current_image_index
    QTest.keyClick(overlay, Qt.Key.Key_Right)
    QTest.keyClick(overlay, Qt.Key.Key_Return)
    assert overlay.index == 2
    QTest.keyClick(overlay, Qt.Key.Key_Left)
    assert overlay.index == 1
    overlay.back_button.click()
    assert overlay.index == 0 and not overlay.back_button.isEnabled()
    assert window.state_manager.current_image_index == index   # → did not change the image
    QTest.keyClick(overlay, Qt.Key.Key_Escape)
    assert window.tour is None and not tour.tour_running(window)


def test_window_cannot_be_used_under_the_tour(window, qapp):
    from ui import tour
    tour.start_tour(window, "getting_started")
    index = window.state_manager.current_image_index
    button = window.sidebar.next_button
    centre = button.mapTo(window, button.rect().center())
    target = window.childAt(centre)
    assert target is window.tour                     # the overlay is on top
    QTest.mouseClick(window.tour, Qt.MouseButton.LeftButton, pos=centre)
    QTest.keyClick(window, Qt.Key.Key_P)             # tool key
    assert window.state_manager.current_image_index == index
    assert window.tool_manager.current_tool is None


def test_follows_window_resize(window, qapp):
    from ui import tour
    overlay = tour.start_tour(window, "getting_started")
    overlay.show_step(4)                              # the class list
    window.resize(900, 600)
    qapp.processEvents()
    assert overlay.size() == window.size()
    assert overlay.rect().contains(overlay.bubble.geometry())


def test_first_tour_is_offered_once(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from ui import tour
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: next(b for b in self.buttons() if b.text() == "Start Tour"))
    assert tour.offer_first_tour(window, settings) is True
    assert tour.tour_running(window)
    window.tour.close_tour(False)
    assert tour.offer_first_tour(window, settings) is False      # never again
    assert not tour.tour_running(window)


def test_not_now_does_not_start(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from ui import tour
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: next(b for b in self.buttons() if b.text() == "Not Now"))
    assert tour.offer_first_tour(window, settings) is False
    assert not tour.tour_running(window)


def test_help_menu_lists_the_tours(window):
    from ui import tour
    texts = [a.text() for a in window.menu_bar.tours_menu.actions()]
    assert len(texts) == len(tour.TOURS)
    assert texts[0].startswith("Getting Started")
    window.menu_bar.tours_menu.actions()[4].trigger()
    assert window.tour is not None and window.tour.name == "Training Your Own Model"


def test_missing_target_falls_back_to_the_middle(window, qapp):
    from ui.tour import Step, TourOverlay
    overlay = TourOverlay(window, [Step("x", "y", lambda w: None),
                                   Step("x", "y", lambda w: 1 / 0)]).start()
    assert overlay.hole is None
    overlay.show_step(1)                               # a broken target never crashes
    assert overlay.hole is None
    centre = overlay.bubble.geometry().center()
    assert abs(centre.x() - overlay.width() // 2) < 5
    overlay.close_tour(False)


def test_table_tour_opens_the_panels(window, qapp):
    from ui import tour
    assert window.annotations is None
    overlay = tour.start_tour(window, "table")
    overlay.show_step(5)                               # Sizes
    for _ in range(3):
        qapp.processEvents()
    assert window.annotations.isVisible() and window.statistics.isVisible()
    assert window.statistics.tabs.currentIndex() == 2
    assert overlay.hole is not None
