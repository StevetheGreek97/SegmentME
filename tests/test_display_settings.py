"""Settings > Display…: mask opacity, outlines only (V), brightness /
contrast, hiding classes. View only, remembered app-wide (hidden classes
not)."""
import json

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from tests.conftest import make_image, wait_for_image
from ui.elements.image_display.mask_render import DEFAULT_STYLE, DisplayStyle, blend_masks

SQUARE = np.array([[20, 20], [60, 20], [60, 60], [20, 60]], np.float32)
RED = (255, 0, 0)


def _image(value=100):
    return np.full((100, 100, 3), value, np.uint8)


def test_default_look_is_unchanged():
    out = blend_masks(_image(), [(1, SQUARE, "a", 0)], {"a": RED})
    assert tuple(out[40, 40]) == (162, 60, 60)          # 40 % red fill, as before
    assert tuple(out[20, 40]) == (0, 0, 0)              # black outline


def test_outlines_only_keeps_the_inside_and_colours_the_edge():
    style = DisplayStyle(outlines_only=True)
    out = blend_masks(_image(), [(1, SQUARE, "a", 0)], {"a": RED}, style=style)
    assert tuple(out[40, 40]) == (100, 100, 100)        # image untouched inside
    assert tuple(out[20, 40]) == RED                    # outline in the class colour
    assert tuple(out[21, 40]) == RED                    # 2 px wide


def test_fill_opacity():
    for percent, expected in ((0, 100), (100, 255), (50, 178)):
        out = blend_masks(_image(), [(1, SQUARE, "a", 0)], {"a": RED},
                          style=DisplayStyle(fill=percent / 100))
        assert abs(int(out[40, 40][0]) - expected) <= 1, percent


def test_hidden_class_is_not_drawn():
    other = SQUARE + 30
    out = blend_masks(_image(), [(1, SQUARE, "a", 0), (2, other, "b", 0)],
                      {"a": RED, "b": (0, 0, 255)}, style=DisplayStyle(hidden=frozenset({"a"})))
    assert tuple(out[30, 30]) == (100, 100, 100)        # only "a" there: not drawn
    assert tuple(out[80, 80])[2] > 100                  # "b" drawn


def test_brightness_and_contrast_are_view_only():
    image = _image(100)
    brighter = DisplayStyle(brightness=50).adjust(image)
    assert brighter[0, 0, 0] > 100 and image[0, 0, 0] == 100      # original untouched
    darker = DisplayStyle(brightness=-50).adjust(image)
    assert darker[0, 0, 0] < 100
    flat = DisplayStyle(contrast=-75).adjust(np.array([[[0, 128, 255]]], np.uint8))
    sharp = DisplayStyle(contrast=100).adjust(np.array([[[60, 128, 196]]], np.uint8))
    assert flat[0, 0, 2] - flat[0, 0, 0] < 100 and sharp[0, 0, 2] - sharp[0, 0, 0] > 200
    assert DEFAULT_STYLE.adjust(image) is not image and np.array_equal(DEFAULT_STYLE.adjust(image), image)


def _photo():
    """A small colourful test picture with shadows and a bright patch."""
    rng = np.random.default_rng(3)
    img = rng.integers(60, 120, (80, 90, 3), dtype=np.uint8)
    img[:, :, 2] //= 2                                   # yellowish background
    img[10:30, 10:40] = (20, 20, 20)                     # dark object in shadow
    img[50:70, 50:80] = (240, 235, 200)                  # glare
    return img


def test_gamma_brightens_mid_tones_only():
    img = np.array([[[0, 128, 255]]], np.uint8)
    out = DisplayStyle(gamma=200).adjust(img)
    assert out[0, 0, 0] == 0 and out[0, 0, 2] == 255 and out[0, 0, 1] > 160


def test_saturation():
    img = _photo()
    grey = DisplayStyle(saturation=-100).adjust(img)
    assert np.all(np.ptp(grey.astype(int), axis=2) <= 2)          # no colour left
    vivid = DisplayStyle(saturation=80).adjust(img)
    assert np.ptp(vivid.astype(int), axis=2).mean() > np.ptp(img.astype(int), axis=2).mean()


def test_auto_contrast_brings_out_dark_areas():
    img = _photo()
    img[10:30, 10:40] = 20 + (np.indices((20, 30)).sum(0) % 6).astype(np.uint8)[..., None]
    out = DisplayStyle(auto_contrast=True).adjust(img)
    assert out[10:30, 10:40].std() > img[10:30, 10:40].std()       # more detail in the shadow


def test_sharpen_adds_edge_contrast():
    img = np.full((40, 40, 3), 100, np.uint8)
    img[:, 20:] = 150
    out = DisplayStyle(sharpen=80).adjust(img)
    assert out[20, 18, 0] < 100 and out[20, 21, 0] > 150           # overshoot at the edge
    assert out[20, 2, 0] == 100                                    # flat areas unchanged


def test_invert_and_channels():
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 10, 20, 200
    assert tuple(DisplayStyle(invert=True).adjust(img)[0, 0]) == (245, 235, 55)
    assert tuple(DisplayStyle(channel="blue").adjust(img)[0, 0]) == (200, 200, 200)
    assert tuple(DisplayStyle(channel="red").adjust(img)[0, 0]) == (10, 10, 10)


def test_adjustments_never_change_the_original():
    img = _photo()
    before = img.copy()
    DisplayStyle(brightness=30, contrast=40, gamma=150, saturation=50, auto_contrast=True,
                 sharpen=70, invert=True, channel="green").adjust(img)
    assert np.array_equal(img, before)


# ---------- in the window ----------
@pytest.fixture
def window(qapp, tmp_path, no_popups):
    from PyQt6.QtGui import QColor
    from services.recent_projects import initialize_project
    from ui.main_window import MainApp
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    make_image(root / "images" / "a.png", (120, 100), color=(100, 100, 100))
    (root / "proj.SEproj").write_text(json.dumps({"name": "proj", "images": ["a.png"]}))
    db = str(root / ".segmentme" / "masks.db")
    win = MainApp(db_path=db)
    win.state_manager.class_manager.add_class("Daphnia", QColor("#ff0000"))
    win.state_manager.class_manager.add_class("Debris", QColor("#0000ff"))
    win.state_manager.mask_manager.save_mask(SQUARE, "a", "Daphnia")
    win.state_manager.mask_manager.save_mask(SQUARE + 40, "a", "Debris")
    initialize_project(win, db)
    wait_for_image(win)
    win.show()
    yield win
    win.display_settings.reset()
    win.close()
    win.state_manager.db.conn.close()
    win.deleteLater()
    qapp.processEvents()


def _shown(win, qapp):
    import time
    end = time.time() + 2
    while time.time() < end:
        qapp.processEvents()
        if not win._display_timer.isActive() and win.image_display._drawn_version == \
                win.image_display._masks_version:
            break
        time.sleep(0.01)
    qapp.processEvents()
    return win.image_display.pixmap_item.array()


def test_v_switches_outlines_only_and_is_remembered(window, qapp):
    from ui.display_settings import DisplaySettings
    QTest.keyClick(window.image_display, Qt.Key.Key_V)
    assert window.display_settings.outlines_only
    picture = _shown(window, qapp)
    assert tuple(picture[40, 40]) == (100, 100, 100)    # no fill
    assert tuple(picture[20, 40]) == RED
    assert DisplaySettings().outlines_only               # remembered for the next start
    QTest.keyClick(window.image_display, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    assert window.display_settings.outlines_only         # Ctrl+V is not V
    QTest.keyClick(window.image_display, Qt.Key.Key_V)
    assert not window.display_settings.outlines_only


def test_hidden_class_cannot_be_clicked_and_is_not_remembered(window, qapp):
    from ui.display_settings import DisplaySettings
    display = window.image_display
    assert display.get_clicked_masks((40, 40))
    display._select_masks(display.get_clicked_masks((40, 40)))
    window.display_settings.set_class_visible("Daphnia", False)
    picture = _shown(window, qapp)
    assert tuple(picture[30, 30]) == (100, 100, 100)
    assert display.get_clicked_masks((40, 40)) == []
    assert display.highlighted_mask_ids == []            # hidden masks leave the selection
    assert window._hidden_classes_note.isVisibleTo(window)
    assert "Daphnia" in window._hidden_classes_note.text()
    assert DisplaySettings().hidden == frozenset()       # shown again at the next start
    window.display_settings.show_all_classes()
    _shown(window, qapp)
    assert display.get_clicked_masks((40, 40))
    assert not window._hidden_classes_note.isVisibleTo(window)


def test_brightness_also_applies_while_masks_are_hidden(window, qapp):
    window.display_settings.set_brightness(60)
    _shown(window, qapp)
    window.image_display.set_peeking(True)               # H held
    plain = window.image_display.pixmap_item.array()
    assert plain[5, 5, 0] > 100                          # brighter, no masks
    assert window.state_manager.current_image[5, 5, 0] == 100   # the image itself untouched
    window.image_display.set_peeking(False)


def test_display_window_controls(window, qapp):
    texts = [a.text() for a in window.menuBar().actions()[3].menu().actions()]
    assert "Display…" in texts
    dialog = window.show_display_settings()
    assert window.show_display_settings() is dialog      # one window
    dialog.fill.setValue(70)
    assert window.display_settings.fill_percent == 70 and dialog.fill_label.text() == "70%"
    dialog.contrast.setValue(25)
    assert window.display_settings.contrast == 25
    dialog.class_checks["Debris"].setChecked(False)
    assert window.display_settings.hidden == {"Debris"}
    window.display_settings.toggle_outlines_only()       # e.g. the V key
    assert dialog.outlines.isChecked() and not dialog.fill.isEnabled()
    window.display_settings.reset()
    assert dialog.fill.value() == 40 and dialog.class_checks["Debris"].isChecked()
    dialog.close()


def test_deleted_class_does_not_stay_hidden(window, qapp):
    window.display_settings.set_class_visible("Debris", False)
    window.state_manager.class_manager.remove_class("Debris")
    assert window.display_settings.hidden == frozenset()


def test_new_image_settings_in_the_window(window, qapp):
    from ui.display_settings import DisplaySettings
    dialog = window.show_display_settings()
    dialog.gamma.setValue(150)
    dialog.saturation.setValue(-40)
    dialog.sharpen.setValue(30)
    dialog.auto_contrast.setChecked(True)
    dialog.invert.setChecked(True)
    dialog.channel.setCurrentIndex(dialog.channel.findData("blue"))
    d = window.display_settings
    assert (d.gamma, d.saturation, d.sharpen, d.auto_contrast, d.invert, d.channel) == \
        (150, -40, 30, True, True, "blue")
    assert dialog.gamma_label.text() == "1.50" and not dialog.saturation.isEnabled()
    picture = _shown(window, qapp)
    expected = d.style().adjust(window.state_manager.current_image)
    assert np.array_equal(picture[90:, :10], expected[90:, :10])   # an area without masks
    again = DisplaySettings()                                      # remembered app-wide
    assert (again.gamma, again.channel, again.auto_contrast) == (150, "blue", True)
    d.reset_image()
    assert (d.gamma, d.sharpen, d.channel, d.invert) == (100, 0, "all", False)
    assert dialog.channel.currentData() == "all" and not dialog.invert.isChecked()
    dialog.close()


def test_image_prepared_in_advance_uses_the_settings(window, qapp, tmp_path):
    """The next image is read and drawn on a worker thread: with the
    adjustments too, and the view adopts that work instead of redoing it."""
    from ui.elements.image_display.image_loader import prepare_image
    d = window.display_settings
    d.set_sharpen(50)
    d.set_auto_contrast(True)
    renderer = window.image_display.renderer
    path = window.state_manager.current_image_path
    prepared = prepare_image(path, window.state_manager.db.db_path, renderer.class_colors(),
                             True, d.style())
    assert prepared["base"] is not None
    assert renderer.prepared_still_valid(prepared)
    expected = renderer.overlay(prepared["image"])
    assert np.array_equal(prepared["picture"], expected)
    d.set_sharpen(10)
    assert not renderer.prepared_still_valid(prepared)            # settings changed meanwhile
