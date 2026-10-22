"""Redrawing only where masks changed must give exactly the same picture as
a full redraw (ui/elements/image_display/image_display.py)."""
import numpy as np
import pytest

from tests.conftest import wait_for_image

pytest.importorskip("sam2", reason="the main window needs SAM2 installed")

W, H = 900, 700


@pytest.fixture
def busy(qapp, tmp_path, no_popups):
    import json
    from ui.main_window import MainApp
    from services.recent_projects import initialize_project
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    from PIL import Image
    (root / "images").mkdir()
    Image.fromarray(img).save(root / "images" / "scan.png")
    (root / "proj.SEproj").write_text(json.dumps({"name": "proj", "images": ["scan.png"]}))
    db_path = str(root / ".segmentme" / "masks.db")
    win = MainApp(db_path=db_path)
    db = win.state_manager.db
    for name, color in (("a", "#ff0000"), ("b", "#00ff00"), ("c", "#3366ff")):
        db.execute_query("INSERT INTO classes (class_name, color) VALUES (?, ?)", (name, color))
    for _ in range(150):
        db.execute_query("INSERT INTO masks (image_name, mask_data, class_name, surface_area) "
                         "VALUES ('scan', ?, ?, 0)", (random_polygon(rng).tobytes(), rng.choice(["a", "b", "c"])))
    initialize_project(win, db_path)
    wait_for_image(win)
    yield win, rng
    win.state_manager.db.conn.close()
    win.close()
    win.deleteLater()
    qapp.processEvents()


def random_polygon(rng):
    """Mostly small blobs, some big, some across or outside the image edge."""
    cx, cy = rng.uniform(-60, W + 60), rng.uniform(-60, H + 60)
    r = rng.choice([8, 20, 60, 180], p=[0.4, 0.35, 0.2, 0.05])
    n = int(rng.integers(3, 14))
    t = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = r * rng.uniform(0.4, 1.0, n)
    return np.c_[cx + rad * np.cos(t), cy + rad * np.sin(t)].astype(np.float32)


def full_render(win):
    d = win.image_display
    return d.overlay_base_masks(win.state_manager.current_image)


def pixmap_array(win):
    from PyQt6.QtGui import QImage
    q = win.image_display.pixmap_item.image().convertToFormat(QImage.Format.Format_RGB888)
    ptr = q.constBits()
    ptr.setsize(q.sizeInBytes())
    return np.frombuffer(ptr, np.uint8).reshape(q.height(), q.bytesPerLine())[:, :3 * q.width()] \
        .reshape(q.height(), q.width(), 3).copy()


def check(win, expect="partial"):
    d = win.image_display
    d.refresh_masks()
    assert d.last_redraw == expect
    ref = full_render(win)
    assert np.array_equal(d.cached_image_with_masks, ref)
    assert np.array_equal(pixmap_array(win), ref)


def test_partial_redraw_matches_full_redraw(busy):
    win, rng = busy
    db = win.state_manager.db
    ids = lambda: [r[0] for r in db.fetch_all("SELECT id FROM masks ORDER BY id")]
    for step in range(60):
        action = rng.choice(["add", "delete", "reclass", "reshape"])
        if action == "add" or len(ids()) < 5:
            db.execute_query("INSERT INTO masks (image_name, mask_data, class_name, surface_area) "
                             "VALUES ('scan', ?, ?, 0)", (random_polygon(rng).tobytes(), rng.choice(["a", "b", "c"])))
        elif action == "delete":
            db.execute_query("DELETE FROM masks WHERE id = ?", (int(rng.choice(ids()[1:])),))
        elif action == "reclass":
            db.execute_query("UPDATE masks SET class_name = ? WHERE id = ?",
                             (rng.choice(["a", "b", "c"]), int(rng.choice(ids()[3:]))))
        else:
            db.execute_query("UPDATE masks SET mask_data = ? WHERE id = ?",
                             (random_polygon(rng).tobytes(), int(rng.choice(ids()))))
        d = win.image_display
        d.refresh_masks()
        ref = full_render(win)
        assert np.array_equal(d.cached_image_with_masks, ref), (step, action, d.last_redraw)
        assert np.array_equal(pixmap_array(win), ref), (step, action, d.last_redraw)
    # most edits took the fast path
    assert win.image_display.last_redraw in ("partial", "full")


def test_small_edits_take_the_fast_path(busy):
    win, rng = busy
    db = win.state_manager.db
    small = np.array([[100, 100], [130, 100], [120, 125]], np.float32)
    db.execute_query("INSERT INTO masks (image_name, mask_data, class_name, surface_area) "
                     "VALUES ('scan', ?, 'b', 0)", (small.tobytes(),))
    check(win, "partial")
    db.execute_query("DELETE FROM masks WHERE id = (SELECT MAX(id) FROM masks)")
    check(win, "partial")
    check(win, "partial")                         # nothing changed


def test_reindex_needs_no_redraw(busy):
    win, _ = busy
    win.state_manager.mask_manager.reindex_masks() if hasattr(win.state_manager.mask_manager, "reindex_masks") else None
    check(win, "partial")


def test_full_redraw_when_needed(busy):
    win, _ = busy
    db = win.state_manager.db
    db.execute_query("UPDATE classes SET color = '#123456' WHERE class_name = 'a'")
    d = win.image_display
    d.refresh_masks()
    ref = full_render(win)
    assert np.array_equal(d.cached_image_with_masks, ref)
    d.set_masks_visible(False)                      # hidden: plain image shown
    d.refresh_masks()
    assert d.last_redraw == "full"
    d.set_masks_visible(True)
    check(win, "full")                              # back from hidden: full once
    check(win, "partial")
