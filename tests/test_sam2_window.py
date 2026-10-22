"""SAM2 click tool: prediction in a full-resolution window around the prompts
(core/tools/sam2_masker.py). A fake predictor stands in for the model: it
"segments" a known object, so no checkpoint is needed."""
import numpy as np
import pytest

pytest.importorskip("sam2", reason="the SAM2 tool imports the sam2 package")

from PyQt6.QtWidgets import QGraphicsScene  # noqa: E402

from core.tools.sam2_masker import SamMasker2  # noqa: E402


class FakePredictor:
    """Returns the part of `truth` (a full-image bool mask) inside the window
    it was last given; records every embedding."""

    def __init__(self, truth):
        self.truth = truth
        self.embedded = []
        self.window = None

    def set_image(self, crop):
        self.embedded.append(crop.shape[:2])

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
        x0, y0, x1, y1 = self.window
        return self.truth[None, y0:y1, x0:x1].astype(np.float32), None, None


class Tool(SamMasker2):
    truth = None

    def _load_model(self):
        return FakePredictor(self.truth)

    def _predict_in_window(self, image, window, points, labels, box):
        self.predictor.window = window
        return super()._predict_in_window(image, window, points, labels, box)


class Parent:
    def __init__(self):
        self.scene = QGraphicsScene()


def make_tool(qapp, truth):
    Tool.truth = truth
    return Tool(Parent())


def bbox(polygon):
    p = np.asarray(polygon)
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


@pytest.fixture
def image():
    return np.zeros((6000, 5000, 3), np.uint8)


def test_small_object_is_found_at_full_resolution(qapp, image):
    truth = np.zeros(image.shape[:2], bool)
    truth[3000:3040, 2000:2060] = True             # 60 x 40 px object
    tool = make_tool(qapp, truth)
    tool.foreground_points = [(2030, 3020)]
    polygon = tool.generate_mask(image)
    assert bbox(polygon) == (2000, 3000, 2059, 3039)
    assert tool.predictor.embedded == [(1024, 1024)]   # one window, not the whole scan


def test_window_grows_until_the_object_fits(qapp, image):
    truth = np.zeros(image.shape[:2], bool)
    truth[1000:4000, 500:4200] = True             # bigger than the first window
    tool = make_tool(qapp, truth)
    tool.foreground_points = [(2300, 2500)]
    polygon = tool.generate_mask(image)
    assert bbox(polygon) == (500, 1000, 4199, 3999)
    sides = [max(s) for s in tool.predictor.embedded]
    assert sides == sorted(sides) and len(sides) > 1


def test_refining_the_same_object_reuses_the_embedding(qapp, image):
    truth = np.zeros(image.shape[:2], bool)
    truth[3000:3040, 2000:2060] = True
    tool = make_tool(qapp, truth)
    tool.foreground_points = [(2030, 3020)]
    tool.generate_mask(image)
    tool.background_points = [(2100, 3100)]
    tool.generate_mask(image)
    assert len(tool.predictor.embedded) == 1


def test_a_new_image_gets_a_new_embedding(qapp, image):
    truth = np.zeros(image.shape[:2], bool)
    truth[3000:3040, 2000:2060] = True
    tool = make_tool(qapp, truth)
    tool.foreground_points = [(2030, 3020)]
    tool.generate_mask(image)
    tool.generate_mask(image.copy())
    assert len(tool.predictor.embedded) == 2


def test_box_dragged_in_any_direction(qapp, image):
    truth = np.zeros(image.shape[:2], bool)
    truth[200:300, 100:400] = True
    tool = make_tool(qapp, truth)
    tool.box = np.array([410, 310, 90, 190])     # dragged bottom-right to top-left
    assert bbox(tool.generate_mask(image)) == (100, 200, 399, 299)


def test_small_image_uses_the_whole_image(qapp):
    small = np.zeros((300, 500, 3), np.uint8)
    truth = np.zeros((300, 500), bool)
    truth[:, :] = True
    tool = make_tool(qapp, truth)
    tool.foreground_points = [(250, 150)]
    tool.generate_mask(small)
    assert tool.predictor.embedded == [(300, 500)]


def test_largest_piece_is_kept(qapp, image):
    truth = np.zeros(image.shape[:2], bool)
    truth[3000:3100, 2000:2100] = True
    truth[3200:3205, 2200:2205] = True            # a speck
    tool = make_tool(qapp, truth)
    tool.foreground_points = [(2050, 3050)]
    assert bbox(tool.generate_mask(image)) == (2000, 3000, 2099, 3099)


# --------------------------------------------------------------------------
# E runs SAM on a worker thread: the window stays responsive
# --------------------------------------------------------------------------
class SlowPredictor(FakePredictor):
    def predict(self, *a, **k):
        import time
        time.sleep(0.4)          # a slow CPU prediction
        return super().predict(*a, **k)


class SlowTool(Tool):
    def _load_model(self):
        return SlowPredictor(self.truth)


def _press_e(display):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtCore import QEvent
    display.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_E, Qt.KeyboardModifier.NoModifier))


def _wait_until(qapp, cond, timeout=10):
    import time
    end = time.time() + timeout
    ticks = 0
    while time.time() < end and not cond():
        qapp.processEvents()
        ticks += 1
        time.sleep(0.005)
    return ticks


@pytest.fixture
def sam_window(qapp, tmp_path, no_popups):
    import json
    from PIL import Image
    from ui.main_window import MainApp
    from services.recent_projects import initialize_project
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    (root / "images").mkdir()
    Image.new("RGB", (3000, 2000), (200, 160, 60)).save(root / "images" / "scan.png")
    Image.new("RGB", (300, 200)).save(root / "images" / "zz.png")
    (root / "proj.SEproj").write_text(json.dumps({"name": "proj", "images": ["scan.png", "zz.png"]}))
    db = str(root / ".segmentme" / "masks.db")
    win = MainApp(db_path=db)
    initialize_project(win, db)
    win.state_manager.image_index = 0
    win.image_display.display_image(win.state_manager.image_paths[0])
    truth = np.zeros((2000, 3000), bool)
    truth[900:960, 1400:1500] = True
    SlowTool.truth = truth
    tool = SlowTool(win.image_display.tool_host)
    win.tool_manager.current_tool = tool
    yield win, tool
    win.state_manager.db.conn.close()
    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_e_runs_in_the_background(sam_window, qapp):
    import time
    win, tool = sam_window
    tool.foreground_points = [(1450.0, 930.0)]
    t = time.perf_counter()
    _press_e(win.image_display)
    assert time.perf_counter() - t < 0.2                  # returned at once
    assert win.image_display.segmenting()
    ticks = _wait_until(qapp, lambda: not win.image_display.segmenting())
    assert ticks > 10                                     # the window kept handling events
    assert bbox(tool.mask) == (1400, 900, 1499, 959)
    assert tool.current_polygon_item is not None


def test_result_for_another_image_is_dropped(sam_window, qapp):
    win, tool = sam_window
    tool.foreground_points = [(1450.0, 930.0)]
    _press_e(win.image_display)
    win.image_display.display_image(win.state_manager.image_paths[1])   # moved on meanwhile
    _wait_until(qapp, lambda: not win.image_display.segmenting())
    assert tool.mask is None and tool.current_polygon_item is None


def test_e_while_running_runs_again_with_new_points(sam_window, qapp):
    win, tool = sam_window
    tool.foreground_points = [(1450.0, 930.0)]
    _press_e(win.image_display)
    tool.background_points = [(1300.0, 800.0)]
    _press_e(win.image_display)                           # queued, not run in parallel
    _wait_until(qapp, lambda: not win.image_display.segmenting())
    assert len(tool.predictor.embedded) == 1              # second run reused the window
    assert bbox(tool.mask) == (1400, 900, 1499, 959)


# --------------------------------------------------------------------------
# DEXTR's E runs on the worker thread too
# --------------------------------------------------------------------------
def _slow_dextr(truth_value=8.0):
    import time
    import torch
    from core.tools.dextr_mask import DEXTRMasker

    class FakeModel:
        def forward(self, inputs):
            time.sleep(0.4)                               # a slow CPU prediction
            out = torch.full((1, 1, 512, 512), -8.0)
            out[..., 150:360, 150:360] = truth_value      # an object in the middle of the crop
            return out

    class SlowDextr(DEXTRMasker):
        def _load_dextr_model(self):
            return FakeModel()
    return SlowDextr


def test_dextr_e_runs_in_the_background(sam_window, qapp):
    import time
    win, _sam = sam_window
    tool = _slow_dextr()(win.image_display.tool_host)
    win.tool_manager.current_tool = tool
    for p in ((1400, 930), (1500, 930), (1450, 900), (1450, 960)):
        tool.add_point(p)
    t = time.perf_counter()
    _press_e(win.image_display)
    assert time.perf_counter() - t < 0.2                  # returned at once
    assert win.image_display.segmenting()
    assert "DEXTR" in win.statusBar().currentMessage()
    ticks = _wait_until(qapp, lambda: not win.image_display.segmenting())
    assert ticks > 10                                     # the window kept handling events
    assert tool.mask is not None and len(tool.mask) >= 3
    x0, y0, x1, y1 = bbox(tool.mask)
    assert 1300 < x0 < x1 < 1600 and 800 < y0 < y1 < 1060   # around the clicked object


def test_dextr_finding_nothing_says_so(sam_window, qapp):
    win, _sam = sam_window
    tool = _slow_dextr(truth_value=-8.0)(win.image_display.tool_host)   # empty prediction
    win.tool_manager.current_tool = tool
    for p in ((1400, 930), (1500, 930), (1450, 900), (1450, 960)):
        tool.add_point(p)
    _press_e(win.image_display)
    _wait_until(qapp, lambda: not win.image_display.segmenting())
    assert tool.mask is None
    assert "found nothing" in win.statusBar().currentMessage()
