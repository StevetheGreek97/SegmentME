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
