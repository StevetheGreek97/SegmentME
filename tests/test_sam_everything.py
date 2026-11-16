"""SAM "find every object in the box" (F): core/tools/sam_everything.py and
the SAM tool around it. A fake predictor stands in for SAM: every object in
the test images has its own colour, and a point "segments" the patch of
that colour under it, so no checkpoint is needed."""
import json

import cv2
import numpy as np
import pytest

from core.tools import sam_everything as se

BG = (230, 230, 230)


class ColourPredictor:
    """Answers a point with the patch of colour under it (the background is
    one big patch). `scores` lowers SAM's confidence for some colours;
    `includes` makes a colour's answer include other colours (an object
    with a spot: the spot alone, or the whole)."""

    def __init__(self, scores=None, includes=None, on_predict=None):
        self.scores = scores or {}
        self.includes = includes or {}
        self.on_predict = on_predict
        self.crops = []
        self.calls = 0

    def set_image(self, crop):
        self.crop = crop
        self.crops.append(crop.shape[:2])

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
        self.calls += 1
        if self.on_predict:
            self.on_predict()
        x, y = (int(v) for v in point_coords[0])
        colour = tuple(int(v) for v in self.crop[y, x])
        same = np.all(self.crop == colour, axis=-1)
        for other in self.includes.get(colour, ()):
            same |= np.all(self.crop == other, axis=-1)
        _n, labels = cv2.connectedComponents(same.astype(np.uint8))
        mask = labels == labels[y, x]
        score = self.scores.get(colour, 0.95)
        return np.stack([mask] * 3).astype(np.float32), np.array([score] * 3, np.float32), None


def scene(size=(600, 800), objects=()):
    """An image with filled circles: objects = [(centre, radius, colour)]."""
    image = np.full(size + (3,), BG, np.uint8)
    for centre, radius, colour in objects:
        cv2.circle(image, centre, radius, colour, -1)
    return image


OBJECTS = [((100, 100), 20, (200, 0, 0)), ((300, 150), 30, (0, 150, 0)),
           ((200, 300), 25, (0, 0, 200)), ((650, 450), 30, (120, 60, 0))]   # the last is outside


def centres(polygons):
    """Centre of each outline's bounding box."""
    return sorted((int(round((p[:, 0].min() + p[:, 0].max()) / 2)),
                   int(round((p[:, 1].min() + p[:, 1].max()) / 2))) for p in polygons)


def near(found, expected, tol=3):
    return len(found) == len(expected) and all(
        abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol for a, b in zip(found, sorted(expected)))


# ------------------------------------------------------------------ the search
def test_finds_every_object_in_the_box_and_nothing_outside():
    image = scene(objects=OBJECTS)
    polygons = se.find_objects(ColourPredictor(), image, (40, 40, 400, 400))
    assert near(centres(polygons), [(100, 100), (300, 150), (200, 300)])
    for p in polygons:
        assert p.dtype == np.float32 and p.ndim == 2 and p.shape[1] == 2


def test_box_dragged_in_any_direction():
    image = scene(objects=OBJECTS)
    polygons = se.find_objects(ColourPredictor(), image, (400, 400, 40, 40))
    assert len(polygons) == 3


def test_background_is_not_an_object_and_its_points_are_skipped():
    image = scene(objects=OBJECTS)
    predictor = ColourPredictor()
    se.find_objects(predictor, image, (40, 40, 400, 400))
    grid = len(se.grid_points((0, 0, 360, 360), 1.0))
    assert predictor.calls < grid / 10       # once on the background, once per object, …


def test_red_point_leaves_an_object_out():
    image = scene(objects=OBJECTS)
    polygons = se.find_objects(ColourPredictor(), image, (40, 40, 400, 400),
                               negatives=[(300, 150)])
    assert near(centres(polygons), [(100, 100), (200, 300)])


def test_red_point_on_the_background_keeps_it_fast():
    """A red point on the paper used to make every background point ask SAM
    again (117 s instead of 3 s on a real seed scan)."""
    image = scene(objects=OBJECTS)
    predictor = ColourPredictor()
    polygons = se.find_objects(predictor, image, (40, 40, 400, 400), negatives=[(380, 380)])
    assert len(polygons) == 3
    grid = len(se.grid_points((0, 0, 360, 360), 1.0))
    assert predictor.calls < grid / 10


def test_green_point_adds_an_object_sam_is_unsure_of():
    image = scene(objects=OBJECTS)
    unsure = ColourPredictor(scores={(0, 150, 0): 0.3})
    assert len(se.find_objects(unsure, image, (40, 40, 400, 400))) == 2
    polygons = se.find_objects(ColourPredictor(scores={(0, 150, 0): 0.3}), image,
                               (40, 40, 400, 400), positives=[(300, 150)])
    assert near(centres(polygons), [(100, 100), (300, 150), (200, 300)])


def test_objects_already_saved_are_not_found_again():
    image = scene(objects=OBJECTS)
    saved = np.array([[80, 80], [120, 80], [120, 120], [80, 120]], np.float32)
    polygons = se.find_objects(ColourPredictor(), image, (40, 40, 400, 400), existing=[saved])
    assert near(centres(polygons), [(300, 150), (200, 300)])


def test_a_part_and_its_whole_are_one_object():
    body, spot = (0, 150, 0), (250, 250, 0)
    image = scene(objects=[((300, 150), 40, body), ((300, 150), 8, spot)])
    predictor = ColourPredictor(scores={spot: 0.8}, includes={body: [spot]})
    polygons = se.find_objects(predictor, image, (40, 40, 400, 400))
    assert len(polygons) == 1
    x0, y0 = polygons[0].min(0)
    x1, y1 = polygons[0].max(0)
    assert x1 - x0 > 70 and y1 - y0 > 70     # the whole, not the spot


def test_whole_image_is_searched_in_full_resolution_pieces():
    """No box: the whole scan, split into pieces SAM sees at full resolution.
    Objects on the border between two pieces are found once, whole."""
    objects = [((1000, 800), 80, (200, 0, 0)), ((3000, 2200), 120, (0, 0, 200)),
               ((60, 2950), 12, (0, 120, 0))]               # small, in a corner
    pieces = se.plan_pieces((0, 0, 4000, 3000), (3000, 4000), min_scale=1.0)
    border_x = pieces[0].core[2]                           # where piece 1 ends and 2 begins
    objects.append(((border_x, 400), 60, (150, 0, 150)))   # straddles that border
    image = scene((3000, 4000), objects=objects)
    predictor = ColourPredictor()
    polygons = se.find_objects(predictor, image, min_scale=1.0)
    assert len(pieces) > 1 and len(predictor.crops) == len(pieces)
    assert all(max(c) <= se.MAX_SIDE for c in predictor.crops)       # never shrunk
    assert near(centres(polygons), [c for c, _r, _col in objects], tol=2)
    for (cx, cy), radius, _colour in objects:              # each one whole, not a cut part
        [p] = [p for p in polygons if abs((p[:, 0].min() + p[:, 0].max()) / 2 - cx) <= 2
               and abs((p[:, 1].min() + p[:, 1].max()) / 2 - cy) <= 2]
        assert abs((p[:, 0].max() - p[:, 0].min()) - 2 * radius) <= 2


def test_big_box_is_split_too():
    image = scene((3000, 4000), objects=[((1000, 800), 14, (200, 0, 0)), ((2500, 2000), 14, (0, 0, 200))])
    predictor = ColourPredictor()
    polygons = se.find_objects(predictor, image, (500, 500, 3000, 2500), min_scale=1.0)
    assert len(polygons) == 2 and len(predictor.crops) > 1
    assert all(max(c) <= se.MAX_SIDE for c in predictor.crops)


def test_object_cut_by_a_piece_edge_is_left_to_its_neighbour():
    """A piece sees only part of an object near its window edge; that part
    must not be saved as an object of its own."""
    pieces = se.plan_pieces((0, 0, 4000, 3000), (3000, 4000), min_scale=1.0)
    edge = pieces[0].window[2]                             # right edge of what piece 1 sees
    image = scene((3000, 4000), objects=[((edge, 400), 100, (150, 0, 150))])
    polygons = se.find_objects(ColourPredictor(), image, min_scale=1.0)
    assert len(polygons) == 1
    p = polygons[0]
    assert abs((p[:, 0].max() - p[:, 0].min()) - 200) <= 2   # the whole object


def test_quick_look_shrinks_a_big_area_once():
    """Big objects: SAM looks at the whole scan once (fast); outlines
    still come back in image pixels."""
    image = scene((3000, 4000), objects=[((1000, 800), 80, (200, 0, 0)), ((3000, 2200), 120, (0, 0, 200))])
    predictor = ColourPredictor()
    big_objects = se.min_scale_for([140, 150, 200, 210])     # big: may shrink a lot
    polygons = se.find_objects(predictor, image, min_scale=big_objects)
    assert len(predictor.crops) == 1 and max(predictor.crops[0]) == se.MAX_SIDE
    assert near(centres(polygons), [(1000, 800), (3000, 2200)], tol=8)
    widths = sorted(float(p[:, 0].max() - p[:, 0].min()) for p in polygons)
    assert abs(widths[0] - 160) < 10 and abs(widths[1] - 240) < 10


def test_quick_look_keeps_points_and_saved_masks_in_place():
    image = scene((3000, 4000), objects=[((1000, 800), 80, (200, 0, 0)), ((3000, 2200), 120, (0, 0, 200)),
                                         ((2000, 1500), 90, (0, 150, 0))])
    saved = np.array([[920, 720], [1080, 720], [1080, 880], [920, 880]], np.float32)
    polygons = se.find_objects(ColourPredictor(), image, negatives=[(3000, 2200)], existing=[saved],
                               min_scale=0.2)
    assert near(centres(polygons), [(2000, 1500)], tol=8)


def test_small_animals_on_a_big_scan_are_not_shrunk_away():
    """A 12000 px scan with 60 px animals: one look would shrink them to
    5 px. The search shrinks only as far as the saved masks allow and
    splits the scan into pieces instead."""
    animals = [((1500, 2000), 30, (200, 0, 0)), ((6000, 9000), 30, (0, 0, 200)),
               ((7300, 500), 25, (0, 150, 0))]
    image = scene((12000, 7600), objects=animals)
    saved_sizes = [45, 50, 55, 60, 70]                      # sqrt(area) of masks saved so far
    min_scale = se.min_scale_for(saved_sizes)
    predictor = ColourPredictor()
    polygons = se.find_objects(predictor, image, min_scale=min_scale)
    assert near(centres(polygons), [c for c, _r, _col in animals], tol=4)
    pieces = se.plan_pieces((0, 0, 7600, 12000), image.shape, min_scale=min_scale)
    assert 1 < len(pieces) < len(se.plan_pieces((0, 0, 7600, 12000), image.shape, min_scale=1.0))
    assert all(p.scale == pytest.approx(min_scale) for p in pieces)
    for (cx, cy), r, _col in animals:                       # outlines in image pixels
        [p] = [p for p in polygons if abs((p[:, 0].min() + p[:, 0].max()) / 2 - cx) <= 4]
        assert abs((p[:, 0].max() - p[:, 0].min()) - 2 * r) <= 2.5 / min_scale   # ±2.5 px of what SAM saw


def test_how_far_the_image_may_shrink():
    assert se.min_scale_for([]) is None                     # nothing saved yet: the default
    assert se.min_scale_for([100] * 10) == pytest.approx(se.TARGET_OBJECT_PX / 100)
    assert se.min_scale_for([8] * 10) == 1.0                # tiny objects: full resolution
    assert se.min_scale_for([10000] * 10) == se.MIN_SCALE_LIMIT
    plan = se.plan_pieces((0, 0, 2464, 2056), (2056, 2464, 3))   # a seed photo, sizes unknown
    assert len(plan) == 1 and plan[0].scale == pytest.approx(se.MAX_SIDE / 2464)   # one quick look


def test_small_box_is_seen_at_full_resolution():
    image = scene((3000, 4000), objects=[((1000, 800), 14, (200, 0, 0))])   # 28 px: tiny on the scan
    predictor = ColourPredictor()
    polygons = se.find_objects(predictor, image, (900, 700, 1100, 900))
    assert len(polygons) == 1
    h, w = predictor.crops[0]
    assert (h, w) == (200 + 2 * 20, 200 + 2 * 20)    # region + margin, not shrunk


def test_tiny_box_finds_nothing():
    assert se.find_objects(ColourPredictor(), scene(objects=OBJECTS), (10, 10, 15, 15)) == []


def test_can_be_stopped():
    image = scene(objects=OBJECTS)
    calls = []
    with pytest.raises(se.Cancelled):
        se.find_objects(ColourPredictor(on_predict=lambda: calls.append(1)), image,
                        (40, 40, 400, 400), cancelled=lambda: len(calls) >= 2)
    assert len(calls) == 2


def test_progress_is_reported():
    seen = []
    se.find_objects(ColourPredictor(), scene(objects=OBJECTS), (40, 40, 400, 400),
                    progress=lambda done, total: seen.append((done, total)))
    assert seen[0][0] == 0 and seen[-1][0] == seen[-1][1] > 0


# ------------------------------------------------------------------ the SAM tool
sam2 = pytest.importorskip("sam2", reason="the SAM tool imports the sam2 package")


class Host:
    def __init__(self, class_name="seed", existing=()):
        from PyQt6.QtWidgets import QGraphicsScene
        self.scene = QGraphicsScene()
        self.class_name = class_name
        self.existing = list(existing)
        self.batches = []

    def image(self):
        return scene(objects=OBJECTS)

    def image_name(self):
        return "img"

    def choose_class(self):
        return self.class_name

    def save_mask(self, polygon, image_name, class_name):
        raise AssertionError("F results are saved in one go")

    def save_masks(self, polygons, image_name, class_name):
        self.batches.append((list(polygons), image_name, class_name))

    def existing_polygons(self):
        return self.existing

    def typical_object_sizes(self):
        return []


def make_tool(host):
    from core.tools.sam2_masker import SamMasker2

    class Tool(SamMasker2):
        def _load_model(self):
            return ColourPredictor()
    return Tool(host)


def run_f(tool, image):
    job = tool.find_all_job()
    return tool.apply_all(tool.compute_all(image, *job))


def test_f_without_a_box_searches_nothing(qapp):
    host = Host()
    tool = make_tool(host)
    tool.add_point((100, 100), 1)                           # a green point alone is not enough
    assert tool.find_all_job() is None
    assert tool.found == []


def test_f_shows_every_object_and_s_saves_them_together(qapp):
    host = Host()
    tool = make_tool(host)
    tool.add_box((40, 40), (400, 400))
    message = run_f(tool, host.image())
    assert len(tool.found) == 3 and len(tool.found_items) == 3
    assert "3 object" in message
    tool.complete_mask()
    [(polygons, name, cls)] = host.batches
    assert len(polygons) == 3 and name == "img" and cls == "seed"
    assert tool.found == [] and not host.scene.items()      # box, points, outlines gone


def test_right_click_on_a_result_leaves_it_out(qapp):
    host = Host()
    tool = make_tool(host)
    tool.add_box((40, 40), (400, 400))
    run_f(tool, host.image())
    tool.right_click((300, 150))
    assert near(centres(tool.found), [(100, 100), (200, 300)])
    assert tool.background_points == [(300, 150)]           # stays out when F runs again
    run_f(tool, host.image())
    assert len(tool.found) == 2


def test_green_points_and_saved_masks_reach_the_search(qapp):
    saved = np.array([[80, 80], [120, 80], [120, 120], [80, 120]], np.float32)
    host = Host(existing=[saved])
    tool = make_tool(host)
    tool.add_box((40, 40), (400, 400))
    run_f(tool, host.image())
    assert near(centres(tool.found), [(300, 150), (200, 300)])


def test_no_class_keeps_the_results(qapp):
    host = Host(class_name=None)
    tool = make_tool(host)
    tool.add_box((40, 40), (400, 400))
    run_f(tool, host.image())
    tool.complete_mask()
    assert host.batches == [] and len(tool.found) == 3


def test_e_replaces_what_f_found(qapp):
    host = Host()
    tool = make_tool(host)
    tool.add_box((40, 40), (400, 400))
    run_f(tool, host.image())
    tool.apply_polygon([[0, 0], [10, 0], [10, 10]], 1024)
    assert tool.found == [] and tool.mask is not None


def test_another_image_stops_a_running_search(qapp):
    host = Host()
    tool = make_tool(host)
    tool.add_box((40, 40), (400, 400))
    job = tool.find_all_job()
    tool.image_changed(None)
    with pytest.raises(se.Cancelled):
        tool.compute_all(host.image(), *job)


# ------------------------------------------------------------------ in the window
@pytest.fixture
def window(qapp, tmp_path, no_popups):
    from PIL import Image
    from ui.main_window import MainApp
    from services.recent_projects import initialize_project
    from tests.conftest import wait_for_image
    from core.tools.sam2_masker import SamMasker2
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    (root / "images").mkdir()
    Image.fromarray(scene(objects=OBJECTS)).save(root / "images" / "a.png")
    (root / "proj.SEproj").write_text(json.dumps({"name": "proj", "images": ["a.png"]}))
    db = str(root / ".segmentme" / "masks.db")
    win = MainApp(db_path=db)
    initialize_project(win, db)
    wait_for_image(win)
    win.state_manager.db.execute_query("INSERT INTO classes (class_name, color) VALUES (?, ?)",
                                       ("seed", "#ff0000"))
    win.sidebar.populate_class_dropdown()
    win.sidebar.class_dropdown.setCurrentIndex(win.sidebar.class_dropdown.findText("seed"))

    class Tool(SamMasker2):
        def _load_model(self):
            return ColourPredictor()
    tool = Tool(win.image_display.tool_host)
    win.tool_manager.current_tool = tool
    yield win, tool
    win.tool_manager.current_tool = None
    tool.clear_temp_items()
    win.close()
    qapp.processEvents()
    win.state_manager.db.conn.close()
    win.deleteLater()
    qapp.processEvents()


def _key(display, key):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    display.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))


def _wait(qapp, cond, timeout=20):
    import time
    end = time.time() + timeout
    while time.time() < end and not cond():
        qapp.processEvents()
        time.sleep(0.005)


def test_f_and_s_in_the_window_save_every_object(window, qapp):
    from PyQt6.QtCore import Qt
    win, tool = window
    updates = []
    win.state_manager.masks_updated.connect(lambda: updates.append(1))
    tool.add_box((40, 40), (400, 400))
    _key(win.image_display, Qt.Key.Key_F)
    _wait(qapp, lambda: not win.image_display.segmenting())
    assert len(tool.found) == 3
    assert "3 object" in win.statusBar().currentMessage()
    _key(win.image_display, Qt.Key.Key_S)
    rows = win.state_manager.db.fetch_all("SELECT class_name, surface_area FROM masks WHERE image_name='a'")
    assert len(rows) == 3 and all(cls == "seed" and area > 1000 for cls, area in rows)
    assert len(updates) == 1                                 # one refresh for all of them
    # F again: the saved objects are not found a second time
    tool.add_box((40, 40), (400, 400))
    _key(win.image_display, Qt.Key.Key_F)
    _wait(qapp, lambda: not win.image_display.segmenting())
    assert tool.found == []
    assert "no new objects" in win.statusBar().currentMessage()


def test_f_without_a_box_asks_for_one(window, qapp):
    from PyQt6.QtCore import Qt
    win, tool = window
    _key(win.image_display, Qt.Key.Key_F)
    assert not win.image_display.segmenting()
    assert "Draw a box first" in win.statusBar().currentMessage()
    assert tool.found == []


def test_shift_f_does_nothing(window, qapp):
    """Only plain F searches (Shift+F was removed)."""
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    win, tool = window
    tool.add_box((40, 40), (400, 400))
    win.image_display.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_F,
                                              Qt.KeyboardModifier.ShiftModifier))
    assert not win.image_display.segmenting()
    assert tool.found == []


def test_save_masks_skips_broken_outlines(window):
    win, _tool = window
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    n = win.state_manager.mask_manager.save_masks([square, square[:2]], "a", "seed")
    assert n == 1
    assert win.state_manager.mask_manager.save_masks([], "a", "seed") == 0
