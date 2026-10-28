"""
Guided tours: the window dims except the real part being explained, and a
speech bubble next to it says what it is and how to use it (Back / Next,
Skip). Read-only: nothing in the window can be clicked while a tour runs.

* "Getting Started" is offered once, the first time a project opens.
* Help > Tours lists every tour (Getting Started plus short topic tours);
  any of them can be replayed at any time.

Keys while a tour runs: → / Enter / Space = Next, ← / Backspace = Back,
Esc = close.

A step names its target as a function of the window, so the tour always
points at the live widget (or a menu title); when the target is missing or
hidden the bubble simply appears in the middle.
"""
from dataclasses import dataclass
from typing import Callable, Optional

from PyQt6.QtCore import QEvent, QPoint, QRect, QRectF, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from services.logger import get_logger

logger = get_logger(__name__)

SETTINGS_ORG = "SegmentME"
KEY_OFFERED = "tour/getting_started_offered"
BUBBLE_WIDTH = 360
HOLE_PADDING = 6
DIM = QColor(0, 0, 0, 170)


@dataclass
class Step:
    title: str
    text: str                                   # simple HTML
    target: Optional[Callable] = None           # window -> widget, [widgets] or menu("File")
    prepare: Optional[Callable] = None          # window -> None: e.g. open the panels shown


def menu(name):
    """Target: a title in the menu bar, e.g. menu("File")."""
    return lambda window: ("menu", name)


def _sidebar(attr):
    return lambda w: getattr(w.sidebar, attr)


def _panels(w, tab=None):
    """Open File > Annotations (table + plots), optionally on a plot tab."""
    w.show_results()
    if tab is not None:
        w.statistics.tabs.setCurrentIndex(tab)


def _tools(w):
    s = w.sidebar
    return [s.manual_mask, s.sam, s.dextr, s.intelligent_scissors]


# --------------------------------------------------------------------------
# The tours
# --------------------------------------------------------------------------
GETTING_STARTED = [
    Step("Welcome to SegmentME",
         "This short tour shows the main window and how to draw your first masks. "
         "It takes about two minutes.<br><br>"
         "Use <b>Next</b> (or the → key) to go on and <b>Back</b> (←) to go back. "
         "<b>Esc</b> closes the tour. You can watch it again any time from "
         "<b>Help → Tours</b>."),
    Step("The image",
         "Your image is shown here.<br><br>"
         "• <b>Mouse wheel</b>: zoom in and out.<br>"
         "• <b>Middle mouse button + drag</b>: move around.<br>"
         "• <b>Ctrl + wheel</b> moves up and down, <b>Shift + wheel</b> left and right.<br>"
         "• Hold <b>H</b> to hide the masks for a moment and see the image underneath.<br>"
         "• Press <b>V</b> to see only the outlines of the masks (V again: back).<br><br>"
         "<b>Settings → Display…</b> changes how strong the mask colour is, which classes "
         "are shown, and how the image looks (brightness, contrast, gamma, auto-contrast, "
         "sharpening, one colour channel...).",
         lambda w: w.image_display),
    Step("All your images",
         "Every image of the project is here. Drag the slider to move through them, "
         "or type part of a name in the search box to jump straight to an image.",
         lambda w: w.slider),
    Step("Previous and next image",
         "These buttons show the previous and the next image. The <b>←</b> and "
         "<b>→</b> keys do the same and are quicker.",
         lambda w: [w.sidebar.prev_button, w.sidebar.next_button]),
    Step("Choose the class first",
         "Before you draw, choose what you are going to mark here (for example "
         "\"Daphnia\"). Every mask you draw gets the class chosen in this list.",
         _sidebar("class_dropdown")),
    Step("Adding and removing classes",
         "<b>Add Class</b> creates a new class: you give it a name and a colour.<br><br>"
         "<b>Remove Selected</b> deletes the chosen class <i>and all its masks</i>. "
         "SegmentME asks first and makes a backup, so it can be undone with "
         "File → Restore Backup….<br><br>"
         "Colours can be changed later in <b>Actions → Class Colours…</b>",
         lambda w: [w.sidebar.add_class_button, w.sidebar.remove_class_button]),
    Step("The drawing tools",
         "Click a tool to switch it on. Click it again, or press <b>Q</b>, to switch "
         "it off.<br><br>Each tool has a key: <b>P</b> Manual, <b>A</b> SAM, "
         "<b>D</b> DEXTR, <b>I</b> Scissors. The next steps show how each one works.<br><br>"
         "With every tool, <b>S</b> saves the outline as a mask of the chosen class.",
         _tools),
    Step("Manual (key P)",
         "Draw the outline yourself: <b>click</b> around the object to place points one "
         "after another. <b>Right-click</b> removes the last point.<br><br>"
         "When the outline is done, press <b>S</b> to save it.",
         _sidebar("manual_mask")),
    Step("SAM (key A)",
         "SAM finds the outline for you.<br><br>"
         "• <b>Click</b> on the object (green point = this is the object).<br>"
         "• <b>Right-click</b> on anything that is not part of it (red point).<br>"
         "• Or hold <b>Ctrl</b> and drag a box around the object.<br><br>"
         "Press <b>E</b> to work out the outline, then <b>S</b> to save it. The first "
         "time, loading SAM takes a little while.",
         _sidebar("sam")),
    Step("DEXTR (key D)",
         "Click the <b>four outermost points</b> of the object: the top-most, "
         "bottom-most, left-most and right-most point.<br><br>"
         "Press <b>E</b> to work out the outline, then <b>S</b> to save it.",
         _sidebar("dextr")),
    Step("Scissors (key I)",
         "Good for objects with clear edges: click a point on the edge and move the "
         "mouse along it. The line follows the edge by itself. Click to fix a point, "
         "and carry on around the object.<br><br>Press <b>S</b> to save.",
         _sidebar("intelligent_scissors")),
    Step("Selecting and editing masks",
         "Switch the tools off (<b>Q</b>) and <b>click a mask</b> to select it.<br><br>"
         "• <b>Delete</b> removes the selected masks.<br>"
         "• <b>Right-click</b> for more: change the class, select all masks, "
         "<i>split</i> a mask by drawing a line through it, or <i>paint and erase</i> "
         "with the brush (<b>[</b> and <b>]</b> change its size).",
         lambda w: w.image_display),
    Step("Your masks in numbers",
         "<b>File → Annotations</b> opens a table of every mask on the image and "
         "statistics per class.<br><br>The File menu also imports more images and "
         "annotations, exports your work, and restores backups.",
         menu("File")),
    Step("That's it!",
         "<b>Your work saves itself</b>: every mask is stored the moment you make it, "
         "and backups are made automatically.<br><br>"
         "More tours are in <b>Help → Tours</b>: editing masks (class, cut, brush), "
         "the table and plots, finding masks automatically, training your own model, "
         "measuring, and exporting.<br><br>Happy annotating!"),
]

EDITING = [
    Step("Selecting masks",
         "Switch the drawing tools off first (press <b>Q</b>). Then:<br><br>"
         "• <b>Click a mask</b> to select it; click more masks to add them.<br>"
         "• Click an empty spot to clear the selection.<br><br>"
         "Selected masks are also selected in the table (File → Annotations).",
         lambda w: w.image_display),
    Step("The mask menu",
         "With the tools off, <b>right-click</b> on the image to open the mask menu:<br><br>"
         "• <b>Set class</b>: give the selected masks another class.<br>"
         "• <b>Select all masks on image</b> / <b>Clear selection</b>.<br>"
         "• <b>Split mask</b>: cut one mask in two.<br>"
         "• <b>Brush edit</b>: paint onto or erase from one mask.<br>"
         "• <b>Delete</b> the selected masks.<br><br>The next steps explain each one.",
         lambda w: w.image_display),
    Step("Changing the class",
         "Wrong class? Select the masks (one or many), right-click → <b>Set class</b> "
         "and pick the right one. All selected masks change at once.<br><br>"
         "In the table you can also <b>double-click the Class</b> of a row.",
         _sidebar("class_dropdown")),
    Step("Cutting a mask in two",
         "Two objects in one mask? Select that one mask, right-click → "
         "<b>Split mask</b>, then <b>drag a line right across the mask</b>, starting and "
         "ending outside it. When you let go, the mask is cut along the line.<br><br>"
         "<b>Esc</b> cancels.",
         lambda w: w.image_display),
    Step("Painting and erasing",
         "To fix an edge, select one mask, right-click → <b>Brush edit</b>:<br><br>"
         "• <b>Left-drag</b> paints: it adds to the mask.<br>"
         "• <b>Right-drag</b> erases.<br>"
         "• <b>[</b> and <b>]</b> (or <b>Alt + wheel</b>) make the brush smaller or "
         "bigger.<br><br><b>S</b> keeps the change, <b>Esc</b> cancels it.",
         lambda w: w.image_display),
    Step("Deleting masks",
         "Select the masks and press <b>Delete</b> (or right-click → Delete).<br><br>"
         "Deleted by mistake? <b>File → Restore Backup…</b> goes back to a copy from "
         "earlier (one is made every 20 minutes while you work).",
         menu("File")),
]

TABLE_AND_PLOTS = [
    Step("Table and plots",
         "<b>File → Annotations</b> opens two panels on the right: a table of the masks "
         "and plots. The next steps open them for you.",
         menu("File")),
    Step("The table",
         "One row per mask on the image you are looking at: its image, ID, class, "
         "<b>area</b>, <b>length</b> and <b>width</b>. With a scale set (Actions → Set "
         "Scale…) the sizes are in mm or µm, otherwise in pixels.<br><br>"
         "Click a column title to sort by it.",
         lambda w: w.annotations.table, lambda w: _panels(w)),
    Step("Table and image work together",
         "• <b>Click a row</b> to select that mask on the image, and clicking a mask on "
         "the image selects its row.<br>"
         "• <b>Double-click the Class</b> of a row to change it.<br>"
         "• <b>Right-click</b> rows for the same menu as on the image: change the class of "
         "many masks at once, delete them, and more.",
         lambda w: w.annotations.table, lambda w: _panels(w)),
    Step("Finding masks",
         "Type in <b>Search</b> to show only matching rows, or choose a class to show "
         "only its masks.",
         lambda w: [w.annotations.search, w.annotations.class_filter], lambda w: _panels(w)),
    Step("Counting masks",
         "The plots count the masks per class:<br><br>"
         "• <b>All Images</b>: the whole project.<br>"
         "• <b>Current Image</b>: only the image on screen.<br><br>"
         "They update by themselves while you draw.",
         lambda w: w.statistics, lambda w: _panels(w, 0)),
    Step("Sizes",
         "<b>Sizes</b> shows how long the masks of each class are on this image, as a "
         "histogram: handy to spot a mask that is far too big or too small.",
         lambda w: w.statistics, lambda w: _panels(w, 2)),
    Step("Closing and exporting",
         "Close a panel with its <b>×</b>; File → Annotations opens it again. Panels can "
         "be dragged to another side or out of the window.<br><br>"
         "For the numbers in Excel or R: <b>File → Export Morphometrics…</b>",
         menu("File")),
]

INFERENCE = [
    Step("Finding masks automatically",
         "<b>Actions → Run Inference</b> lets a model find the masks on all images of "
         "the project for you.",
         menu("Actions")),
    Step("Choosing a model",
         "In the window that opens, choose the kind of model:<br><br>"
         "• <b>YOLO</b>: a model you trained yourself (Actions → Train Custom Model) "
         "or one put in the models folder.<br>"
         "• <b>HiReS</b>: the same YOLO model, run piece by piece, for very large scans.<br>"
         "• <b>SAM</b>: outlines every object it sees, without classes. More SAM models "
         "can be downloaded in <b>Settings → Models…</b><br><br>"
         "<b>Confidence</b>: lower finds more objects but also more mistakes."),
    Step("While it runs",
         "A small window shows the progress. You can keep working meanwhile: masks "
         "appear on each image as soon as it is done.<br><br>"
         "<b>Cancel</b> stops the rest. The masks already found are kept."),
    Step("Check the results",
         "Models make mistakes. Go through the images (→ key), fix wrong classes with a "
         "right-click on the mask, delete wrong masks (Delete key), and draw the missing "
         "ones with the tools.<br><br>New class names from the model are added to this "
         "list by themselves.",
         _sidebar("class_dropdown")),
]

TRAINING = [
    Step("Teach a model your objects",
         "A model learns from the masks you drew. The more images you annotated "
         "carefully, the better it gets. A few dozen images is a good start."),
    Step("First: export the annotations",
         "Training reads your masks from an exported dataset: <b>File → Export "
         "annotations</b>, format YOLO. If you forget, or changed masks since, "
         "SegmentME offers to do it for you when you start training.",
         menu("File")),
    Step("Start training",
         "<b>Actions → Train Custom Model</b>. Choose the model size (the letter: "
         "<b>n</b> is fastest, <b>x</b> most accurate but slow) and the number of epochs "
         "(how many times it studies all images). The other settings can stay as "
         "they are.<br><br>The first training needs the internet once, to download the "
         "starting model.",
         menu("Actions")),
    Step("While it trains",
         "The Training Monitor shows the progress and the curves. On an ordinary "
         "computer without a graphics card this can take hours. Leave it running.<br><br>"
         "<b>Stop</b> ends it early. The best model so far is kept."),
    Step("Use your model",
         "The trained model appears in the list of <b>Actions → Run Inference</b>. "
         "Let it annotate the remaining images, correct its mistakes, and train again: "
         "each round it gets better.",
         menu("Actions")),
]

MEASURING = [
    Step("Set the scale",
         "To measure in mm or µm instead of pixels, SegmentME needs to know how big a "
         "pixel is: <b>Actions → Set Scale…</b><br><br>"
         "• <b>Draw on the image…</b> along something of known length (a ruler on "
         "the scan) and type its length.<br>"
         "• Or type the size of a pixel directly.<br>"
         "• Or use the resolution (DPI) stored in the image files.",
         menu("Actions")),
    Step("The scale bar",
         "With a scale set, a scale bar appears in the corner of the image. It can be "
         "switched off in <b>View → Show Scale Bar</b>.",
         lambda w: w.image_display),
    Step("Sizes of your masks",
         "<b>File → Annotations</b>: the table lists every mask with its area and "
         "length in real units, and the statistics show the sizes per class.",
         menu("File")),
    Step("Export the measurements",
         "<b>File → Export Morphometrics…</b> saves one row per mask with all its "
         "measurements as a CSV file, ready for Excel or R. Choose which classes and "
         "images to include.",
         menu("File")),
]

IMPORT_EXPORT = [
    Step("Adding images",
         "<b>File → Import Images</b> adds more images to the project. You choose to "
         "<b>copy</b> them into the project folder (safe, uses disk space) or to "
         "<b>link</b> to the originals (no copy, but the originals must stay where "
         "they are).",
         menu("File")),
    Step("Importing annotations",
         "<b>File → Import Annotations…</b> reads masks made elsewhere (YOLO .txt files "
         "with their data.yaml) for images that are already in the project. A summary "
         "shows what will happen before anything is changed.",
         menu("File")),
    Step("Exporting a dataset",
         "<b>File → Export annotations</b> writes all masks as a dataset in YOLO or COCO "
         "format, split into training, validation and test images, into the project "
         "folder. Use it for training here or in other programs."),
]

SAFETY = [
    Step("Your work is saved",
         "There is no Save button: every mask is stored the moment you make it.<br><br>"
         "Backups are made when the project opens, every 20 minutes while you work, and "
         "before risky actions. <b>File → Restore Backup…</b> goes back to one of them.",
         menu("File")),
    Step("When something goes wrong",
         "<b>Help → Save Diagnostic Report…</b> makes a zip file to send to the "
         "developer: it tells what happened, without your images or masks.<br><br>"
         "<b>Help → Check for Updates…</b> looks for a newer version. SegmentME also "
         "checks by itself once a day.",
         menu("Help")),
    Step("Settings",
         "<b>Settings → Display…</b>: mask colour strength, outlines only (V), hiding "
         "classes, and image adjustments (brightness, contrast, gamma, saturation, "
         "auto-contrast, sharpen, invert, one colour channel). Only the view changes, "
         "never your files.<br><br>"
         "<b>Settings → Models…</b> downloads and chooses the SAM models. Automatic "
         "update checks can be switched off here too.",
         menu("Settings")),
]

TOURS = {
    "getting_started": ("Getting Started", GETTING_STARTED),
    "editing": ("Editing Masks", EDITING),
    "table": ("Table and Plots", TABLE_AND_PLOTS),
    "inference": ("Finding Masks Automatically", INFERENCE),
    "training": ("Training Your Own Model", TRAINING),
    "measuring": ("Scale and Measurements", MEASURING),
    "import_export": ("Importing and Exporting", IMPORT_EXPORT),
    "safety": ("Saving, Backups and Help", SAFETY),
}


# --------------------------------------------------------------------------
# The overlay
# --------------------------------------------------------------------------
class TourOverlay(QWidget):
    """Covers the window; dims everything but the current step's target."""

    finished = pyqtSignal(bool)        # True = went through to the end

    def __init__(self, window, steps, name="tour"):
        super().__init__(window)
        self.window_ = window
        self.steps = steps
        self.name = name
        self.index = 0
        self.hole = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setGeometry(window.rect())
        window.installEventFilter(self)

        self.bubble = QFrame(self)
        self.bubble.setObjectName("TourBubble")
        pal = window.palette()
        base = pal.color(QPalette.ColorRole.Base).name()
        text = pal.color(QPalette.ColorRole.Text).name()
        accent = pal.color(QPalette.ColorRole.Highlight).name()
        self.bubble.setStyleSheet(
            f"#TourBubble {{ background: {base}; color: {text}; border: 2px solid {accent}; "
            f"border-radius: 10px; }} #TourBubble QLabel {{ color: {text}; }}")
        v = QVBoxLayout(self.bubble)
        v.setContentsMargins(16, 14, 16, 12)
        v.setSpacing(8)
        head = QHBoxLayout()
        self.title = QLabel()
        self.title.setStyleSheet("font-weight: bold; font-size: 15px;")
        self.title.setWordWrap(True)
        self.counter = QLabel()
        self.counter.setStyleSheet("color: gray;")
        head.addWidget(self.title, 1)
        head.addWidget(self.counter, 0, Qt.AlignmentFlag.AlignTop)
        v.addLayout(head)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self.text)
        row = QHBoxLayout()
        self.skip_button = QPushButton("Skip Tour")
        self.skip_button.setFlat(True)
        self.back_button = QPushButton("Back")
        self.next_button = QPushButton("Next")
        for b in (self.skip_button, self.back_button, self.next_button):
            b.setAutoDefault(False)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # keys go to the tour
        self.skip_button.clicked.connect(lambda: self.close_tour(False))
        self.back_button.clicked.connect(self.back)
        self.next_button.clicked.connect(self.next)
        row.addWidget(self.skip_button)
        row.addStretch(1)
        row.addWidget(self.back_button)
        row.addWidget(self.next_button)
        v.addLayout(row)
        self.bubble.setFixedWidth(BUBBLE_WIDTH)

    # ---------- steps ----------
    def start(self):
        logger.info("Tour %r started (%d steps)", self.name, len(self.steps))
        self.show()
        self.raise_()
        self.setFocus()
        self.show_step(0)
        return self

    def show_step(self, index):
        self.index = max(0, min(index, len(self.steps) - 1))
        step = self.steps[self.index]
        if step.prepare is not None:
            try:
                step.prepare(self.window_)
            except Exception:
                logger.exception("Tour step %r: preparing failed", step.title)
            QTimer.singleShot(0, self._layout)       # panels just opened settle first
        self.title.setText(step.title)
        self.text.setText(step.text)
        self.counter.setText(f"{self.index + 1} / {len(self.steps)}")
        self.back_button.setEnabled(self.index > 0)
        last = self.index == len(self.steps) - 1
        self.next_button.setText("Finish" if last else "Next")
        self.skip_button.setVisible(not last)
        self._layout()
        logger.debug("Tour %r step %d: %s", self.name, self.index + 1, step.title)

    def next(self):
        if self.index >= len(self.steps) - 1:
            self.close_tour(True)
        else:
            self.show_step(self.index + 1)

    def back(self):
        if self.index > 0:
            self.show_step(self.index - 1)

    def close_tour(self, completed):
        logger.info("Tour %r %s at step %d of %d", self.name,
                    "finished" if completed else "closed", self.index + 1, len(self.steps))
        self.window_.removeEventFilter(self)
        self.hide()
        self.finished.emit(completed)
        self.deleteLater()

    # ---------- geometry ----------
    def target_rect(self, step):
        """The target's rectangle in window coordinates, or None."""
        if step.target is None:
            return None
        try:
            target = step.target(self.window_)
        except Exception:
            logger.exception("Tour step %r: target not found", step.title)
            return None
        if isinstance(target, tuple) and len(target) == 2 and target[0] == "menu":
            bar = self.window_.menuBar()
            for action in bar.actions():
                if action.text().replace("&", "") == target[1]:
                    r = bar.actionGeometry(action)
                    if r.isEmpty():
                        return None
                    return QRect(bar.mapTo(self.window_, r.topLeft()), r.size())
            return None
        widgets = target if isinstance(target, (list, tuple)) else [target]
        rect = None
        for widget in widgets:
            if widget is None or not widget.isVisible():
                continue
            r = QRect(widget.mapTo(self.window_, QPoint(0, 0)), widget.size())
            rect = r if rect is None else rect.united(r)
        return rect

    def _layout(self):
        if not self.isVisible():
            return
        self.setGeometry(self.window_.rect())
        self.raise_()
        target = self.target_rect(self.steps[self.index])
        self.hole = (target.adjusted(-HOLE_PADDING, -HOLE_PADDING, HOLE_PADDING, HOLE_PADDING)
                     .intersected(self.rect()) if target is not None else None)
        self.bubble.setFixedWidth(min(BUBBLE_WIDTH, max(200, self.width() - 24)))
        self.bubble.adjustSize()
        self.bubble.move(self.bubble_position(self.hole, self.bubble.size()))
        self.update()

    def bubble_position(self, hole, size):
        """Next to the hole (right, below, left, above), inside the window; in
        the middle without a hole; inside a hole that fills most of the window."""
        W, H, bw, bh, gap = self.width(), self.height(), size.width(), size.height(), 14
        area = QRect(0, 0, W, H)
        if hole is None:
            return QPoint(max(0, (W - bw) // 2), max(0, (H - bh) // 2))
        c = hole.center()

        def fit_x(x):
            return min(max(x, 8), max(8, W - bw - 8))

        def fit_y(y):
            return min(max(y, 8), max(8, H - bh - 8))
        candidates = [
            QPoint(hole.right() + gap, fit_y(c.y() - bh // 2)),        # right
            QPoint(fit_x(c.x() - bw // 2), hole.bottom() + gap),       # below
            QPoint(hole.left() - gap - bw, fit_y(c.y() - bh // 2)),    # left
            QPoint(fit_x(c.x() - bw // 2), hole.top() - gap - bh),     # above
        ]
        if hole.top() < 40:                 # a menu title: below, not over its neighbours
            candidates.insert(0, candidates.pop(1))
        for p in candidates:
            box = QRect(p.x(), p.y(), bw, bh)
            if area.contains(box) and not box.intersects(hole):
                return p
        # The target is too big to stand next to: inside it, at the bottom
        return QPoint(min(max(c.x() - bw // 2, 8), max(8, W - bw - 8)),
                      max(8, min(hole.bottom() - bh - 16, H - bh - 8)))

    # ---------- painting and input ----------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRect(QRectF(self.rect()))
        if self.hole is not None:
            hole = QPainterPath()
            hole.addRoundedRect(QRectF(self.hole), 8, 8)
            path = path.subtracted(hole)
        painter.fillPath(path, DIM)
        if self.hole is not None:
            pen = QPen(self.palette().color(QPalette.ColorRole.Highlight), 3)
            painter.setPen(pen)
            painter.drawRoundedRect(QRectF(self.hole), 8, 8)

    def mousePressEvent(self, event):
        event.accept()                     # read-only: the window can't be used meanwhile

    def mouseReleaseEvent(self, event):
        event.accept()

    def wheelEvent(self, event):
        event.accept()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.next()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Backspace):
            self.back()
        elif key == Qt.Key.Key_Escape:
            self.close_tour(False)
        event.accept()

    def eventFilter(self, obj, event):
        if obj is self.window_ and event.type() in (QEvent.Type.Resize, QEvent.Type.LayoutRequest):
            self._layout()
        return False


# --------------------------------------------------------------------------
# Starting
# --------------------------------------------------------------------------
def start_tour(window, key):
    """Run tour `key` (see TOURS) over `window`; returns the overlay."""
    current = getattr(window, "tour", None)
    if current is not None:
        try:
            current.close_tour(False)
        except RuntimeError:
            pass
    name, steps = TOURS[key]
    overlay = TourOverlay(window, steps, name)
    window.tour = overlay
    overlay.finished.connect(lambda _done: setattr(window, "tour", None))
    return overlay.start()


def tour_running(window) -> bool:
    overlay = getattr(window, "tour", None)
    try:
        return overlay is not None and overlay.isVisible()
    except RuntimeError:
        return False


def offer_first_tour(window, settings=None):
    """The first time a project opens: offer Getting Started once."""
    settings = settings or QSettings(SETTINGS_ORG, SETTINGS_ORG)
    if settings.value(KEY_OFFERED, False, type=bool):
        return False
    settings.setValue(KEY_OFFERED, True)
    box = QMessageBox(QMessageBox.Icon.Question, "Welcome to SegmentME",
                      "New to SegmentME? Take the two-minute tour of the window and "
                      "the drawing tools.", parent=window)
    box.setInformativeText("You can start it later from Help → Tours.")
    start = box.addButton("Start Tour", QMessageBox.ButtonRole.AcceptRole)
    later = box.addButton("Not Now", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(start)
    box.setEscapeButton(later)
    box.exec()
    if box.clickedButton() is start:
        start_tour(window, "getting_started")
        return True
    logger.info("First-start tour declined")
    return False
