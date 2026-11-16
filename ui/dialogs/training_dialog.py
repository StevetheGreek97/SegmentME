"""
Train Custom Model: the settings window (Actions -> Train Custom Model).

Two tabs, Training and Augmentation, with plain-language labels and a short
explanation under each group; the YOLO option name is in each tooltip. A
summary at the top shows what the model will learn from (annotated images,
masks per class) and whether the exported dataset is up to date.

The last settings used are remembered (QSettings "training/*"); "Reset to
defaults" brings back YOLO's own. "Start from" can continue from a model
trained earlier in this project or any .pt file the user browses for (a detection-only model is refused: training here
makes segmentation models).
"""
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QRegularExpression, QSettings, Qt
from PyQt6.QtGui import QPalette, QRegularExpressionValidator
from PyQt6.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from core.trainer.trainer_settings import TrainingSettings
from services import hardware, model_store
from services.logger import get_logger

logger = get_logger(__name__)

SETTINGS_ORG = "SegmentME"
SETTINGS_GROUP = "training"
BROWSE = "__browse__"        # itemData of "Browse for a model (.pt)…" in Start from

# (file, label) -- the model the training starts from
MODELS = [
    ("yolo11n-seg.pt", "YOLO11 Nano · fastest"),
    ("yolo11s-seg.pt", "YOLO11 Small"),
    ("yolo11m-seg.pt", "YOLO11 Medium"),
    ("yolo11l-seg.pt", "YOLO11 Large"),
    ("yolo11x-seg.pt", "YOLO11 Extra large · most accurate, slowest"),
    None,
    ("yolov8n-seg.pt", "YOLOv8 Nano"),
    ("yolov8s-seg.pt", "YOLOv8 Small"),
    ("yolov8m-seg.pt", "YOLOv8 Medium"),
    ("yolov8l-seg.pt", "YOLOv8 Large"),
    ("yolov8x-seg.pt", "YOLOv8 Extra large"),
    None,
    ("yolov9c-seg.pt", "YOLOv9 Compact"),
    ("yolov9e-seg.pt", "YOLOv9 Extended"),
]
DEFAULT_MODEL = "yolo11n-seg.pt"

DEFAULTS = {f: getattr(TrainingSettings, f) for f in (
    "patience", "lr0", "lrf", "momentum", "weight_decay", "cls_pw", "hsv_h", "hsv_s", "hsv_v",
    "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "bgr",
    "mosaic", "mixup", "cutmix", "copy_paste", "copy_paste_mode", "auto_augment", "erasing")}
DEFAULTS.update(model=DEFAULT_MODEL, epochs=100, imgsz=640, batch=0, time=0.0,
                optimizer="auto", project="my-model", run="run")

AUGMENT_KEYS = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
                "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix",
                "copy_paste", "copy_paste_mode", "auto_augment", "erasing")

NAME_PATTERN = QRegularExpression(r"[A-Za-z0-9_\-. ]{1,60}")

# Tooltips: Ultralytics' own description of each training argument
# (docs.ultralytics.com/modes/train, "Train Settings" and "Augmentation
# Settings"), shortened only where the docs describe options this app does
# not offer. widget attribute -> (argument, default shown, description)
DOCS = {
    "start_from": ("model", "a new pre-trained model",
                   "Specifies the model file for training. Accepts a path to either a .pt "
                   "pretrained model or a .yaml configuration file. Essential for defining the "
                   "model structure or initializing weights."),
    "model_selector": ("model", "yolo11n-seg.pt",
                       "Specifies the model file for training. Accepts a path to either a .pt "
                       "pretrained model or a .yaml configuration file. Essential for defining "
                       "the model structure or initializing weights."),
    "epoch_box": ("epochs", "100",
                  "Total number of training epochs. Each epoch represents a full pass over the "
                  "entire dataset. Adjusting this value can affect training duration and model "
                  "performance."),
    "patience_box": ("patience", "100",
                     "Number of epochs to wait without improvement in validation metrics before "
                     "early stopping the training. Helps prevent overfitting by stopping training "
                     "when performance plateaus."),
    "time_limit": ("time", "None",
                   "Maximum training time in hours. If set, this overrides the epochs argument, "
                   "allowing training to automatically stop after the specified duration. Useful "
                   "for time-constrained training scenarios."),
    "imgsz_box": ("imgsz", "640",
                  "Target image size for training. Images are resized to squares with sides "
                  "equal to the specified value, preserving aspect ratio. Affects model accuracy "
                  "and computational complexity."),
    "batch_box": ("batch", "Auto (-1)",
                  "Batch size, with three modes: set as an integer (e.g., batch=16), auto mode "
                  "for 60% of available GPU memory (batch=-1), or auto mode with a specified "
                  "fraction of available GPU memory (batch=0.70)."),
    "device_label": ("device", "chosen automatically",
                     "Specifies the computational device(s) for training: a single GPU "
                     "(device=0), multiple GPUs (device=[0,1]), CPU (device=cpu), or MPS for "
                     "Apple silicon (device=mps)."),
    "project_name": ("project", "my-model",
                     "Name of the project directory where training outputs are saved. Allows "
                     "for organized storage of different experiments."),
    "run_name": ("name", "run",
                 "Name of the training run. Used for creating a subdirectory within the project "
                 "folder, where training logs and outputs are stored."),
    "optimizer_input": ("optimizer", "auto",
                        "Choice of optimizer for training. Options include SGD, Adam, AdamW, "
                        "NAdam, RAdam, RMSProp etc., or auto for automatic selection based on "
                        "model configuration. Affects convergence speed and stability."),
    "lr0": ("lr0", "0.01",
            "Initial learning rate (e.g. SGD=1E-2, Adam=1E-3). Ignored under the default "
            "optimizer='auto'; name an optimizer explicitly to use it."),
    "lrf": ("lrf", "0.01",
            "Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in "
            "conjunction with schedulers to adjust the learning rate over time."),
    "momentum": ("momentum", "0.937",
                 "SGD momentum or beta1 for Adam optimizers, influencing the incorporation of "
                 "past gradients in the current update."),
    "weight_decay": ("weight_decay", "0.0005",
                     "L2 regularization term, penalizing large weights to prevent overfitting."),
    "cls_pw": ("cls_pw", "0.0 (off) · range 0.0 – 1.0",
               "Power for class weighting to handle class imbalance using inverse class "
               "frequency. 0.0 disables class weighting, 1.0 applies full inverse frequency "
               "weighting. Values between 0 and 1 provide partial weighting."),
    "hsv_h": ("hsv_h", "0.015 · range 0.0 – 1.0",
              "Adjusts the hue of the image by a fraction of the color wheel, introducing color "
              "variability. Helps the model generalize across different lighting conditions."),
    "hsv_s": ("hsv_s", "0.7 · range 0.0 – 1.0",
              "Alters the saturation of the image by a fraction, affecting the intensity of "
              "colors. Useful for simulating different environmental conditions."),
    "hsv_v": ("hsv_v", "0.4 · range 0.0 – 1.0",
              "Modifies the value (brightness) of the image by a fraction, helping the model to "
              "perform well under various lighting conditions."),
    "degrees": ("degrees", "0 · range 0.0 – 180",
                "Rotates the image randomly within the specified degree range, improving the "
                "model's ability to recognize objects at various orientations."),
    "translate": ("translate", "0.1 · range 0.0 – 1.0",
                  "Translates the image horizontally and vertically by a fraction of the image "
                  "size, aiding in learning to detect partially visible objects."),
    "scale": ("scale", "0.5 · range 0 – 1",
              "Scales the image by a gain factor, simulating objects at different distances "
              "from the camera."),
    "shear": ("shear", "0 · range -180 – +180",
              "Shears the image by a specified degree, mimicking the effect of objects being "
              "viewed from different angles."),
    "perspective": ("perspective", "0 · range 0.0 – 0.001",
                    "Applies a random perspective transformation to the image, enhancing the "
                    "model's ability to understand objects in 3D space."),
    "flipud": ("flipud", "0 · range 0.0 – 1.0",
               "Flips the image upside down with the specified probability, increasing the data "
               "variability without affecting the object's characteristics."),
    "fliplr": ("fliplr", "0.5 · range 0.0 – 1.0",
               "Flips the image left to right with the specified probability, useful for "
               "learning symmetrical objects and increasing dataset diversity."),
    "bgr": ("bgr", "0 · range 0.0 – 1.0",
            "Flips the image channels from RGB to BGR with the specified probability, useful "
            "for increasing robustness to incorrect channel ordering."),
    "mosaic": ("mosaic", "1 · range 0.0 – 1.0",
               "Combines four training images into one, simulating different scene compositions "
               "and object interactions. Effective for complex scene understanding."),
    "mixup": ("mixup", "0 · range 0.0 – 1.0",
              "Blends two images and their labels, creating a composite image. Enhances the "
              "model's ability to generalize by introducing label noise and visual variability."),
    "cutmix": ("cutmix", "0 · range 0.0 – 1.0",
               "Combines portions of two images, creating a partial blend while maintaining "
               "distinct regions. Enhances model robustness by creating occlusion scenarios."),
    "copy_paste": ("copy_paste", "0 · range 0.0 – 1.0",
                   "Fraction of eligible objects pasted; flip mirrors them within the image, "
                   "while mixup also uses the value as its cross-image application probability."),
    "copy_paste_mode": ("copy_paste_mode", "flip",
                        "Specifies the copy-paste strategy to use. Options include 'flip' and "
                        "'mixup'."),
    "auto_augment": ("auto_augment", "none",
                     "Applies a predefined augmentation policy ('randaugment', 'autoaugment', "
                     "or 'augmix') to enhance model performance through visual diversity."),
    "erasing": ("erasing", "0.4 · range 0.0 – 0.9",
                "Randomly erases regions of the image during training to encourage the model "
                "to focus on less obvious features."),
}


def tooltip_html(argument, default, description):
    return (f"<p style='margin:0'><b>{argument}</b> &nbsp;·&nbsp; default {default}</p>"
            f"<p style='margin-top:4px'>{description}</p>")

STYLE = """
QFrame#summary {
    border: 1px solid palette(mid);
    border-radius: 8px;
    background: palette(base);
}
QFrame#card {             /* outlined, not filled: the input boxes keep their own colour */
    border: 1px solid palette(mid);
    border-radius: 8px;
    background: transparent;
}
QFrame#card QLabel, QFrame#summary QLabel { background: transparent; border: none; }
QLabel[role="section"] { font-weight: 600; font-size: 14px; }
QLabel[role="headline"] { font-weight: 600; font-size: 15px; }
QToolButton#advanced { border: none; font-weight: 600; font-size: 14px; padding: 0;
                       background: transparent; }
"""
NUMBER_WIDTH = 120        # every number box: same width, so the columns line up
TWO_COLUMNS_FROM = 760    # page width (px) from which cards sit side by side


class _CardGrid(QWidget):
    """Cards in two columns when there is room, one column when narrow.
    A wide card always takes the full width."""

    def __init__(self):
        super().__init__()
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(12, 12, 12, 12)
        self._grid.setSpacing(12)
        self._cards = []           # (widget, wide)
        self._columns = None

    def add(self, card, wide=False):
        self._cards.append((card, wide))
        self._flow(self._columns or 2)

    def _flow(self, columns):
        self._columns = columns
        while self._grid.count():
            self._grid.takeAt(0)
        row = col = 0
        for card, wide in self._cards:
            if wide or columns == 1:
                if col:
                    row, col = row + 1, 0
                self._grid.addWidget(card, row, 0, 1, columns)
                row += 1
            else:
                self._grid.addWidget(card, row, col)
                col += 1
                if col == columns:
                    row, col = row + 1, 0
        for c in range(2):
            self._grid.setColumnStretch(c, 1 if c < columns else 0)
        self._grid.setRowStretch(row + 1, 1)

    def resizeEvent(self, event):
        columns = 2 if event.size().width() >= TWO_COLUMNS_FROM else 1
        if columns != self._columns:
            self._flow(columns)
        super().resizeEvent(event)


def _muted(label: QLabel) -> QLabel:
    label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
    label.setWordWrap(True)
    return label


def _double(lo, hi, decimals, step):
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    return box


class TrainingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Train Custom Model")
        self.setStyleSheet(STYLE)
        self.resize(920, 780)
        self.setMinimumSize(560, 560)
        self.project_root = Path(self.parent.state_manager.project_root)
        self.models_dir = str(self.project_root / ".models")
        self._settings = QSettings(SETTINGS_ORG, SETTINGS_ORG)
        self._forms = []                 # every QFormLayout, for the tooltips

        self.device_info = hardware.describe_device()
        self.detected_device = self.device_info.device

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(12)
        layout.addWidget(self._build_summary())

        self.tabs = QTabWidget(self)
        self.training_tab = self._scrolling(self._build_training_tab())
        self.augment_tab = self._scrolling(self._build_augment_tab())
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.augment_tab, "Augmentation")
        layout.addWidget(self.tabs, 1)

        buttons = QHBoxLayout()
        self.reset_button = QPushButton("Reset to defaults")
        self.reset_button.setAutoDefault(False)
        self.reset_button.setToolTip("Put every setting back to YOLO's defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)
        cancel_button = QPushButton("Cancel")
        cancel_button.setAutoDefault(False)
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(cancel_button)
        self.train_button = QPushButton("Start Training")
        self.train_button.setDefault(True)
        self.train_button.clicked.connect(self.validate_and_accept)
        buttons.addWidget(self.train_button)
        layout.addLayout(buttons)

        self._add_tooltips()
        self._load_remembered()
        self.update_output_path()
        self.update_class_weights()
        logger.info("Training dialog opened (device: %s)", self.device_info.label)

    # ------------------------------------------------------------- summary
    def _build_summary(self):
        frame = QFrame(objectName="summary")
        box = QVBoxLayout(frame)
        box.setContentsMargins(14, 12, 14, 12)
        box.setSpacing(4)
        self.summary_headline = QLabel()
        self.summary_headline.setProperty("role", "headline")
        self.summary_classes = QLabel()
        self.summary_classes.setWordWrap(True)
        self.summary_classes.setTextFormat(Qt.TextFormat.RichText)
        self.summary_export = _muted(QLabel())
        for w in (self.summary_headline, self.summary_classes, self.summary_export):
            box.addWidget(w)
        self.refresh_summary()
        return frame

    def dataset_summary(self):
        """(annotated images, all images, [(class, colour, masks)]) from the project."""
        db = self.parent.state_manager.db
        try:
            annotated = db.fetch_one("SELECT COUNT(DISTINCT image_name) FROM masks")[0] or 0
            classes = db.fetch_all(
                "SELECT c.class_name, c.color, COUNT(m.id) FROM classes c "
                "LEFT JOIN masks m ON m.class_name = c.class_name "
                "GROUP BY c.class_name, c.color ORDER BY COUNT(m.id) DESC, c.class_name")
        except Exception:
            logger.exception("Training dialog: could not count the masks")
            annotated, classes = 0, []
        total = len(getattr(self.parent.state_manager, "image_paths", None) or [])
        return annotated, total, [(name, color, int(n)) for name, color, n in classes]

    def refresh_summary(self):
        annotated, total, classes = self.dataset_summary()
        self._classes = classes
        if hasattr(self, "class_weights_label"):
            self.update_class_weights()
        masks = sum(n for _name, _color, n in classes)
        used = [c for c in classes if c[2] > 0]
        self.summary_headline.setText(
            f"{annotated:,} of {total:,} images annotated · {masks:,} masks · "
            f"{len(used)} class{'es' if len(used) != 1 else ''}")
        chips = [f"<span style='color:{color}'>●</span>&nbsp;{name}&nbsp;<b>{n:,}</b>"
                 for name, color, n in used]
        empty = [name for name, _c, n in classes if n == 0]
        text = "&nbsp;&nbsp;&nbsp;".join(chips) or "No masks yet: draw some first."
        if empty:
            text += f"<br><i>No masks (not learned): {', '.join(empty)}</i>"
        self.summary_classes.setText(text)

        hints = [self.export_status()]
        if 0 < annotated < 20:
            hints.append("Few annotated images: expect a rough first model.")
        self.summary_export.setText("\n".join(hints))

    def export_status(self) -> str:
        exported = self.project_root / "data.yaml"
        if not exported.is_file() or self.validate_project_structure(self.project_root):
            return "Not exported yet: you'll be asked to export when you start."
        if self.export_is_outdated():
            return "Masks changed since the last export: you'll be asked to export again."
        when = datetime.fromtimestamp(exported.stat().st_mtime)
        return f"✓ Export is up to date ({when.day} {when:%b}, {when:%H:%M})."

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _scrolling(content):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _form(self):
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Only fields that want to (name boxes) stretch; number boxes and lists keep their size.
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        self._forms.append(form)
        return form

    def _card(self, grid, title, caption=None, wide=False):
        """A titled card on `grid`; returns its form."""
        card = QFrame(objectName="card")
        box = QVBoxLayout(card)
        box.setContentsMargins(16, 12, 16, 14)
        box.setSpacing(6)
        label = QLabel(title)
        label.setProperty("role", "section")
        box.addWidget(label)
        if caption:
            box.addWidget(_muted(QLabel(caption)))
        box.addSpacing(2)
        form = self._form()
        box.addLayout(form)
        box.addStretch(1)
        grid.add(card, wide)
        return form

    def _folding_card(self, grid, title, build):
        """A full-width card that opens when its title is clicked; `build(vbox)` fills it."""
        card = QFrame(objectName="card")
        box = QVBoxLayout(card)
        box.setContentsMargins(16, 10, 16, 10)
        box.setSpacing(6)
        button = QToolButton(objectName="advanced")
        button.setText(title)
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setArrowType(Qt.ArrowType.RightArrow)
        box.addWidget(button)
        body = QWidget()
        inner = QVBoxLayout(body)
        inner.setContentsMargins(0, 4, 0, 4)
        build(inner)
        body.setVisible(False)
        box.addWidget(body)

        def toggle(on):
            body.setVisible(on)
            button.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        button.toggled.connect(toggle)
        grid.add(card, wide=True)
        return button

    @staticmethod
    def _size_fields(page):
        for box in page.findChildren(QAbstractSpinBox):
            box.setFixedWidth(NUMBER_WIDTH)
        for combo in page.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

    def _add_tooltips(self):
        """Ultralytics' description on each field and on its label."""
        tips = {}
        for attr, doc in DOCS.items():
            widget = getattr(self, attr)
            tips[widget] = tooltip_html(*doc)
            widget.setToolTip(tips[widget])
        for form in self._forms:
            for row in range(form.rowCount()):
                field = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
                label = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if field and label and field.widget() in tips and label.widget():
                    label.widget().setToolTip(tips[field.widget()])

    # ------------------------------------------------------- training tab
    def _build_training_tab(self):
        grid = _CardGrid()

        form = self._card(grid, "Model", wide=True)
        self.start_from = QComboBox()
        self.start_from.addItem("A new model (pre-trained on everyday photos)", None)
        for weights in model_store.trained_models(self.project_root):
            run = weights.parent.parent
            when = datetime.fromtimestamp(weights.stat().st_mtime)
            self.start_from.addItem(f"Continue: {run.parent.name}/{run.name} "
                                    f"({when.day} {when:%b}, {when:%H:%M})", str(weights))
        self.start_from.insertSeparator(self.start_from.count())
        self.start_from.addItem("Browse for a model (.pt)…", BROWSE)
        self._start_index = 0
        self.start_from.currentIndexChanged.connect(self._start_from_changed)
        self.start_from.activated.connect(self._start_from_activated)
        form.addRow("Start from:", self.start_from)
        self.model_selector = QComboBox()
        for entry in MODELS:
            if entry is None:
                self.model_selector.insertSeparator(self.model_selector.count())
            else:
                self.model_selector.addItem(entry[1], entry[0])
        self.available_models = [e[0] for e in MODELS if e]
        form.addRow("Model size:", self.model_selector)

        form = self._card(grid, "Length", "One epoch = the model studies every training image once.")
        self.epoch_box = QSpinBox()
        self.epoch_box.setRange(1, 5000)
        form.addRow("Epochs:", self.epoch_box)
        self.patience_box = QSpinBox()
        self.patience_box.setRange(0, 1000)
        self.patience_box.setSpecialValueText("Off")
        form.addRow("Patience (epochs):", self.patience_box)
        self.time_limit = _double(0, 168, 1, 0.5)
        self.time_limit.setSpecialValueText("No limit")
        form.addRow("Time limit (hours):", self.time_limit)

        form = self._card(grid, "Images", "Larger images find small objects better but train slower.")
        self.imgsz_box = QSpinBox()
        self.imgsz_box.setRange(320, 4096)
        self.imgsz_box.setSingleStep(32)
        self.imgsz_box.editingFinished.connect(self._snap_image_size)
        form.addRow("Image size (px):", self.imgsz_box)
        self.batch_box = QSpinBox()
        self.batch_box.setRange(0, 512)
        self.batch_box.setSpecialValueText("Auto")
        form.addRow("Batch size:", self.batch_box)
        info = self.device_info
        colour = "#0ca30c" if info.fast else "#d58a00"
        self.device_label = QLabel(f"<span style='color:{colour}'>●</span>&nbsp;{info.label}")
        form.addRow("Runs on:", self.device_label)
        if not info.fast:
            form.addRow(_muted(QLabel("Training on the CPU is slow: expect hours for 100 "
                                      "epochs. A smaller model or image size helps.")))

        form = self._card(grid, "Class balance", "Makes rare classes count more when the model "
                          "learns which class an object is. 0.3–0.5 suits strongly unbalanced data.")
        self.cls_pw = _double(0.0, 1.0, 2, 0.1)
        self.cls_pw.setSpecialValueText("Off")
        form.addRow("Class weight power:", self.cls_pw)
        self.class_weights_label = _muted(QLabel())
        self.class_weights_label.setTextFormat(Qt.TextFormat.RichText)
        form.addRow(self.class_weights_label)
        self.cls_pw.valueChanged.connect(self.update_class_weights)

        form = self._card(grid, "Save as", "Trained models are kept in the project's trainings "
                          "folder and appear in Actions → Run Inference.")
        validator = QRegularExpressionValidator(NAME_PATTERN, self)
        self.project_name = QLineEdit()
        self.project_name.setValidator(validator)
        self.project_name.setPlaceholderText("my-model")
        form.addRow("Model name:", self.project_name)
        self.run_name = QLineEdit()
        self.run_name.setValidator(validator)
        self.run_name.setPlaceholderText("run")
        form.addRow("Run:", self.run_name)
        self.output_dir_input = _muted(QLabel())
        self.output_dir_input.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Saved to:", self.output_dir_input)
        self.project_name.textChanged.connect(self.update_output_path)
        self.run_name.textChanged.connect(self.update_output_path)

        self.advanced_training_button = self._folding_card(
            grid, "Advanced: optimizer and learning rate", self._build_advanced_training)

        self.data_path_input = QLabel(str(self.project_root / "data.yaml"))   # what get_settings reads
        self._size_fields(grid)
        return grid

    def _build_advanced_training(self, vbox):
        vbox.addWidget(_muted(QLabel("YOLO's defaults suit most datasets. Change these only "
                                     "if you know why.")))
        row = QHBoxLayout()
        row.setSpacing(32)
        left, right = self._form(), self._form()
        self.optimizer_input = QComboBox()
        self.optimizer_input.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        left.addRow("Optimizer:", self.optimizer_input)
        self.lr0 = _double(0.00001, 1.0, 5, 0.001)
        left.addRow("Initial learning rate:", self.lr0)
        self.lrf = _double(0.00001, 1.0, 5, 0.001)
        left.addRow("Final learning rate:", self.lrf)
        self.momentum = _double(0.0, 1.0, 3, 0.01)
        right.addRow("Momentum:", self.momentum)
        self.weight_decay = _double(0.0, 0.1, 5, 0.0001)
        right.addRow("Weight decay:", self.weight_decay)
        row.addLayout(left)
        row.addLayout(right)
        row.addStretch(1)
        vbox.addLayout(row)

    @staticmethod
    def class_weights(counts, power):
        """Ultralytics' cls_pw weights: (1 / count) ** power, scaled to a mean
        of 1 (classes without masks get the mean weight)."""
        counts = [float(n) for n in counts]
        present = [n for n in counts if n > 0]
        if not present or power <= 0:
            return [1.0] * len(counts)
        raw = [(1.0 / n) ** power for n in present]
        mean_present = sum(raw) / len(raw)
        weights = [(1.0 / n) ** power if n > 0 else mean_present for n in counts]
        mean = sum(weights) / len(weights)
        return [w / mean for w in weights]

    def update_class_weights(self):
        """The weight each class would get, from the masks in the project."""
        power = self.cls_pw.value()
        classes = [(name, color, n) for name, color, n in getattr(self, "_classes", []) if n > 0]
        if power <= 0:
            self.class_weights_label.setText("Off: every class counts the same.")
            return
        if len(classes) < 2:
            self.class_weights_label.setText("Needs masks of at least two classes.")
            return
        weights = self.class_weights([n for _name, _c, n in classes], power)
        chips = [f"<span style='color:{color}'>●</span>&nbsp;{name}&nbsp;<b>×{w:.2f}</b>"
                 for (name, color, _n), w in zip(classes, weights)]
        self.class_weights_label.setText("&nbsp;&nbsp;&nbsp;".join(chips) +
                                         "<br>Estimated from all masks; training counts "
                                         "the training images only.")

    def _start_from_changed(self):
        # Continuing keeps the size of the model trained before.
        self.model_selector.setEnabled(self.start_from.currentData() is None)
        if self.start_from.currentData() != BROWSE:
            self._start_index = self.start_from.currentIndex()

    def _start_from_activated(self, index):
        if self.start_from.itemData(index) != BROWSE:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Start From a Model", str(self.project_root),
                                              "PyTorch model (*.pt)")
        if not path or not self.use_custom_model(path):
            self.start_from.setCurrentIndex(self._start_index)   # back to what it was

    def use_custom_model(self, path) -> bool:
        """Select `path` in Start from (added once). False, with a message, when
        it is not a segmentation model."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            task = model_store.yolo_task(path)
        finally:
            QApplication.restoreOverrideCursor()
        if task not in (None, "segment"):
            logger.info("Start from %s refused: a %s model", path, task)
            QMessageBox.warning(
                self, "Not a Segmentation Model",
                f"{Path(path).name} is a {task} model. Training here makes segmentation "
                "models (masks), so it needs a segmentation model (…-seg.pt) to start from.")
            return False
        index = self.start_from.findData(str(path))
        if index < 0:
            index = self.start_from.findData(BROWSE) - 1       # before the separator
            self.start_from.insertItem(index, f"Custom: {Path(path).name}", str(path))
            self.start_from.setItemData(index, str(path), Qt.ItemDataRole.ToolTipRole)
        self.start_from.setCurrentIndex(index)
        logger.info("Start from a custom model: %s (task %s)", path, task or "unknown")
        return True

    def _snap_image_size(self):
        value = self.imgsz_box.value()
        snapped = max(320, int(round(value / 32)) * 32)
        if snapped != value:
            self.imgsz_box.setValue(snapped)

    # ------------------------------------------------------ augmentation tab
    def _build_augment_tab(self):
        grid = _CardGrid()
        intro = _muted(QLabel("During training each image is shown slightly changed (colour, "
                              "position, flipped…), so the model copes with images that are "
                              "not exactly like yours."))
        grid.add(intro, wide=True)

        def prob():
            return _double(0.0, 1.0, 2, 0.05)

        form = self._card(grid, "Colour")
        self.hsv_h = _double(0.0, 1.0, 3, 0.005)
        form.addRow("Hue:", self.hsv_h)
        self.hsv_s = _double(0.0, 1.0, 2, 0.05)
        form.addRow("Saturation:", self.hsv_s)
        self.hsv_v = _double(0.0, 1.0, 2, 0.05)
        form.addRow("Brightness:", self.hsv_v)

        form = self._card(grid, "Flips", "Chance that an image is flipped (0 = never, 1 = always).")
        self.flipud = prob()
        form.addRow("Upside down:", self.flipud)
        self.fliplr = prob()
        form.addRow("Left to right:", self.fliplr)
        self.bgr = prob()
        form.addRow("Swap red/blue:", self.bgr)

        form = self._card(grid, "Position and shape")
        self.degrees = _double(0.0, 180.0, 1, 5.0)
        form.addRow("Rotation (°):", self.degrees)
        self.translate = _double(0.0, 1.0, 2, 0.05)
        form.addRow("Move:", self.translate)
        self.scale = _double(0.0, 1.0, 2, 0.05)
        form.addRow("Zoom:", self.scale)
        self.shear = _double(-180.0, 180.0, 1, 1.0)
        form.addRow("Shear (°):", self.shear)
        self.perspective = _double(0.0, 0.001, 5, 0.0001)
        form.addRow("Perspective:", self.perspective)

        form = self._card(grid, "Mixing images", "Combine several training images into one.")
        self.mosaic = prob()
        form.addRow("Mosaic:", self.mosaic)

        self._folding_card(grid, "More mixing and effects", self._build_advanced_augment)
        self._size_fields(grid)
        return grid

    def _build_advanced_augment(self, vbox):
        row = QHBoxLayout()
        row.setSpacing(32)
        left, right = self._form(), self._form()
        self.mixup = _double(0.0, 1.0, 2, 0.05)
        left.addRow("Mixup:", self.mixup)
        self.cutmix = _double(0.0, 1.0, 2, 0.05)
        left.addRow("CutMix:", self.cutmix)
        self.copy_paste = _double(0.0, 1.0, 2, 0.05)
        left.addRow("Copy-paste objects:", self.copy_paste)
        self.copy_paste_mode = QComboBox()
        self.copy_paste_mode.addItems(["flip", "mixup"])
        right.addRow("Copy-paste mode:", self.copy_paste_mode)
        self.auto_augment = QComboBox()
        self.auto_augment.addItems(["none", "randaugment", "autoaugment", "augmix"])
        right.addRow("Auto augment:", self.auto_augment)
        self.erasing = _double(0.0, 0.9, 2, 0.05)
        right.addRow("Random erasing:", self.erasing)
        row.addLayout(left)
        row.addLayout(right)
        row.addStretch(1)
        vbox.addLayout(row)

    # --------------------------------------------------- values in and out
    def _widgets(self):
        names = ["epochs", "patience", "time", "imgsz", "batch", "optimizer", "lr0", "lrf",
                 "momentum", "weight_decay", "cls_pw", "project", "run", "model", *AUGMENT_KEYS]
        alias = {"epochs": self.epoch_box, "patience": self.patience_box, "time": self.time_limit,
                 "imgsz": self.imgsz_box, "batch": self.batch_box, "optimizer": self.optimizer_input,
                 "project": self.project_name, "run": self.run_name, "model": self.model_selector}
        return {n: alias[n] if n in alias else getattr(self, n) for n in names}

    def _set(self, name, value):
        w = self._widgets()[name]
        try:
            if isinstance(w, QComboBox):
                index = w.findData(value)
                if index < 0:
                    index = w.findText(str(value))
                if index >= 0:
                    w.setCurrentIndex(index)
            elif isinstance(w, QLineEdit):
                w.setText(str(value))
            elif isinstance(w, QSpinBox):
                w.setValue(int(float(value)))
            else:
                w.setValue(float(value))
        except (TypeError, ValueError):
            logger.warning("Training dialog: ignored the remembered %s=%r", name, value)

    def _get(self, name):
        w = self._widgets()[name]
        if isinstance(w, QComboBox):
            return w.currentData() if w.currentData() is not None else w.currentText()
        if isinstance(w, QLineEdit):
            return w.text()
        return w.value()

    def reset_to_defaults(self):
        for name, value in DEFAULTS.items():
            self._set(name, value)
        self.start_from.setCurrentIndex(0)
        logger.info("Training settings reset to the defaults")

    def _load_remembered(self):
        self.reset_to_defaults()
        s = self._settings
        s.beginGroup(SETTINGS_GROUP)
        try:
            for name in self._widgets():
                if s.contains(name):
                    self._set(name, s.value(name))
            start_from = s.value("start_from", "") or ""
        finally:
            s.endGroup()
        if start_from and Path(start_from).is_file():             # still there?
            index = self.start_from.findData(start_from)
            if index >= 0:
                self.start_from.setCurrentIndex(index)
            else:
                self.use_custom_model(start_from)

    def remember(self):
        s = self._settings
        s.beginGroup(SETTINGS_GROUP)
        try:
            for name in self._widgets():
                s.setValue(name, self._get(name))
            s.setValue("start_from", self.start_from.currentData() or "")
        finally:
            s.endGroup()
        s.sync()

    # -------------------------------------------------------------- output
    def _names(self):
        project = self.project_name.text().strip() or DEFAULTS["project"]
        run = self.run_name.text().strip() or DEFAULTS["run"]
        return project, run

    def _output_dir(self) -> Path:
        """The run folder this training will write: run, run1, run2, … (never an old one)."""
        project, run = self._names()
        base = self.project_root / "trainings" / project
        path, counter = base / run, 1
        while path.exists():
            path = base / f"{run}{counter}"
            counter += 1
        return path

    def update_output_path(self):
        path = self._output_dir()
        try:
            shown = path.relative_to(self.project_root)
        except ValueError:
            shown = path
        self.output_dir_input.setText(str(Path(self.project_root.name) / shown))
        self.output_dir_input.setToolTip(str(path))

    # ------------------------------------------------------------- accept
    def validate_and_accept(self):
        """Training learns from the exported dataset (File > Export
        annotations). Offer to export when that was never done, or when the
        masks changed after the last export (it would train on old masks)."""
        self._snap_image_size()
        missing = self.validate_project_structure(self.project_root)
        if missing:
            logger.info("Training needs an export first; missing %s", missing)
            if not self._ask_export(
                    "Export Needed",
                    "Training learns from your annotations exported as a dataset, and they "
                    "have not been exported yet.\n\nExport them now?",
                    "Export Now", None):
                return
        elif self.export_is_outdated():
            answer = self._ask_export(
                "Annotations Changed",
                "You changed masks or classes after the last export, so training would use "
                "the old annotations.\n\nExport them again first?",
                "Export Again", "Train on Old Export")
            if answer is None:                       # Cancel
                return
        if self.validate_project_structure(self.project_root):
            self.refresh_summary()
            return                                   # export cancelled or failed
        self.remember()
        self.accept()

    def _ask_export(self, title, text, export_label, skip_label):
        """True after exporting, False when skipped (skip_label), None on Cancel."""
        box = QMessageBox(QMessageBox.Icon.Question, title, text, parent=self)
        export = box.addButton(export_label, QMessageBox.ButtonRole.AcceptRole)
        skip = box.addButton(skip_label, QMessageBox.ButtonRole.ActionRole) if skip_label else None
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(export)
        box.setEscapeButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is export:
            from ui.flows import data_io
            data_io.export_annotations_dataset(self.parent)
            return True
        if skip is not None and clicked is skip:
            return False
        return None

    def export_is_outdated(self) -> bool:
        """The project database changed after data.yaml was written."""
        exported = self.project_root / "data.yaml"
        db = Path(self.parent.state_manager.db.db_path)
        try:
            export_time = exported.stat().st_mtime
            changed = max(p.stat().st_mtime for p in (db, db.with_name(db.name + "-wal"))
                          if p.exists())
        except (OSError, ValueError):
            return False
        return changed > export_time + 5

    def validate_project_structure(self, root: Path):
        required = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt",
                    "data.yaml", "images", "labels"]
        missing = []
        for item in required:
            path = root / item
            ok = path.is_dir() if item in ("images", "labels") else path.is_file()
            if not ok:
                missing.append(item)
        return missing

    def get_settings(self):
        output_dir = self._output_dir()
        self.update_output_path()
        start_from = self.start_from.currentData()
        model = start_from or str(Path(self.models_dir) / self.model_selector.currentData())
        project, _run = self._names()
        batch = self.batch_box.value()
        settings = TrainingSettings(
            model=model,
            data=str(Path(self.data_path_input.text())),
            epochs=self.epoch_box.value(),
            batch=batch if batch > 0 else -1,
            imgsz=self.imgsz_box.value(),
            device=self.detected_device,
            optimizer=self.optimizer_input.currentText(),
            project=project,
            name=output_dir.name,
            time=self.time_limit.value() or None,
            patience=self.patience_box.value(),
            lr0=self.lr0.value(),
            lrf=self.lrf.value(),
            momentum=self.momentum.value(),
            cls_pw=self.cls_pw.value(),
            weight_decay=self.weight_decay.value(),
            **{k: (getattr(self, k).currentText() if isinstance(getattr(self, k), QComboBox)
                   else getattr(self, k).value()) for k in AUGMENT_KEYS},
            output_dir=str(output_dir),
        )
        logger.info("Training settings: model=%s epochs=%s imgsz=%s batch=%s device=%s -> %s",
                    settings.model, settings.epochs, settings.imgsz, settings.batch,
                    settings.device, output_dir)
        return settings
