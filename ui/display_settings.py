"""
How masks and the image look (Settings > Display…): mask fill opacity,
outlines only (V), classes shown or hidden, and OpenCV image adjustments:
brightness, contrast, gamma, saturation, auto-contrast (CLAHE), sharpening,
invert and single colour channels.

Everything is view only -- the saved image and masks never change, and the
drawing tools still work on the original image. Everything is remembered
for the whole app, except hidden classes: they are shown again when
SegmentME starts, so masks never seem to have "disappeared" for good.
"""
from functools import partial

from PyQt6.QtCore import QObject, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSlider, QVBoxLayout, QWidget,
)

from services.logger import get_logger
from ui.elements.image_display.mask_render import ALPHA, DisplayStyle

logger = get_logger(__name__)

SETTINGS_ORG = "SegmentME"
KEYS = {"fill": "display/fill_percent", "outlines_only": "display/outlines_only",
        "brightness": "display/brightness", "contrast": "display/contrast",
        "gamma": "display/gamma", "saturation": "display/saturation",
        "auto_contrast": "display/auto_contrast", "sharpen": "display/sharpen",
        "invert": "display/invert", "channel": "display/channel"}
FILL_DEFAULT = round(ALPHA * 100)
BRIGHTNESS_RANGE = (-100, 100)
CONTRAST_RANGE = (-75, 100)
GAMMA_RANGE = (30, 300)            # x100
SATURATION_RANGE = (-100, 100)
SHARPEN_RANGE = (0, 100)
CHANNEL_LABELS = {"all": "All colours", "red": "Red only", "green": "Green only",
                  "blue": "Blue only"}
# name: (default, (low, high) or None for on/off)
IMAGE_SETTINGS = {"brightness": (0, BRIGHTNESS_RANGE), "contrast": (0, CONTRAST_RANGE),
                  "gamma": (100, GAMMA_RANGE), "saturation": (0, SATURATION_RANGE),
                  "sharpen": (0, SHARPEN_RANGE), "auto_contrast": (False, None),
                  "invert": (False, None)}


def _clamp(value, low, high):
    return max(low, min(high, int(value)))


class DisplaySettings(QObject):
    """The window's display settings (window.display_settings)."""

    changed = pyqtSignal()

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.settings = settings or QSettings(SETTINGS_ORG, SETTINGS_ORG)
        v = self.settings.value
        self.fill_percent = _clamp(v(KEYS["fill"], FILL_DEFAULT, type=int), 0, 100)
        self.outlines_only = v(KEYS["outlines_only"], False, type=bool)
        for name, (default, limits) in IMAGE_SETTINGS.items():
            if limits is None:
                value = v(KEYS[name], default, type=bool)
            else:
                value = _clamp(v(KEYS[name], default, type=int), *limits)
            setattr(self, name, value)
        channel = v(KEYS["channel"], "all", type=str)
        self.channel = channel if channel in CHANNEL_LABELS else "all"
        self.hidden = frozenset()

    def style(self) -> DisplayStyle:
        return DisplayStyle(fill=self.fill_percent / 100.0, outlines_only=self.outlines_only,
                            hidden=self.hidden, channel=self.channel,
                            **{name: getattr(self, name) for name in IMAGE_SETTINGS})

    def _set(self, name, value, key=None):
        if getattr(self, name) == value:
            return
        setattr(self, name, value)
        if key:
            self.settings.setValue(key, value)
        logger.info("Display: %s = %s", name, sorted(value) if name == "hidden" else value)
        self.changed.emit()

    def set_fill(self, percent):
        self._set("fill_percent", _clamp(percent, 0, 100), KEYS["fill"])

    def set_outlines_only(self, on):
        self._set("outlines_only", bool(on), KEYS["outlines_only"])

    def toggle_outlines_only(self):
        self.set_outlines_only(not self.outlines_only)

    def set_brightness(self, value):
        self._set("brightness", _clamp(value, *BRIGHTNESS_RANGE), KEYS["brightness"])

    def set_contrast(self, value):
        self._set("contrast", _clamp(value, *CONTRAST_RANGE), KEYS["contrast"])

    def set_gamma(self, value):
        self._set("gamma", _clamp(value, *GAMMA_RANGE), KEYS["gamma"])

    def set_saturation(self, value):
        self._set("saturation", _clamp(value, *SATURATION_RANGE), KEYS["saturation"])

    def set_sharpen(self, value):
        self._set("sharpen", _clamp(value, *SHARPEN_RANGE), KEYS["sharpen"])

    def set_auto_contrast(self, on):
        self._set("auto_contrast", bool(on), KEYS["auto_contrast"])

    def set_invert(self, on):
        self._set("invert", bool(on), KEYS["invert"])

    def set_channel(self, channel):
        if channel in CHANNEL_LABELS:
            self._set("channel", channel, KEYS["channel"])

    def reset_image(self):
        """The image as it is (every adjustment off)."""
        for name, (default, _limits) in IMAGE_SETTINGS.items():
            getattr(self, f"set_{name}")(default)
        self.set_channel("all")

    def set_class_visible(self, name, visible):
        hidden = set(self.hidden)
        (hidden.discard if visible else hidden.add)(name)
        self._set("hidden", frozenset(hidden))

    def show_all_classes(self):
        self._set("hidden", frozenset())

    def reset(self):
        """Back to the defaults (all classes shown)."""
        self.set_fill(FILL_DEFAULT)
        self.set_outlines_only(False)
        self.reset_image()
        self.show_all_classes()


class DisplayDialog(QDialog):
    """Settings > Display…: stays open beside the image; changes show at once."""

    def __init__(self, window):
        super().__init__(window)
        self.window_ = window
        self.display = window.display_settings
        self.setWindowTitle("Display")
        self.setModal(False)
        self.setMinimumWidth(360)
        v = QVBoxLayout(self)

        masks = QGroupBox("Masks")
        form = QFormLayout(masks)
        self.fill = self._slider(0, 100, self.display.fill_percent, self.display.set_fill)
        self.fill_label = QLabel()
        form.addRow("Fill opacity", self._with_label(self.fill, self.fill_label))
        self.outlines = QCheckBox("Outlines only (V)")
        self.outlines.setToolTip("No colour fill, outlines in the class colour: see the edges "
                                 "and the image under the masks. The V key switches it too.")
        self.outlines.toggled.connect(self.display.set_outlines_only)
        form.addRow("", self.outlines)
        v.addWidget(masks)

        image = QGroupBox("Image (view only, the file is not changed)")
        form = QFormLayout(image)
        self.brightness = self._slider(*BRIGHTNESS_RANGE, self.display.brightness,
                                       self.display.set_brightness)
        self.brightness_label = QLabel()
        form.addRow("Brightness", self._with_label(self.brightness, self.brightness_label))
        self.contrast = self._slider(*CONTRAST_RANGE, self.display.contrast,
                                     self.display.set_contrast)
        self.contrast_label = QLabel()
        form.addRow("Contrast", self._with_label(self.contrast, self.contrast_label))
        self.gamma = self._slider(*GAMMA_RANGE, self.display.gamma, self.display.set_gamma)
        self.gamma.setToolTip("Brightens (right) or darkens (left) the mid-tones without "
                              "washing out white or black")
        self.gamma_label = QLabel()
        form.addRow("Gamma", self._with_label(self.gamma, self.gamma_label))
        self.saturation = self._slider(*SATURATION_RANGE, self.display.saturation,
                                       self.display.set_saturation)
        self.saturation.setToolTip("Stronger colours (right) or greyer, down to black and "
                                   "white (left)")
        self.saturation_label = QLabel()
        form.addRow("Saturation", self._with_label(self.saturation, self.saturation_label))
        self.sharpen = self._slider(*SHARPEN_RANGE, self.display.sharpen,
                                    self.display.set_sharpen)
        self.sharpen.setToolTip("Crisper edges, legs and antennae")
        self.sharpen_label = QLabel()
        form.addRow("Sharpen", self._with_label(self.sharpen, self.sharpen_label))
        self.auto_contrast = QCheckBox("Auto-contrast (evens out light and dark areas)")
        self.auto_contrast.setToolTip("OpenCV CLAHE: brings out faint objects in shadows and "
                                      "on bright glare")
        self.auto_contrast.toggled.connect(self.display.set_auto_contrast)
        form.addRow("", self.auto_contrast)
        self.channel = QComboBox()
        for key, label in CHANNEL_LABELS.items():
            self.channel.addItem(label, key)
        self.channel.setToolTip("Show one colour channel as grey: dark objects on a yellow "
                                "trap often stand out best in Blue only")
        self.channel.currentIndexChanged.connect(
            lambda _i: self.display.set_channel(self.channel.currentData()))
        form.addRow("Channel", self.channel)
        self.invert = QCheckBox("Invert (negative)")
        self.invert.toggled.connect(self.display.set_invert)
        form.addRow("", self.invert)
        reset_image = QPushButton("Reset Image")
        reset_image.setAutoDefault(False)
        reset_image.clicked.connect(self.display.reset_image)
        form.addRow("", reset_image)
        v.addWidget(image)

        classes = QGroupBox("Classes shown")
        cv = QVBoxLayout(classes)
        note = QLabel("Untick a class to hide its masks (they can't be clicked while hidden). "
                      "All classes are shown again when SegmentME starts.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        cv.addWidget(note)
        self.class_box = QWidget()
        self.class_rows = QVBoxLayout(self.class_box)
        self.class_rows.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.class_box)
        scroll.setMaximumHeight(200)
        cv.addWidget(scroll)
        show_all = QPushButton("Show All Classes")
        show_all.clicked.connect(self.display.show_all_classes)
        cv.addWidget(show_all, 0, Qt.AlignmentFlag.AlignLeft)
        v.addWidget(classes, 1)

        row = QHBoxLayout()
        reset = QPushButton("Reset to Defaults")
        reset.clicked.connect(self.display.reset)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        for b in (reset, close):
            b.setAutoDefault(False)
        row.addWidget(reset)
        row.addStretch(1)
        row.addWidget(close)
        v.addLayout(row)

        self.class_checks = {}
        self._fill_classes()
        self.sync()
        self.display.changed.connect(self.sync)
        window.state_manager.classes_changed.connect(self._fill_classes)

    @staticmethod
    def _slider(low, high, value, on_change):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.setPageStep(10)
        slider.valueChanged.connect(on_change)
        return slider

    @staticmethod
    def _with_label(slider, label):
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        label.setMinimumWidth(42)
        h.addWidget(label)
        return box

    def _fill_classes(self):
        while self.class_rows.count():
            item = self.class_rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.class_checks = {}
        rows = self.window_.state_manager.class_manager.list_classes()
        if not rows:
            self.class_rows.addWidget(QLabel("No classes yet."))
        for _id, name, color in rows:
            check = QCheckBox(name)
            c = QColor(color)
            check.setStyleSheet(f"QCheckBox::indicator:checked {{ background: {c.name()}; "
                                f"border: 1px solid gray; border-radius: 3px; }}")
            check.setChecked(name not in self.display.hidden)
            check.toggled.connect(partial(self.display.set_class_visible, name))
            self.class_rows.addWidget(check)
            self.class_checks[name] = check
        self.class_rows.addStretch(1)

    def sync(self):
        """Show the current values (V, Reset, Show All change them from outside)."""
        d = self.display
        for widget, value in ((self.fill, d.fill_percent), (self.brightness, d.brightness),
                              (self.contrast, d.contrast), (self.gamma, d.gamma),
                              (self.saturation, d.saturation), (self.sharpen, d.sharpen)):
            if widget.value() != value:
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
        self.fill_label.setText(f"{d.fill_percent}%")
        self.brightness_label.setText(f"{d.brightness:+d}" if d.brightness else "0")
        self.contrast_label.setText(f"{d.contrast:+d}" if d.contrast else "0")
        self.gamma_label.setText(f"{d.gamma / 100:.2f}")
        self.saturation_label.setText(f"{d.saturation:+d}" if d.saturation else "0")
        self.sharpen_label.setText(str(d.sharpen))
        self.saturation.setEnabled(d.channel == "all")      # one channel is grey anyway
        for box, value in ((self.auto_contrast, d.auto_contrast), (self.invert, d.invert)):
            if box.isChecked() != value:
                box.blockSignals(True)
                box.setChecked(value)
                box.blockSignals(False)
        index = self.channel.findData(d.channel)
        if self.channel.currentIndex() != index:
            self.channel.blockSignals(True)
            self.channel.setCurrentIndex(index)
            self.channel.blockSignals(False)
        self.fill.setEnabled(not d.outlines_only)
        if self.outlines.isChecked() != d.outlines_only:
            self.outlines.blockSignals(True)
            self.outlines.setChecked(d.outlines_only)
            self.outlines.blockSignals(False)
        for name, check in self.class_checks.items():
            if check.isChecked() == (name in d.hidden):
                check.blockSignals(True)
                check.setChecked(name not in d.hidden)
                check.blockSignals(False)


def connect_window(window, settings):
    """Wire the window's display settings: a redraw (merged while a slider
    moves) and a status-bar note while classes are hidden."""
    window.display_settings = settings
    timer = QTimer(window)
    timer.setSingleShot(True)
    timer.setInterval(120)                 # dragging a slider redraws a few times a second
    timer.timeout.connect(window.image_display.restyle)
    window._display_timer = timer

    note = QLabel()
    note.setStyleSheet("color: palette(link); padding: 0 6px;")
    note.hide()
    window.statusBar().addPermanentWidget(note)
    window._hidden_classes_note = note

    def changed():
        timer.start()
        hidden = sorted(window.display_settings.hidden)
        if hidden:
            note.setText("Hidden: " + ", ".join(hidden[:3]) + (" …" if len(hidden) > 3 else ""))
            note.setToolTip("Masks of these classes are hidden (Settings > Display…)")
            note.show()
        else:
            note.hide()
    window.display_settings.changed.connect(changed)

    # A class that is deleted or renamed can't stay hidden by name
    def classes_changed():
        names = set(window.state_manager.class_manager.get_all_class_names())
        settings = window.display_settings
        if settings.hidden - names:
            settings._set("hidden", frozenset(settings.hidden & names))
    window.state_manager.classes_changed.connect(classes_changed)
