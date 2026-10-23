"""
Actions > Class Colours…: every class with its colour and number of masks;
clicking a colour picks a new one (palette or Custom…). A colour that looks
almost like another class's is allowed after a warning. The change shows
at once on the image, in the table and the statistics, and is saved in the
project (database and .SEproj).
"""
from functools import partial

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog, QDialog, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from services import colors
from services.logger import get_logger

logger = get_logger(__name__)


def _swatch_style(hex_color, size=22, selected=False):
    border = "3px solid palette(highlight)" if selected else "1px solid rgba(128, 128, 128, 160)"
    return (f"QPushButton {{ background: {hex_color}; border: {border}; border-radius: 4px; "
            f"min-width: {size}px; min-height: {size}px; max-width: {size}px; max-height: {size}px; }}")


class ColorPicker(QDialog):
    """The palette as a grid plus Custom…. Colours another class already uses
    are marked with a dot (and named in their tooltip)."""

    def __init__(self, parent, class_name, current, used):
        super().__init__(parent)
        self.setWindowTitle(f"Colour of {class_name}")
        self.chosen = None
        used_by = {}
        for name, hex_color in used.items():
            used_by.setdefault(QColor(hex_color).name(), []).append(name)

        v = QVBoxLayout(self)
        v.addWidget(QLabel(f"Pick a colour for <b>{class_name}</b>:"))
        grid = QGridLayout()
        grid.setSpacing(6)
        self.buttons = {}
        for i, hex_color in enumerate(colors.PALETTE):
            btn = QPushButton("•" if hex_color in used_by else "")
            btn.setStyleSheet(_swatch_style(hex_color, 28, selected=hex_color == QColor(current).name())
                              + f"QPushButton {{ color: {_contrast(hex_color)}; font-weight: bold; }}")
            tip = hex_color
            if hex_color in used_by:
                tip += "  (used by " + ", ".join(used_by[hex_color]) + ")"
            btn.setToolTip(tip)
            btn.clicked.connect(partial(self._choose, hex_color))
            grid.addWidget(btn, i // 6, i % 6)
            self.buttons[hex_color] = btn
        v.addLayout(grid)

        row = QHBoxLayout()
        custom = QPushButton("Custom…")
        custom.clicked.connect(lambda: self._custom(current))
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(custom)
        row.addStretch(1)
        row.addWidget(cancel)
        v.addLayout(row)

    def _choose(self, hex_color):
        self.chosen = hex_color
        self.accept()

    def _custom(self, current):
        color = QColorDialog.getColor(QColor(current), self, "Custom colour")
        if color.isValid():
            self._choose(color.name())


def _contrast(hex_color):
    c = QColor(hex_color)
    return "black" if (0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()) > 140 else "white"


class ClassColorsDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window_ = window
        self.setWindowTitle("Class Colours")
        self.setMinimumWidth(380)
        v = QVBoxLayout(self)
        intro = QLabel("Click a colour to change it. The image, table and statistics "
                       "update at once.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        self.rows = QWidget()
        self.rows_layout = QGridLayout(self.rows)
        self.rows_layout.setColumnStretch(1, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.rows)
        v.addWidget(scroll, 1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        v.addLayout(row)
        self.swatches = {}
        self._fill()

    def _classes(self):
        """[(name, "#rrggbb"), ...] in the project's class order."""
        return [(name, QColor(color).name())
                for _id, name, color in self.window_.state_manager.class_manager.list_classes()]

    def _fill(self):
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.swatches = {}
        classes = self._classes()
        if not classes:
            self.rows_layout.addWidget(QLabel("No classes yet (add one in the sidebar)."), 0, 0)
            return
        counts = dict(self.window_.state_manager.mask_manager.counts_by_class())
        for r, (name, hex_color) in enumerate(classes):
            swatch = QPushButton()
            swatch.setStyleSheet(_swatch_style(hex_color))
            swatch.setCursor(Qt.CursorShape.PointingHandCursor)
            swatch.setToolTip(f"{hex_color}: click to change")
            swatch.clicked.connect(partial(self.change, name))
            self.rows_layout.addWidget(swatch, r, 0)
            self.rows_layout.addWidget(QLabel(name), r, 1)
            n = counts.get(name, 0)
            count = QLabel(f"{n:,} mask{'s' if n != 1 else ''}")
            count.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.rows_layout.addWidget(count, r, 2)
            self.swatches[name] = swatch
        self.rows_layout.setRowStretch(len(classes), 1)

    def sizeHint(self):
        return QSize(420, min(160 + 34 * len(self._classes()), 560))

    def change(self, class_name):
        """Pick a new colour for a class (palette or custom) and apply it."""
        current = dict(self._classes())
        old = current.get(class_name)
        others = {n: c for n, c in current.items() if n != class_name}
        picker = ColorPicker(self, class_name, old, others)
        if picker.exec() != QDialog.DialogCode.Accepted or not picker.chosen:
            return
        new = QColor(picker.chosen).name()
        if new == old:
            return
        if not self.confirm_similar(class_name, new, others):
            return
        self.window_.change_class_color(class_name, new)
        self._fill()

    def confirm_similar(self, class_name, new, others) -> bool:
        """Warn when `new` looks almost like another class's colour."""
        twin, distance = colors.closest(new, others)
        if twin is None or distance >= colors.SIMILAR_DELTA_E:
            return True
        same = "the same as" if distance < 2 else "very close to"
        answer = QMessageBox.question(
            self, "Similar colour",
            f"This colour is {same} the colour of '{twin}', so their masks will be hard "
            f"to tell apart.\n\nUse it for '{class_name}' anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes
