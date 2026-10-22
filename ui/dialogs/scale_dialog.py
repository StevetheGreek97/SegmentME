"""
Actions > Set Scale…: how big a pixel is (core/measure/scale.py).

Images that store a real resolution (scans: 2400 DPI) use it; for the others
the project scale applies, set by drawing along a ruler on the image or by
typing a value.
"""
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from core.measure.scale import UNITS, ProjectScale, describe, file_um_per_px
from services.logger import get_logger
from ui.elements.image_display.ruler_mode import RulerMode

logger = get_logger(__name__)

# How a typed value is given -> converts the number to micrometres per pixel
_TYPED = {
    "µm per pixel": lambda v: v,
    "pixels per mm": lambda v: 1000.0 / v,
    "pixels per cm": lambda v: 10000.0 / v,
    "DPI (pixels per inch)": lambda v: 25400.0 / v,
}


class ScaleDialog(QDialog):
    def __init__(self, main_window, scale: ProjectScale):
        super().__init__(main_window)
        self.setWindowTitle("Set Scale")
        self._main = main_window
        self._image = main_window.state_manager.current_image_path
        self._um = scale.um_per_px
        self._source = scale.source

        # --- the current image's own resolution ---------------------------
        own = file_um_per_px(self._image) if self._image else None
        name = os.path.basename(self._image) if self._image else "no image"
        self.file_info = QLabel(
            f"<b>{name}</b> stores: {describe(own)}" if own else
            f"<b>{name}</b> stores no resolution (or only a 72/96 DPI default).")
        self.file_info.setWordWrap(True)
        self.use_file = QCheckBox("Use the resolution stored in each image file (DPI) when it has one")
        self.use_file.setChecked(scale.use_file_dpi)
        self.use_file.toggled.connect(self._update)

        # --- project scale for the other images ---------------------------
        self.value = QDoubleSpinBox()
        self.value.setDecimals(4)
        self.value.setRange(0.0, 1e7)
        self.value.setSpecialValueText("not set")
        self.how = QComboBox()
        self.how.addItems(list(_TYPED))
        if self._um:
            self.value.setValue(self._um)
        self.value.valueChanged.connect(self._typed)
        self.how.currentIndexChanged.connect(self._typed)
        ruler = QPushButton("Draw on the image…")
        ruler.setToolTip("Drag along a ruler or anything of known length, then type its length")
        ruler.clicked.connect(self._draw)
        ruler.setEnabled(bool(self._image))
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self._set_project(None, ""))

        row = QHBoxLayout()
        row.addWidget(self.value, 1)
        row.addWidget(self.how)
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(ruler)
        buttons_row.addWidget(clear)
        buttons_row.addStretch(1)
        self.project_info = QLabel()
        self.project_info.setWordWrap(True)
        box = QGroupBox("Scale for images without a stored resolution")
        inner = QVBoxLayout(box)
        inner.addLayout(row)
        inner.addLayout(buttons_row)
        inner.addWidget(self.project_info)

        # --- units + scale bar ----------------------------------------------
        self.unit = QComboBox()
        self.unit.addItems(list(UNITS))
        self.unit.setCurrentText(scale.unit)
        self.show_bar = QCheckBox("Show a scale bar on the image")
        self.show_bar.setChecked(scale.show_bar)
        form = QFormLayout()
        form.addRow("Lengths in", self.unit)
        form.addRow("", self.show_bar)

        self.result = QLabel()
        self.result.setWordWrap(True)
        self.result.setTextFormat(Qt.TextFormat.RichText)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.file_info)
        layout.addWidget(self.use_file)
        layout.addWidget(box)
        layout.addLayout(form)
        layout.addWidget(self.result)
        layout.addWidget(buttons)
        self.setMinimumWidth(520)
        self._file_um = own
        self._update()

    # ------------------------------------------------------------- values
    def _typed(self, *_):
        v = self.value.value()
        if v <= 0:
            self._set_project(None, "", update_field=False)
            return
        self._set_project(_TYPED[self.how.currentText()](v), "typed", update_field=False)

    def _set_project(self, um, source, update_field=True):
        self._um, self._source = um, source
        if update_field:
            self.value.blockSignals(True)
            self.how.blockSignals(True)
            self.how.setCurrentIndex(0)
            self.value.setValue(um or 0.0)
            self.value.blockSignals(False)
            self.how.blockSignals(False)
        self._update()

    def _update(self, *_):
        self.project_info.setText(describe(self._um) if self._um else
                                  "Not set: images without a stored resolution are measured in pixels.")
        if self.use_file.isChecked() and self._file_um:
            um, why = self._file_um, "from the image file"
        elif self._um:
            um, why = self._um, "project scale"
        else:
            um, why = None, ""
        self.result.setText(f"Current image: <b>{describe(um)}</b>" + (f" — {why}" if why else ""))

    def scale(self, old: ProjectScale) -> ProjectScale:
        """The settings chosen here (keeps the file-resolution cache)."""
        old.um_per_px = self._um
        old.source = self._source if self._um else ""
        old.use_file_dpi = self.use_file.isChecked()
        old.unit = self.unit.currentText()
        old.show_bar = self.show_bar.isChecked()
        return old

    # ------------------------------------------------------------- ruler
    def _draw(self):
        """Hide this window, let the user drag a line on the image, then ask
        how long it is."""
        display = self._main.image_display
        logger.debug("Ruler started on %s", self._image)
        self.hide()
        self._main.statusBar().showMessage(
            "Drag along a known length on the image (Esc to cancel)…")
        display.setFocus()
        display.start_edit_mode(RulerMode(display, self._ruler_done))

    def _ruler_done(self, length_px):
        self._main.statusBar().clearMessage()
        self.show()
        self.raise_()
        if not length_px:
            logger.debug("Ruler cancelled")
            return
        real = KnownLengthDialog(self, length_px)
        if real.exec() == QDialog.DialogCode.Accepted and real.um() > 0:
            um = real.um() / length_px
            logger.info("Ruler on %s: %.1f px = %g %s -> %s", os.path.basename(self._image or ""),
                        length_px, real.length.value(), real.unit.currentText(), describe(um))
            self._set_project(um, "ruler")
        else:
            logger.debug("Ruler: no known length entered (%.1f px line)", length_px)


class KnownLengthDialog(QDialog):
    """'That line is 946.3 px long. Real length: [5] [mm]'."""

    def __init__(self, parent, length_px):
        super().__init__(parent)
        self.setWindowTitle("Known length")
        self.length = QDoubleSpinBox()
        self.length.setDecimals(3)
        self.length.setRange(0.0, 1e6)
        self.length.setValue(1.0)
        self.unit = QComboBox()
        self.unit.addItems(list(UNITS))
        row = QHBoxLayout()
        row.addWidget(self.length, 1)
        row.addWidget(self.unit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"The line is {length_px:.1f} pixels long. How long is it really?"))
        layout.addLayout(row)
        layout.addWidget(buttons)
        self.length.selectAll()

    def um(self) -> float:
        return self.length.value() * UNITS[self.unit.currentText()]
