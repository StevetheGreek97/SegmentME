"""
File > Save Results…: choose what to export, then write the measurement CSV
(core/measure/export.py) on a worker thread with a progress window.
"""
import os
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QRadioButton, QVBoxLayout,
)

from core.measure import export as measure_export
from core.measure.scale import UNITS, ProjectScale
from services.logger import get_logger
from ui.dialogs.background_task import run_in_background

logger = get_logger(__name__)


class SaveResultsDialog(QDialog):
    def __init__(self, main_window, scale: ProjectScale, classes, current_image_name):
        super().__init__(main_window)
        self.setWindowTitle("Save Results")
        self._main = main_window
        self._current = current_image_name

        self.all_images = QRadioButton("All images")
        self.current_image = QRadioButton(f"Current image only ({current_image_name})")
        self.all_images.setChecked(True)
        self.current_image.setEnabled(bool(current_image_name))

        self.classes = QListWidget()
        self.classes.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for name in classes:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.classes.addItem(item)
        self.classes.setMaximumHeight(140)

        self.unit = QComboBox()
        self.unit.addItems(list(UNITS))
        self.unit.setCurrentText(scale.unit)
        self.per_class = QCheckBox("One file per class")

        default = Path(main_window.state_manager.project_root) / \
            f"measurements_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.path = QLineEdit(str(default))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path, 1)
        path_row.addWidget(browse)

        scale_note = QLabel(
            "Sizes are given in real units where the image has a scale (its own DPI, or "
            "Actions > Set Scale…) and always in pixels too.")
        scale_note.setWordWrap(True)
        scale_note.setStyleSheet("color: gray;")

        form = QFormLayout()
        scope = QVBoxLayout()
        scope.addWidget(self.all_images)
        scope.addWidget(self.current_image)
        form.addRow("Images", scope)
        form.addRow("Classes", self.classes)
        form.addRow("Lengths in", self.unit)
        form.addRow("", self.per_class)
        form.addRow("Save to", path_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(scale_note)
        layout.addWidget(buttons)
        self.setMinimumWidth(560)

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save measurements", self.path.text(), "CSV (*.csv)")
        if path:
            self.path.setText(path if path.lower().endswith(".csv") else path + ".csv")

    def _checked_classes(self):
        return [self.classes.item(i).text() for i in range(self.classes.count())
                if self.classes.item(i).checkState() == Qt.CheckState.Checked]

    def _ok(self):
        if not self._checked_classes():
            QMessageBox.information(self, "Save Results", "Tick at least one class.")
            return
        folder = Path(self.path.text()).expanduser().parent
        if not folder.is_dir():
            QMessageBox.warning(self, "Save Results", f"The folder {folder} does not exist.")
            return
        self.accept()

    def options(self) -> dict:
        classes = self._checked_classes()
        return {
            "out_path": str(Path(self.path.text()).expanduser()),
            "unit": self.unit.currentText(),
            "image_names": [self._current] if self.current_image.isChecked() else None,
            "classes": None if len(classes) == self.classes.count() else classes,
            "per_class": self.per_class.isChecked(),
        }


def run_results_export(main_window, options: dict):
    """Write the CSV(s) on a worker thread; progress window with Cancel."""
    state = main_window.state_manager
    image_paths = {os.path.splitext(os.path.basename(p))[0]: p for p in state.image_paths}

    def work(progress, cancelled):
        return measure_export.export_measurements(
            main_window.db_path,
            options["out_path"], image_paths, main_window.project_scale,
            unit=options["unit"], image_names=options["image_names"],
            classes=options["classes"], per_class=options["per_class"],
            progress=progress, cancelled=cancelled)

    def label(done, total, stage):
        return f"{stage} ({done:,} of {total:,} masks)" if total else stage

    run = run_in_background(main_window, "Save Results", work, label=label,
                            cancel_exception=measure_export.ExportCancelled,
                            start_text="Starting…")
    if run.cancelled:
        QMessageBox.information(main_window, "Save Results", "Cancelled; nothing was saved.")
        return None
    if run.error is not None:
        QMessageBox.critical(main_window, "Save Results", f"Could not save the results:\n{run.error}")
        return None
    result = run.result
    files = "\n".join(str(p) for p in result.files) or "(no masks matched)"
    text = f"Saved {result.rows:,} mask(s) to:\n{files}"
    if result.without_scale:
        text += (f"\n\n{result.without_scale} image(s) have no scale: their sizes are in pixels "
                 f"only (Actions > Set Scale…).")
    if result.skipped:
        text += f"\n\n{result.skipped} mask(s) without area were left out."
    logger.info("Saved measurements: %d rows to %s", result.rows, [str(p) for p in result.files])
    QMessageBox.information(main_window, "Save Results", text)
    return result
