"""Settings -> Models: see which model checkpoints are installed, download
or import the missing ones, and choose which variant the SAM tool uses.

Also home to `ensure_model_available()`, the "this tool needs a model that
is not installed yet -- download it now?" prompt ToolManager shows the
first time SAM or DEXTR is enabled on a fresh install.
"""
from pathlib import Path

from PyQt6.QtCore import Qt, QEventLoop, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar,
    QProgressDialog, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from core.tools import sam_registry
from services import model_store
from services.logger import get_logger

logger = get_logger(__name__)

_FILE_FILTER = "PyTorch checkpoint (*.pt *.pth);;All files (*)"


def _type_label(spec) -> str:
    if spec.family == "sam2":
        return "SAM2.1" if spec.key.startswith("sam2.1") else "SAM2"
    return {"sam3": "SAM3", "dextr": "DEXTR"}.get(spec.family, spec.family)


class TransferThread(QThread):
    """Downloads a model, or copies in a user-supplied file, off the GUI thread."""

    # Byte counts can exceed a C int (SAM3 is 3.4 GB): pass them as objects.
    progress = pyqtSignal(object, object)   # bytes done, bytes total (0 = unknown)
    succeeded = pyqtSignal(str)             # installed path
    failed = pyqtSignal(str)                # error text; "" when cancelled

    def __init__(self, spec, source=None, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.source = source
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self.source:
                path = model_store.import_checkpoint(
                    self.source, self.spec,
                    progress=self.progress.emit, should_cancel=lambda: self._cancelled)
            else:
                path = model_store.download(
                    self.spec,
                    progress=self.progress.emit, should_cancel=lambda: self._cancelled)
        except model_store.DownloadCancelled:
            self.failed.emit("")
        except Exception as exc:
            logger.exception("Model transfer failed for %s", self.spec.key)
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(str(path))


def _progress_text(action, spec, done, total):
    total_text = model_store.format_size(total) if total else "?"
    return f"{action} {spec.label}...\n{model_store.format_size(done)} of {total_text}"


def run_transfer_modal(parent, spec, source=None):
    """Download `spec` (or import `source` as it) behind a modal progress
    dialog. Returns the installed path, or None if cancelled or failed
    (failures are shown to the user)."""
    action = "Importing" if source else "Downloading"
    dialog = QProgressDialog(f"{action} {spec.label} ({spec.size_label})...",
                             "Cancel", 0, 1000, parent)
    dialog.setWindowTitle("Models")
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setMinimumWidth(420)

    thread = TransferThread(spec, source, parent)
    outcome = {}

    def on_progress(done, total):
        if total:
            dialog.setRange(0, 1000)
            dialog.setValue(int(done * 1000 / total))
        else:
            dialog.setRange(0, 0)
        dialog.setLabelText(_progress_text(action, spec, done, total))

    thread.progress.connect(on_progress)
    thread.succeeded.connect(lambda path: outcome.update(path=path))
    thread.failed.connect(lambda error: outcome.update(error=error))
    dialog.canceled.connect(thread.cancel)

    loop = QEventLoop()
    thread.finished.connect(loop.quit)
    thread.start()
    dialog.show()
    loop.exec()
    dialog.close()
    dialog.deleteLater()
    thread.deleteLater()

    if "path" in outcome:
        logger.info("Installed %s at %s", spec.label, outcome["path"])
        return Path(outcome["path"])
    error = outcome.get("error")
    if error:
        QMessageBox.critical(parent, "Models", f"Could not install {spec.label}:\n\n{error}")
    else:
        logger.info("%s of %s cancelled", action, spec.label)
    return None


def ensure_model_available(parent, spec) -> bool:
    """True once `spec`'s checkpoint is installed, offering to download it
    (or, for gated models, to import a hand-downloaded file) if it is not."""
    if model_store.is_available(spec):
        return True

    if spec.gated:
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Model not installed")
        box.setText(f"{spec.label} is not installed.")
        box.setInformativeText(
            f"{spec.label} ({spec.size_label}) has to be downloaded by hand:\n\n"
            f"1. Request access at {spec.download_url}\n"
            f"2. Download {spec.checkpoint}\n"
            "3. Click Import and select the downloaded file."
        )
        import_button = box.addButton("Import file...", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is not import_button:
            return False
        source, _ = QFileDialog.getOpenFileName(parent, f"Select {spec.checkpoint}", "", _FILE_FILTER)
        if not source:
            return False
        run_transfer_modal(parent, spec, source)
    else:
        try:
            target_dir = model_store.writable_models_dir()
        except OSError as exc:
            QMessageBox.critical(parent, "Models", str(exc))
            return False
        reply = QMessageBox.question(
            parent, "Model not installed",
            f"{spec.label} is not installed yet.\n\n"
            f"Download it now ({spec.size_label})?\nIt will be saved to:\n{target_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False
        run_transfer_modal(parent, spec)

    return model_store.is_available(spec)


class ModelManagerDialog(QDialog):
    """Table of every known model with Download / Import buttons, plus the
    SAM-tool variant picker. `selected_key()` tells the caller which SAM
    variant was chosen when the dialog is accepted."""

    _COL_MODEL, _COL_TYPE, _COL_SIZE, _COL_STATUS, _COL_ACTION = range(5)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Models")
        self.setMinimumSize(760, 560)
        self._thread = None
        self._active_spec = None
        self._action = ""

        layout = QVBoxLayout(self)

        # Where models live
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Models folder:"))
        self.folder_label = QLabel()
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        folder_row.addWidget(self.folder_label, 1)
        open_button = QPushButton("Open folder")
        open_button.clicked.connect(self._open_folder)
        folder_row.addWidget(open_button)
        layout.addLayout(folder_row)

        hint = QLabel(
            "Downloaded and imported models are stored in this folder; you can also copy "
            "checkpoint files into it by hand. YOLO weights for Run Inference go in a "
            f"\"yolo\" subfolder. Set the {model_store.ENV_MODELS_DIR} environment "
            "variable to use a different folder."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(Mid); font-size: 11px;")
        layout.addWidget(hint)

        # Which SAM variant the sidebar tool runs
        sam_row = QHBoxLayout()
        sam_row.addWidget(QLabel("SAM tool uses:"))
        self.sam_combo = QComboBox()
        self.sam_combo.setToolTip("The model behind the sidebar's SAM button. "
                                  "A model that is not installed is downloaded when the tool is enabled.")
        sam_row.addWidget(self.sam_combo, 1)
        layout.addLayout(sam_row)

        # Model table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Model", "Type", "Size", "Status", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self._COL_MODEL, QHeaderView.ResizeMode.Stretch)
        for col in (self._COL_TYPE, self._COL_SIZE, self._COL_STATUS, self._COL_ACTION):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)

        # Transfer progress
        self.progress_label = QLabel()
        self.progress_bar = QProgressBar()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_transfer)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress_label, 1)
        progress_row.addWidget(self.progress_bar, 2)
        progress_row.addWidget(self.cancel_button)
        layout.addLayout(progress_row)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._set_busy(False)
        self.refresh()

    # ------------------------------------------------------------------ state
    def refresh(self):
        """Re-read the disk: fills the folder label, SAM combo and table."""
        primary = model_store.models_dir()
        try:
            writable = model_store.writable_models_dir()
        except OSError:
            writable = None
        text = str(primary)
        if writable is not None and writable != primary:
            text += f"   (read-only; downloads go to {writable})"
        self.folder_label.setText(text)

        current_key = self.sam_combo.currentData() or sam_registry.get_selected_key()
        self.sam_combo.blockSignals(True)
        self.sam_combo.clear()
        for spec in sam_registry.SAM_VARIANTS.values():
            suffix = "" if model_store.is_available(spec) else "  (not installed)"
            self.sam_combo.addItem(spec.label + suffix, spec.key)
        index = self.sam_combo.findData(current_key)
        self.sam_combo.setCurrentIndex(max(index, 0))
        self.sam_combo.blockSignals(False)

        self.table.setRowCount(0)
        for spec in model_store.MODELS.values():
            self._add_row(spec)
        self.table.resizeRowsToContents()

    def _add_row(self, spec):
        row = self.table.rowCount()
        self.table.insertRow(row)
        path = model_store.find_checkpoint(spec.checkpoint)

        model_item = QTableWidgetItem(spec.label)
        model_item.setToolTip(f"{spec.description}\nFile: {spec.checkpoint}")
        self.table.setItem(row, self._COL_MODEL, model_item)
        self.table.setItem(row, self._COL_TYPE, QTableWidgetItem(_type_label(spec)))
        self.table.setItem(row, self._COL_SIZE, QTableWidgetItem(spec.size_label))

        if path is not None:
            status = QTableWidgetItem("Installed")
            status.setToolTip(str(path))
            button = None
        elif spec.gated:
            status = QTableWidgetItem("Not installed (manual download)")
            status.setToolTip(spec.description)
            button = QPushButton("Import file...")
        else:
            status = QTableWidgetItem("Not installed")
            button = QPushButton("Download")
        self.table.setItem(row, self._COL_STATUS, status)
        if button is not None:
            button.clicked.connect(lambda _checked=False, s=spec: self._start_transfer(s))
            self.table.setCellWidget(row, self._COL_ACTION, button)

    def selected_key(self):
        """Key of the SAM variant chosen in the combo, or None."""
        return self.sam_combo.currentData()

    # -------------------------------------------------------------- transfers
    def _start_transfer(self, spec):
        if self._thread is not None:
            return
        source = None
        if spec.gated:
            source, _ = QFileDialog.getOpenFileName(self, f"Select {spec.checkpoint}", "", _FILE_FILTER)
            if not source:
                return
        self._active_spec = spec
        self._action = "Importing" if source else "Downloading"
        self._thread = TransferThread(spec, source, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.succeeded.connect(self._on_succeeded)
        self._thread.failed.connect(self._on_failed)
        self._set_busy(True, f"{self._action} {spec.label}...")
        self._thread.start()

    def _on_progress(self, done, total):
        if total:
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(int(done * 1000 / total))
        else:
            self.progress_bar.setRange(0, 0)
        self.progress_label.setText(
            _progress_text(self._action, self._active_spec, done, total).replace("\n", "  "))

    def _on_succeeded(self, path):
        logger.info("Installed %s at %s", self._active_spec.label, path)
        self._finish_transfer()

    def _on_failed(self, error):
        spec = self._active_spec
        self._finish_transfer()
        if error:
            QMessageBox.critical(self, "Models", f"Could not install {spec.label}:\n\n{error}")
        else:
            logger.info("Transfer of %s cancelled", spec.label)

    def _finish_transfer(self):
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
        self._active_spec = None
        self._set_busy(False)
        self.refresh()

    def _cancel_transfer(self):
        if self._thread is not None:
            self._thread.cancel()
            self.progress_label.setText("Cancelling...")

    def _set_busy(self, busy, text=""):
        self.progress_label.setText(text)
        self.progress_label.setVisible(busy)
        self.progress_bar.setVisible(busy)
        self.progress_bar.setRange(0, 0 if busy else 1)
        self.progress_bar.setValue(0)
        self.cancel_button.setVisible(busy)
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, self._COL_ACTION)
            if widget is not None:
                widget.setEnabled(not busy)

    def _open_folder(self):
        try:
            folder = model_store.writable_models_dir()
        except OSError:
            folder = model_store.models_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def done(self, result):
        # Closing mid-transfer: stop the thread first (a partial download is
        # kept on disk and resumes next time).
        if self._thread is not None:
            self._thread.cancel()
            self._thread.wait(10_000)
        super().done(result)
