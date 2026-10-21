"""
Review window for "Import Images with Annotations", and the final report.

The window scans the chosen dataset (core/importers/yolo_importer.py), shows
what will happen -- counts, problems, and a class table where each class of the
dataset is merged into a same-named project class or added as a new one (it
can be renamed or merged into another existing class) -- and returns the scan
and the chosen class names. The main window then runs the import.
"""
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFontDatabase, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressDialog,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from core.importers.yolo_importer import (
    CLASS_NAME_RE, ImportCancelled, ImportFailed, clean_class_name, describe, detect_layout,
    pick_new_colors, read_class_names, scan_dataset,
)
from services.logger import get_logger

logger = get_logger(__name__)

_OK, _WARN, _BAD = "#2e7d32", "#e65100", "#c62828"


def _swatch(color_hex: str) -> QIcon:
    pm = QPixmap(12, 12)
    pm.fill(QColor(color_hex))
    return QIcon(pm)


def _short(path: Path, base: Path) -> str:
    try:
        rel = Path(path).relative_to(base)
        return f"{rel}/" if str(rel) != "." else f"{Path(path).name}/"
    except ValueError:
        return str(path)


class ImportAnnotationsDialog(QDialog):
    """Pick a dataset, review what will be imported, then Import or Cancel."""

    COLUMNS = ("#", "Name in data.yaml", "Masks", "Project class", "Result")

    def __init__(self, parent, folder, project_image_names, existing_classes):
        """
        folder: the dataset folder chosen by the user.
        project_image_names: file names of the images already in the project.
        existing_classes: project class name -> colour (hex).
        """
        super().__init__(parent)
        self.setWindowTitle("Import Images with Annotations")
        self._project_image_names = list(project_image_names)
        self._existing = dict(existing_classes)
        self._existing_lower = {name.lower(): name for name in self._existing}

        self.scan_result = None
        self._class_names = None
        self._folder = self._images_dir = self._labels_dir = self._yaml_path = None

        # --- dataset ---------------------------------------------------------
        self.folder_edit = QLineEdit(readOnly=True)
        folder_btn = QPushButton("Browse…")
        folder_btn.clicked.connect(self._choose_folder)
        self.layout_label = QLabel()
        self.layout_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        labels_btn = QPushButton("Labels folder…")
        labels_btn.setToolTip("Use .txt files from another folder")
        labels_btn.clicked.connect(self._choose_labels)
        self.yaml_edit = QLineEdit(readOnly=True)
        self.yaml_edit.setPlaceholderText("Choose the dataset's data.yaml (class names)")
        yaml_btn = QPushButton("Browse…")
        yaml_btn.clicked.connect(self._choose_yaml)
        self.subfolders = QCheckBox("Include subfolders (e.g. train / val / test)")
        self.subfolders.toggled.connect(lambda _: self._rescan())
        rescan_btn = QPushButton("Rescan")
        rescan_btn.clicked.connect(self._rescan)

        grid = QGridLayout()
        grid.addWidget(QLabel("Dataset folder"), 0, 0)
        grid.addWidget(self.folder_edit, 0, 1)
        grid.addWidget(folder_btn, 0, 2)
        grid.addWidget(QLabel("Files"), 1, 0)
        grid.addWidget(self.layout_label, 1, 1)
        grid.addWidget(labels_btn, 1, 2)
        grid.addWidget(QLabel("Class names"), 2, 0)
        grid.addWidget(self.yaml_edit, 2, 1)
        grid.addWidget(yaml_btn, 2, 2)
        options = QHBoxLayout()
        options.addWidget(self.subfolders)
        options.addStretch(1)
        options.addWidget(rescan_btn)
        grid.addLayout(options, 3, 1, 1, 2)
        grid.setColumnStretch(1, 1)

        # --- summary, classes, details --------------------------------------
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        for col, mode in enumerate((QHeaderView.ResizeMode.ResizeToContents,
                                    QHeaderView.ResizeMode.Stretch,
                                    QHeaderView.ResizeMode.ResizeToContents,
                                    QHeaderView.ResizeMode.Stretch,
                                    QHeaderView.ResizeMode.Stretch)):
            header.setSectionResizeMode(col, mode)
        self.classes_note = QLabel()
        self.classes_note.setWordWrap(True)

        self.details = QPlainTextEdit(readOnly=True)
        self.details.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.details.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.details.hide()
        self.details_btn = QPushButton("Show details")
        self.details_btn.setCheckable(True)
        self.details_btn.toggled.connect(self._toggle_details)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.import_btn = QPushButton("Import")
        self.import_btn.setDefault(True)
        self.import_btn.setEnabled(False)
        self.import_btn.clicked.connect(self._confirm)
        buttons = QHBoxLayout()
        buttons.addWidget(self.details_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.import_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(self.summary)
        layout.addWidget(QLabel("<b>Classes</b> — merged into the project class with the same name, "
                                "otherwise added. Type a new name, or pick an existing class to merge into."))
        layout.addWidget(self.table, 2)
        layout.addWidget(self.classes_note)
        layout.addWidget(self.details, 3)
        layout.addLayout(buttons)
        self.resize(860, 640)

        self._set_folder(folder)

    # ------------------------------------------------------------- inputs
    def _set_folder(self, folder):
        images_dir, labels_dir, yaml_path = detect_layout(folder)
        self._folder = Path(folder)
        self._images_dir, self._labels_dir = images_dir, labels_dir
        if yaml_path is not None or self._yaml_path is None:
            self._yaml_path = yaml_path
        self.folder_edit.setText(str(self._folder))
        self.subfolders.blockSignals(True)
        self.subfolders.setChecked(images_dir != self._folder)  # images/ layout: may hold splits
        self.subfolders.blockSignals(False)
        QTimer.singleShot(0, self._rescan)  # once the window is up

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Dataset folder", str(self._folder or ""))
        if folder:
            self._set_folder(folder)

    def _choose_labels(self):
        folder = QFileDialog.getExistingDirectory(self, "Folder with the .txt files",
                                                  str(self._labels_dir or self._folder or ""))
        if folder:
            self._labels_dir = Path(folder)
            self._rescan()

    def _choose_yaml(self):
        start = str(self._yaml_path.parent if self._yaml_path else self._folder or "")
        path, _ = QFileDialog.getOpenFileName(self, "Dataset data.yaml", start,
                                              "Dataset config (*.yaml *.yml)")
        if path:
            self._yaml_path = Path(path)
            self._rescan()

    # ------------------------------------------------------------- scanning
    def _rescan(self):
        self.scan_result = None
        self.import_btn.setEnabled(False)
        self.table.setRowCount(0)
        self.classes_note.clear()
        self.details.clear()
        self.layout_label.setText(f"images: {_short(self._images_dir, self._folder)}   "
                                  f".txt: {_short(self._labels_dir, self._folder)}")
        self.yaml_edit.setText(str(self._yaml_path) if self._yaml_path else "")

        if self._yaml_path is None:
            self._show_problem("Choose the dataset's data.yaml: it gives the name of each class number.")
            return
        try:
            self._class_names = read_class_names(self._yaml_path)
        except ImportFailed as exc:
            self._show_problem(str(exc))
            return

        progress = QProgressDialog("Checking images and annotations…", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(300)

        def report(done, total):
            progress.setMaximum(total)
            progress.setValue(done)
            QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            scan = scan_dataset(self._images_dir, self._labels_dir, self._class_names,
                                self._project_image_names, self.subfolders.isChecked(), report)
        except ImportCancelled:
            self._show_problem("Check cancelled. Press Rescan to check the dataset again.")
            return
        except Exception as exc:
            logger.exception("Scanning dataset %s failed", self._images_dir)
            self._show_problem(f"The dataset could not be read: {exc}")
            return
        finally:
            progress.close()

        self.scan_result = scan
        self._show_summary(scan)
        self._fill_class_table(scan)
        self.details.setPlainText(describe(scan) or "Nothing to report.")
        exists = scan.with_status("exists")
        if exists:
            names = "\n".join(f"  • {e.path.name}" for e in exists[:15])
            more = f"\n  … and {len(exists) - 15} more" if len(exists) > 15 else ""
            QMessageBox.information(
                self, "Already in the project",
                f"{len(exists)} image(s) already exist in this project and will be skipped "
                f"(their annotations are not imported):\n\n{names}{more}")

    def _show_problem(self, text):
        self.summary.setText(f"<span style='color:{_BAD}'>{text}</span>")

    def _show_summary(self, scan):
        n_images = len(scan.entries)
        if n_images == 0:
            self._show_problem(f"No images found in {scan.images_dir}"
                               + ("" if self.subfolders.isChecked() else
                                  " (subfolders are not included)") + ".")
            return
        readable = [e for e in scan.to_import if e.masks_ok]   # .txt read fine (may be empty)
        with_masks = [e for e in readable if e.label.polygons]
        lines = [f"<b>{n_images}</b> image(s) found."]

        def line(color, mark, text):
            lines.append(f"<span style='color:{color}'>{mark}</span> {text}")

        line(_OK, "✔", f"<b>{len(scan.to_import)}</b> will be imported, "
                       f"<b>{len(with_masks)}</b> of them with masks (<b>{scan.mask_count}</b> masks)")
        if scan.without_label:
            line(_WARN, "•", f"{len(scan.without_label)} have no .txt file: imported without masks")
        if scan.broken_labels:
            line(_WARN, "⚠", f"{len(scan.broken_labels)} .txt file(s) are broken: skipped, their "
                             f"images imported without masks (see details)")
        for status, text in (("exists", "already in the project: skipped"),
                             ("duplicate", "have the same name as another image in this dataset: skipped"),
                             ("unreadable", "can't be read as images: skipped")):
            entries = scan.with_status(status)
            if entries:
                line(_BAD, "✖", f"{len(entries)} {text}")
        conf = sum(1 for e in readable if e.label.confidence_ignored)
        if conf:
            line(_WARN, "•", f"confidence values ignored in {conf} .txt file(s)")
        dropped = sum(e.label.dropped_empty for e in readable)
        if dropped:
            line(_WARN, "•", f"{dropped} polygon(s) without area left out")
        if scan.orphan_labels:
            line(_WARN, "•", f"{len(scan.orphan_labels)} .txt file(s) have no image of the same name")
        self.summary.setText("<br>".join(lines))

    # ------------------------------------------------------------- classes
    def _default_target(self, name, number):
        for candidate in (str(name).strip(), clean_class_name(name, number)):
            if candidate.lower() in self._existing_lower:
                return self._existing_lower[candidate.lower()]
        return clean_class_name(name, number)

    def _fill_class_table(self, scan):
        counts = scan.class_counts
        used = sorted(counts)
        existing = sorted(self._existing, key=str.lower)
        self.table.setRowCount(len(used))
        for row, number in enumerate(used):
            name = scan.class_names[number]
            for col, value in ((0, str(number)), (1, name), (2, str(counts[number]))):
                item = QTableWidgetItem(value)
                if col != 1:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(existing)
            combo.setCurrentText(self._default_target(name, number))
            combo.currentTextChanged.connect(lambda _: self._update_results())
            self.table.setCellWidget(row, 3, combo)
            self.table.setItem(row, 4, QTableWidgetItem())
        unused = len(scan.class_names) - len(used)
        self.classes_note.setText(
            f"{unused} class(es) in data.yaml are not used by these files and are not added."
            if unused else "")
        self._update_results()

    def class_targets(self):
        """Class number -> project class name (existing spelling when merging)."""
        targets = {}
        for row in range(self.table.rowCount()):
            number = int(self.table.item(row, 0).text())
            text = self.table.cellWidget(row, 3).currentText().strip()
            targets[number] = self._existing_lower.get(text.lower(), text)
        return targets

    def _update_results(self):
        targets = self.class_targets()
        new = []
        for name in targets.values():
            if (CLASS_NAME_RE.fullmatch(name) and name.lower() not in self._existing_lower
                    and name.lower() not in {n.lower() for n in new}):
                new.append(name)
        colors = dict(zip((n.lower() for n in new), pick_new_colors(len(new), self._existing.values())))

        valid = True
        for row in range(self.table.rowCount()):
            name = targets[int(self.table.item(row, 0).text())]
            item = self.table.item(row, 4)
            item.setIcon(QIcon())
            if not CLASS_NAME_RE.fullmatch(name):
                item.setText("Invalid name: letters, numbers and _ only")
                item.setForeground(QColor(_BAD))
                valid = False
            elif name.lower() in self._existing_lower:
                item.setText(f"Merge into existing '{self._existing_lower[name.lower()]}'")
                item.setForeground(QColor(_OK))
                item.setIcon(_swatch(self._existing[self._existing_lower[name.lower()]]))
            else:
                item.setText("New class")
                item.setForeground(self.palette().text().color())
                item.setIcon(_swatch(colors[name.lower()]))
        scan = self.scan_result
        self.import_btn.setEnabled(bool(scan and scan.to_import and valid))

    # ------------------------------------------------------------- finish
    def _toggle_details(self, shown):
        self.details.setVisible(shown)
        self.details_btn.setText("Hide details" if shown else "Show details")

    def _confirm(self):
        if self.scan_result is not None and self.scan_result.to_import:
            self.accept()


class ImportReportDialog(QDialog):
    """The import's outcome as text, with Save report…"""

    def __init__(self, parent, text, save_dir=None):
        super().__init__(parent)
        self.setWindowTitle("Import finished")
        self._save_dir = save_dir
        self.text = QPlainTextEdit(readOnly=True)
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setPlainText(text)
        save_btn = QPushButton("Save report…")
        save_btn.clicked.connect(self._save)
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(save_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout = QVBoxLayout(self)
        layout.addWidget(self.text)
        layout.addLayout(buttons)
        self.resize(760, 480)

    def _save(self):
        start = str(Path(self._save_dir or Path.home()) / "import_report.txt")
        path, _ = QFileDialog.getSaveFileName(self, "Save import report", start, "Text (*.txt)")
        if path:
            try:
                Path(path).write_text(self.text.toPlainText(), encoding="utf-8")
            except OSError as exc:
                QMessageBox.warning(self, "Save report", f"Could not save the report:\n{exc}")
