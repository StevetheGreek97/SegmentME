"""
Review window for "Import Annotations", and the final report.

The window reads every .txt in the chosen folder and its subfolders
(core/importers/yolo_importer.py) and matches each to the project image of the
same name. It shows what will happen -- counts, problems, and a class table
where each class of data.yaml is merged into a same-named project class or
added as a new one (it can be renamed or merged into another existing class)
-- and returns the scan and the chosen class names. The main window then runs
the import. Images are never copied or changed here.
"""
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFontDatabase, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QFileDialog, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressDialog,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from core.importers.yolo_importer import (
    AMBIGUOUS, CLASS_NAME_RE, DUPLICATE, NO_IMAGE, UNREADABLE, ImportCancelled, ImportFailed,
    clean_class_name, describe, find_yaml, pick_new_colors, read_class_names, scan_annotations,
)
from services.logger import get_logger

logger = get_logger(__name__)

_OK, _WARN, _BAD = "#2e7d32", "#e65100", "#c62828"
_CHOOSE_FOLDER = ("Choose the folder with the .txt annotations: <b>Browse…</b> next to "
                  "“Annotations folder”. Then choose the data.yaml if it isn't found there.")


def _swatch(color_hex: str) -> QIcon:
    pm = QPixmap(12, 12)
    pm.fill(QColor(color_hex))
    return QIcon(pm)


class ImportAnnotationsDialog(QDialog):
    """Pick a folder of .txt annotations, review, then Import or Cancel."""

    COLUMNS = ("#", "Name in data.yaml", "Masks", "Project class", "Result")

    def __init__(self, parent, folder, project_images, existing_classes, existing_masks=None):
        """
        folder: the folder with the .txt files, or None to let the user
            choose it with Browse… (the dialog then opens empty).
        project_images: paths of the images in the project.
        existing_classes: project class name -> colour (hex).
        existing_masks: image name (stem) -> number of masks it already has.
        """
        super().__init__(parent)
        self.setWindowTitle("Import Annotations")
        self._project_images = list(project_images)
        self._existing_masks = dict(existing_masks or {})
        self._existing = dict(existing_classes)
        self._existing_lower = {name.lower(): name for name in self._existing}

        self.scan_result = None
        self._class_names = None
        self._folder = self._yaml_path = None

        # --- folder and data.yaml -------------------------------------------
        self.folder_edit = QLineEdit(readOnly=True)
        folder_btn = QPushButton("Browse…")
        folder_btn.clicked.connect(self._choose_folder)
        self.yaml_edit = QLineEdit(readOnly=True)
        self.yaml_edit.setPlaceholderText("Choose the dataset's data.yaml (class names)")
        yaml_btn = QPushButton("Browse…")
        yaml_btn.clicked.connect(self._choose_yaml)
        rescan_btn = QPushButton("Rescan")
        rescan_btn.clicked.connect(self._rescan)

        grid = QGridLayout()
        grid.addWidget(QLabel("Annotations folder"), 0, 0)
        grid.addWidget(self.folder_edit, 0, 1)
        grid.addWidget(folder_btn, 0, 2)
        grid.addWidget(QLabel("Class names"), 1, 0)
        grid.addWidget(self.yaml_edit, 1, 1)
        grid.addWidget(yaml_btn, 1, 2)
        note = QLabel("All .txt files in this folder and its subfolders are read; each belongs "
                      "to the project image with exactly the same name.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        grid.addWidget(note, 2, 1)
        grid.addWidget(rescan_btn, 2, 2, Qt.AlignmentFlag.AlignTop)
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
        self.resize(860, 600)

        self.folder_edit.setPlaceholderText("Choose the folder with the .txt annotations (Browse…)")
        if folder:
            self._set_folder(folder)
        else:
            self._show_hint(_CHOOSE_FOLDER)
            folder_btn.setDefault(True)
            folder_btn.setFocus()

    # ------------------------------------------------------------- inputs
    def _set_folder(self, folder):
        self._folder = Path(folder)
        yaml_path = find_yaml(self._folder)
        if yaml_path is not None or self._yaml_path is None:
            self._yaml_path = yaml_path
        self.folder_edit.setText(str(self._folder))
        QTimer.singleShot(0, self._rescan)  # once the window is up

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Folder with the .txt annotations",
                                                  str(self._folder or Path.home()))
        if folder:
            self._set_folder(folder)

    def _choose_yaml(self):
        start = str(self._yaml_path.parent if self._yaml_path else self._folder or Path.home())
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
        self.yaml_edit.setText(str(self._yaml_path) if self._yaml_path else "")

        if self._folder is None:
            self._show_hint(_CHOOSE_FOLDER)
            return
        if self._yaml_path is None:
            self._show_problem("Choose the dataset's data.yaml: it gives the name of each class number.")
            return
        try:
            self._class_names = read_class_names(self._yaml_path)
        except ImportFailed as exc:
            self._show_problem(str(exc))
            return

        self.summary.setText("Checking the .txt files and reading the size of each matching image…")
        progress = QProgressDialog("Checking annotations…", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(300)

        def report(done, total):
            progress.setMaximum(total)
            progress.setValue(done)
            QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            scan = scan_annotations(self._folder, self._class_names, self._project_images,
                                    self._existing_masks, report)
        except ImportCancelled:
            self._show_problem("Check cancelled. Press Rescan to check the folder again.")
            return
        except Exception as exc:
            logger.exception("Scanning annotations in %s failed", self._folder)
            self._show_problem(f"The folder could not be read: {exc}")
            return
        finally:
            progress.close()

        self.scan_result = scan
        self._show_summary(scan)
        self._fill_class_table(scan)
        self.details.setPlainText(describe(scan) or "Nothing to report.")

    def _show_hint(self, text):
        self.summary.setText(text)

    def _show_problem(self, text):
        self.summary.setText(f"<span style='color:{_BAD}'>{text}</span>")

    def _show_summary(self, scan):
        n_files = len(scan.entries)
        if n_files == 0:
            self._show_problem(f"No .txt files found in {scan.folder} or its subfolders.")
            return
        lines = [f"<b>{n_files}</b> .txt file(s) found."]

        def line(color, mark, text):
            lines.append(f"<span style='color:{color}'>{mark}</span> {text}")

        if scan.mask_count:
            line(_OK, "✔", f"<b>{scan.mask_count}</b> mask(s) will be added to "
                           f"<b>{scan.images_with_masks}</b> project image(s)")
        else:
            line(_BAD, "✖", "nothing to import: no .txt file with polygons matches a project image")
        existing = scan.with_existing_masks
        if existing:
            line(_WARN, "•", f"{len(existing)} of these images already have masks: the imported "
                             f"masks are added next to them")
        if scan.broken_labels:
            line(_WARN, "⚠", f"{len(scan.broken_labels)} .txt file(s) are broken: skipped (see details)")
        for status, text in ((NO_IMAGE, "have no project image with the same name: skipped"),
                             (DUPLICATE, "have the same name as another .txt in the folder: skipped"),
                             (AMBIGUOUS, "match more than one project image (e.g. a.png and a.jpg): skipped"),
                             (UNREADABLE, "belong to images that can't be read: skipped")):
            entries = scan.with_status(status)
            if entries:
                line(_BAD if status != NO_IMAGE else _WARN, "✖" if status != NO_IMAGE else "•",
                     f"{len(entries)} .txt file(s) {text}")
        conf = sum(1 for e in scan.to_import if e.label.confidence_ignored)
        if conf:
            line(_WARN, "•", f"confidence values ignored in {conf} .txt file(s)")
        dropped = sum(e.label.dropped_empty for e in scan.to_import)
        if dropped:
            line(_WARN, "•", f"{dropped} polygon(s) without area left out")
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
        self.import_btn.setEnabled(bool(scan and scan.mask_count and valid))

    # ------------------------------------------------------------- finish
    def _toggle_details(self, shown):
        self.details.setVisible(shown)
        self.details_btn.setText("Hide details" if shown else "Show details")

    def _confirm(self):
        if self.scan_result is not None and self.scan_result.mask_count:
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
