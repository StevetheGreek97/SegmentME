from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QFileDialog, QMessageBox, QLineEdit, QProgressBar,
    QAbstractItemView, QStyle, QApplication
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QPalette
from services import recent_projects
from services.recent_projects import load_recent_projects, remove_recent_project
from ui.recent_item import RecentItemWidget
from ui.dialogs.image_mode import ImageModeChoice
from ui.dialogs.place_progress import enough_space, run_placement
from services.logger import get_logger

logger = get_logger(__name__)
from services import image_files
from services.file_handlers import find_images, is_image_file
import shutil, os, re, pathlib, platform, json
from datetime import datetime

# ---------- Helpers: theme-aware stylesheet ----------

def _rgba(c: QColor, a: int = 255) -> str:
    return f"rgba({c.red()},{c.green()},{c.blue()},{a})"

def _is_dark(widget) -> bool:
    bg = widget.palette().color(QPalette.ColorRole.Window)
    return bg.lightness() < 128

def _build_stylesheet(widget) -> str:
    pal = widget.palette()
    dark = _is_dark(widget)

    bg    = pal.color(QPalette.ColorRole.Window)
    fg    = pal.color(QPalette.ColorRole.WindowText)
    base  = pal.color(QPalette.ColorRole.Base)

    # Carefully chosen neutrals for both modes
    card      = QColor(28, 28, 30) if dark else QColor(255, 255, 255)
    border    = QColor(60, 60, 65) if dark else QColor(225, 225, 230)
    hover     = QColor(40, 40, 44) if dark else QColor(246, 246, 248)
    muted     = QColor(170, 170, 175) if dark else QColor(105, 105, 110)
    accent    = QColor(99, 163, 255) if dark else QColor(33, 108, 214)
    accent_fg = QColor(0, 0, 0) if dark else QColor(255, 255, 255)

    return f"""
    /* Container */
    QDialog {{
        background: {bg.name()};
        color: {fg.name()};
    }}

    /* Titles */
    QLabel#title {{
        font-size: 22px; font-weight: 600; letter-spacing: 0.2px;
    }}
    QLabel#subtitle {{ color: {muted.name()}; margin-top: 2px; }}
    QLabel#section {{
        margin-top: 14px; margin-bottom: 6px;
        font-weight: 600; color: {muted.name()};
        letter-spacing: 0.3px; font-size: 11.5pt;
    }}

    /* General buttons */
    QPushButton {{
        background: {card.name()};
        border: 1px solid {border.name()};
        border-radius: 12px;
        padding: 10px 14px;
        min-width: 96px;
        color: {fg.name()};
    }}
    QPushButton:hover {{ border-color: {accent.name()}; background: {_rgba(hover, 255)}; }}
    QPushButton:pressed {{ transform: translateY(1px); }}

    /* Primary (highlighted) buttons */
    QPushButton#primary {{
        background: {accent.name()};
        color: {accent_fg.name()};
        border: 1px solid {accent.name()};
    }}
    QPushButton#primary:hover {{ background: {_rgba(accent, 230)}; }}

    /* Compact action buttons (Open/Delete in recents) */
    QPushButton#action {{
        padding: 4px 10px;
        min-width: 60px;
        border-radius: 8px;
        font-size: 10pt;
        text-align: center;
    }}


    /* Inputs */
    QLineEdit {{
        border: 1px solid {border.name()};
        border-radius: 10px;
        padding: 8px 10px;
        background: {base.name()};
    }}
    QProgressBar {{
        border: 1px solid {border.name()};
        border-radius: 10px;
        text-align: center;
        height: 12px;
    }}
    QProgressBar::chunk {{
        background-color: {accent.name()};
        border-radius: 10px;
    }}
    """

# ---------- Optional: drag-and-drop image list for the create dialog ----------

class DraggableImageList(QListWidget):
    """The images for a new project. Images or folders can be dropped on it;
    a folder adds every image in it and its subfolders."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.file_list = []
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.on_skipped = None   # called with the file names left out as duplicates

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        paths = [pathlib.Path(url.toLocalFile()) for url in event.mimeData().urls()]
        self.add_paths(paths)

    def add_paths(self, paths):
        """Add image files, and every image inside folders (with subfolders).
        Returns (number added, names skipped because an image with that name
        is already listed)."""
        files = []
        for path in paths:
            path = pathlib.Path(path)
            if path.is_dir():
                files.extend(find_images(path))
            elif path.is_file() and is_image_file(path):
                files.append(str(path))
        added, skipped = 0, []
        for f in files:
            if self.add_file(f):
                added += 1
            else:
                skipped.append(pathlib.Path(f).name)
        if skipped and self.on_skipped is not None:
            self.on_skipped(skipped)
        logger.debug("Added %d image(s) to the new project list; %d duplicate name(s) skipped",
                     added, len(skipped))
        return added, skipped

    def add_file(self, path) -> bool:
        """Add one image; False if an image with that file name is already
        listed (names must be unique inside the project's images folder)."""
        path = pathlib.Path(path)
        if any(pathlib.Path(f).name.lower() == path.name.lower() for f in self.file_list):
            return False
        self.file_list.append(str(path))
        self.addItem(path.name)
        return True

# ---------- Create Project sub-dialog ----------

class CreateProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.setMinimumWidth(520)
        self.setStyleSheet(_build_stylesheet(self))

        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(10)

        # Location
        v.addWidget(QLabel("Save Project To:"))
        row1 = QHBoxLayout()
        self.path_input = QLineEdit(self._default_desktop_path())
        browse_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon), " Browse")
        browse_btn.clicked.connect(self._select_folder)
        row1.addWidget(self.path_input)
        row1.addWidget(browse_btn)
        v.addLayout(row1)

        # Name
        v.addWidget(QLabel("Project Name (letters, numbers, _ or -):"))
        self.name_input = QLineEdit("MyCoolProject")
        v.addWidget(self.name_input)

        # Import (optional)
        v.addWidget(QLabel("Import Images (optional)"))
        row2 = QHBoxLayout()
        import_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView), " Add Images…")
        folder_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon), " Add Folder…")
        folder_btn.setToolTip("Add every image in a folder, including its subfolders")
        row2.addWidget(import_btn)
        row2.addWidget(folder_btn)
        row2.addStretch(1)
        v.addLayout(row2)

        self.img_list = DraggableImageList()
        self.img_list.setMinimumHeight(120)
        self.img_list.on_skipped = self._show_skipped
        v.addWidget(self.img_list)

        # Images left out because their name is already listed
        self.skipped_note = QLabel()
        self.skipped_note.setWordWrap(True)
        self.skipped_note.hide()
        v.addWidget(self.skipped_note)

        self.image_mode = ImageModeChoice(self)
        v.addWidget(self.image_mode)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        v.addWidget(self.progress)

        # Actions
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        create_btn = QPushButton("Create Project")
        create_btn.setObjectName("primary")
        action_row.addWidget(create_btn)
        v.addLayout(action_row)

        # Wire
        import_btn.clicked.connect(self._import_images)
        folder_btn.clicked.connect(self._import_folder)
        create_btn.clicked.connect(self._finalize)

        self.selected_project_db = None  # will be set on success

    def _default_desktop_path(self):
        system_platform = platform.system()
        if system_platform == "Windows":
            return os.path.join(os.environ.get("USERPROFILE", str(pathlib.Path.home())), "Desktop")
        return str(pathlib.Path.home() / "Desktop")

    def _select_folder(self):
        start_dir = self._default_desktop_path()
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", start_dir)
        if folder:
            self.path_input.setText(folder)

    def _import_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            self._default_desktop_path(),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        self.img_list.add_paths(files)

    def _import_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select a Folder of Images (subfolders included)", self._default_desktop_path())
        if not folder:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)   # a cloud folder lists slowly
        try:
            added, _skipped = self.img_list.add_paths([folder])
        finally:
            QApplication.restoreOverrideCursor()
        if not added and not _skipped:
            QMessageBox.information(self, "No Images", f"No images found in\n{folder}\nor its subfolders.")

    def _show_skipped(self, names):
        """Images have to have different names: they share one images folder
        and masks are stored by image name."""
        shown = ", ".join(names[:5]) + (f" and {len(names) - 5} more" if len(names) > 5 else "")
        self.skipped_note.setText(
            f"{len(names)} image(s) not added: an image with the same name is already in the "
            f"list ({shown}). Images in a project need different names.")
        self.skipped_note.show()

    def _finalize(self):
        base_dir = pathlib.Path(self.path_input.text()).expanduser().resolve()
        name = self.name_input.text().strip()
        if not self.img_list.file_list:
            QMessageBox.warning(self, "No Images", "You must import at least one image to continue.")
            return
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter a project name.")
            return

        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            QMessageBox.warning(self, "Invalid Name", "Use letters, numbers, underscores or hyphens.")
            return

        project_path = base_dir / name
        if project_path.exists():
            QMessageBox.warning(self, "Exists", "This project folder already exists.")
            return

        mode = self.image_mode.mode()
        if not enough_space(self, self.img_list.file_list, project_path, mode, "Create Project"):
            return
        try:
            (project_path / "images").mkdir(parents=True)
            (project_path / ".segmentme").mkdir(parents=True)

            # Copying runs on a worker thread with a progress window and Cancel.
            jobs = [(f, str(project_path / "images" / pathlib.Path(f).name))
                    for f in self.img_list.file_list]
            logger.info("Creating project %s with %d image(s), %s", project_path, len(jobs), mode)
            run = run_placement(self, jobs, mode, title="Create Project", stop_on_error=True)
            if run.cancelled:
                logger.info("Creating project %s cancelled; folder removed", project_path)
                shutil.rmtree(project_path, ignore_errors=True)
                return  # back to the dialog, nothing created
            if run.error is not None:
                raise run.error
            placements = run.result.placements
            self.progress.setValue(100)

            seproj_data = {
                "name": name,
                "created": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "images": sorted(pathlib.Path(f).name for f in self.img_list.file_list),
            }
            with open(project_path / f"{name}.SEproj", "w", encoding="utf-8") as fh:
                json.dump(seproj_data, fh, indent=2)

            self.selected_project_db = str(project_path / ".segmentme" / "masks.db")
            logger.info("Created project %s (%s)", project_path,
                        image_files.summarize(placements) or "no images")
            fallback = [how for how, reason in placements if how == image_files.COPIED and reason]
            if mode == image_files.LINK and fallback:
                QMessageBox.information(self, "Some images were copied",
                                        image_files.summarize(placements))
            self.accept()

        except Exception as e:
            # Don't leave a half-made project behind (it would block the name).
            # Removing it deletes links only, never the original images.
            logger.exception("Creating project %s failed; folder removed", project_path)
            shutil.rmtree(project_path, ignore_errors=True)
            QMessageBox.critical(self, "Error", str(e))

# ---------- Minimal, elegant Start Dialog ----------

class ProjectStartupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SegmentME")
        self.setMinimumSize(560, 420)
        self.setModal(True)

        self.selected_project_path = None
        self.is_new_project = None

        self.setStyleSheet(_build_stylesheet(self))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(14)

        # Header
        title = QLabel("SegmentME")
        title.setObjectName("title")
        subtitle = QLabel("Start a new project or open a recent one")
        subtitle.setObjectName("subtitle")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        # Primary actions
        actions = QHBoxLayout()
        actions.setSpacing(10)

        new_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder), "  New Project")
        new_btn.setObjectName("primary")
        new_btn.clicked.connect(self._create_project)



        actions.addWidget(new_btn, 1)
 
        outer.addLayout(actions)

        # Recents
        rec_label = QLabel("Recent")
        rec_label.setObjectName("section")
        outer.addWidget(rec_label)

        self.recent = QListWidget()
        self.recent.setUniformItemSizes(True)
        self.recent.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.recent.itemActivated.connect(self._open_recent_item)
        outer.addWidget(self.recent, 1)

        self._populate_recents()

    # ------- Actions -------
    def _populate_recents(self):
        self.recent.clear()

        db_paths = load_recent_projects()  # includes projects not found right now
        if not db_paths:
            empty = QListWidgetItem("No recent projects found.")
            empty.setFlags(empty.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.recent.addItem(empty)
            return

        for db_path in db_paths:
            # Derive a nice display name from the expected structure
            project_root = os.path.dirname(os.path.dirname(db_path))
            project_name = os.path.basename(project_root) or "Project"

            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, db_path)
            item.setSizeHint(QSize(item.sizeHint().width(), 48))

            widget = RecentItemWidget(
                project_name=project_name,
                db_path=db_path,
                on_open=self._open_recent_by_path,
                on_delete=self._delete_recent_by_path,
                parent=self.recent,
                missing=not os.path.exists(db_path),
            )

            self.recent.addItem(item)
            self.recent.setItemWidget(item, widget)
            item.setSizeHint(widget.sizeHint())

    def _open_recent_by_path(self, db_path: str):
        if not os.path.exists(db_path):
            # Often a USB or network drive that is not connected: keep it listed
            answer = QMessageBox.warning(
                self, "Project Not Found",
                f"The project was not found:\n{os.path.dirname(os.path.dirname(db_path))}\n\n"
                "If it is on a drive that is not connected, connect it and try again.\n\n"
                "Remove it from the list?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer == QMessageBox.StandardButton.Yes:
                remove_recent_project(db_path)
                self._populate_recents()
            return
        self.selected_project_path = db_path
        self.is_new_project = False
        self.accept()

    def _delete_recent_by_path(self, db_path: str):
        root = recent_projects.project_root_to_delete(db_path)
        if root is None:
            # Not a normal project folder (or already gone): only the list entry goes.
            recent_projects.delete_project(db_path)
            QMessageBox.information(
                self, "Removed from List",
                "The project was removed from the list. No files were deleted.")
            self._populate_recents()
            return
        bin_name = "Recycle Bin" if platform.system() == "Windows" else "Trash"
        box = QMessageBox(QMessageBox.Icon.Warning, "Delete Project?",
                          f"Delete the project '{root.name}'?", parent=self)
        box.setInformativeText(
            f"The whole project folder goes to the {bin_name}: its images, masks, trained "
            f"models and exports.\n\n{root}\n\nOriginal images that were only linked are "
            "not touched.")
        delete = box.addButton("Move to " + bin_name, QMessageBox.ButtonRole.DestructiveRole)
        keep = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(keep)
        box.setEscapeButton(keep)
        box.exec()
        if box.clickedButton() is not delete:
            return

        result = recent_projects.delete_project(db_path)
        if result == recent_projects.NO_TRASH:
            answer = QMessageBox.warning(
                self, "Delete Permanently?",
                f"This drive has no {bin_name}, so '{root.name}' cannot be moved there.\n\n"
                "Delete it permanently? This cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
            result = recent_projects.delete_project(db_path, permanently=True)
        if result == recent_projects.FAILED:
            QMessageBox.critical(self, "Delete Failed",
                                 "The project folder could not be deleted (is a file in it "
                                 "open in another program?).")
        self._populate_recents()

    def _open_recent_item(self, item: QListWidgetItem):
        # still works if user double-clicks the row
        db_path = item.data(Qt.ItemDataRole.UserRole)
        self._open_recent_by_path(db_path)

    def _create_project(self):
        dlg = CreateProjectDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.selected_project_path = dlg.selected_project_db
            self.is_new_project = True
            self.accept()
