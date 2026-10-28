from pathlib import Path
from core.config import ProjectConfigManager
from services.file_handlers import loader
from services.seproj import get_saved_images, update_images as seproj_update_images, update_classes as seproj_update_classes
from PyQt6.QtWidgets import QMessageBox, QFileDialog
from PyQt6.QtCore import QTimer


# services/recent_projects.py

import json, os, platform, shutil, datetime
from typing import List

from services.logger import get_logger
from services import session_state

logger = get_logger(__name__)

APP_DIR_NAME = "segmentme"
RECENT_FILE_NAME = "recent.json"
BACKUP_SUFFIX = ".bak"

def _app_support_dir() -> Path:
    if platform.system() == "Windows":
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / APP_DIR_NAME
    elif platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    else:
        return Path.home() / ".config" / APP_DIR_NAME

def get_recent_file() -> Path:
    base = _app_support_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / RECENT_FILE_NAME

def _read_json_safely(p: Path):
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # Ensure it’s a list of strings
        if isinstance(data, list):
            return [str(x) for x in data if isinstance(x, str)]
        return []
    except Exception:
        # backup corrupt file and reset
        logger.warning("Recent-projects file %s is corrupt; backing it up and starting fresh", p)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            p.rename(p.with_suffix(p.suffix + f".{ts}{BACKUP_SUFFIX}"))
        except Exception:
            logger.exception("Could not back up corrupt recent-projects file %s", p)
        return []

def _write_json_safely(p: Path, items: List[str]):
    p.write_text(json.dumps(items, indent=2), encoding="utf-8")

def _project_root_from_db(db_path: str) -> Path:
    # db_path -> .../<project>/.segmentme/masks.db
    return Path(db_path).resolve().parent.parent

def load_recent_projects() -> List[str]:
    """
    The recent projects' database paths, newest first -- including ones
    that are not found right now: a project on a USB or network drive that
    is not connected must not drop off the list for good. The list shows
    those as "not found"; the user removes them.
    """
    return _read_json_safely(get_recent_file())

def all_recent_projects_raw() -> List[str]:
    """
    Raw list (may include non-existing). For UI cleanup routines.
    """
    return _read_json_safely(get_recent_file())

def save_recent_project(db_path: str):
    db_path = str(Path(db_path).resolve())
    items = _read_json_safely(get_recent_file())
    if db_path in items:
        items.remove(db_path)
    items.insert(0, db_path)
    # de-duplicate & cap length if you want (e.g., 20)
    seen, dedup = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    _write_json_safely(get_recent_file(), dedup[:50])

def remove_recent_project(db_path: str):
    db_path = str(Path(db_path).resolve())
    items = _read_json_safely(get_recent_file())
    if db_path in items:
        items.remove(db_path)
        _write_json_safely(get_recent_file(), items)

def project_root_to_delete(db_path: str):
    """The project folder of `db_path`, only when it really is a SegmentME
    project: the database sits in <project>/.segmentme/ and the folder has
    a .SEproj file. Anything else (an old masks.db lying in some folder)
    gives None, so "Delete" can never take a parent folder with it."""
    db = Path(db_path).absolute()
    if db.parent.name != ".segmentme":
        return None
    root = db.parent.parent
    try:
        if root == root.parent or root.resolve() in (Path.home().resolve(), Path.home().resolve().parent):
            return None
        if not any(root.glob("*.SEproj")):
            return None
    except OSError:
        return None
    return root


# What delete_project() did
TRASHED, DELETED, NOT_A_PROJECT, NO_TRASH, FAILED, ALREADY_GONE = (
    "trashed", "deleted", "not a project", "no trash", "failed", "already gone")


def delete_project(db_path: str, permanently: bool = False) -> str:
    """Move a project's folder to the Trash / Recycle Bin (or, with
    permanently=True, delete it) and take it off the recent list.
    Linked images are links inside the folder: their originals are never
    touched. Returns one of the constants above; nothing is deleted for
    NOT_A_PROJECT and NO_TRASH (the Trash is not available there, e.g. some
    USB and network drives: ask before calling again with permanently=True)."""
    root = project_root_to_delete(db_path)
    if root is None:
        if not Path(db_path).exists():
            remove_recent_project(db_path)
            return ALREADY_GONE
        logger.warning("Refusing to delete %s: not inside a SegmentME project folder", db_path)
        remove_recent_project(db_path)
        return NOT_A_PROJECT
    try:
        if permanently:
            shutil.rmtree(root)
            logger.info("Deleted project folder %s", root)
            remove_recent_project(db_path)
            return DELETED
        from PyQt6.QtCore import QFile
        if QFile.moveToTrash(str(root)):
            logger.info("Moved project folder %s to the trash", root)
            remove_recent_project(db_path)
            return TRASHED
        logger.warning("Could not move %s to the trash", root)
        return NO_TRASH
    except Exception:
        logger.exception("Failed to delete project folder %s", root)
        return FAILED



_CHUNK = 900  # stay under SQLite's 999-variable limit

def _sync_seproj_images(main_window, project_root, image_paths) -> bool:
    """
    Compare the image list in the .SEproj file against what's on disk.
    If images are missing: show a popup, delete their masks from the DB
    via the app's own connection (no stale snapshot), emit masks_updated,
    and update the .SEproj. Returns False if the user cancels.
    A backup is made first, so File > Restore Backup… can undo it.
    """
    saved = get_saved_images(project_root)
    actual_names = [os.path.basename(p) for p in image_paths]

    if not saved:
        seproj_update_images(project_root, actual_names)
        return True

    actual_set = set(actual_names)
    missing = [name for name in saved if name not in actual_set]

    if not missing:
        seproj_update_images(project_root, actual_names)
        return True

    preview = "\n".join(f"  • {n}" for n in missing[:20])
    if len(missing) > 20:
        preview += f"\n  … and {len(missing) - 20} more"

    reply = QMessageBox.warning(
        main_window,
        "Missing Images Detected",
        f"{len(missing)} image(s) were removed from the project folder:\n\n"
        f"{preview}\n\n"
        f"Their annotations will be removed from the database.\n"
        f"(A backup is made first: File > Restore Backup… brings them back.)\n\n"
        f"Do you want to continue?",
        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
    )
    if reply == QMessageBox.StandardButton.Cancel:
        return False

    # The DB stores image_name WITHOUT extension (e.g. "fish1"), but .SEproj
    # stores filenames WITH extension (e.g. "fish1.jpg") — strip before deleting.
    missing_db = [os.path.splitext(n)[0] for n in missing]

    from services.backups import make_backup
    make_backup(main_window.state_manager.db.db_path, "before removing missing images")
    try:
        main_window.state_manager.mask_manager.delete_for_images(missing_db, chunk=_CHUNK)
    except Exception as exc:
        logger.exception("Failed to remove orphaned masks for %d missing image(s)", len(missing))
        QMessageBox.warning(
            main_window, "DB Cleanup Warning",
            f"Could not remove orphaned masks:\n{exc}"
        )

    seproj_update_images(project_root, actual_names)
    return True


def initialize_project(main_window, db_path):
    """
    Initializes project state and UI in the main window.
    """
    project_root = os.path.dirname(os.path.dirname(db_path))
    config = ProjectConfigManager(project_root)

    images_dir = config.get_images_dir()
    if not os.path.isdir(images_dir):
        reply = QMessageBox.question(
            main_window, "Images Folder Not Found",
            f"The project's images folder was not found:\n{images_dir}\n\n"
            "If it is on a drive that is not connected, connect it and open the project "
            "again. Do you want to choose the folder now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        folder = ""
        if reply == QMessageBox.StandardButton.Yes:
            folder = QFileDialog.getExistingDirectory(main_window, "Locate Images Folder")
        if folder:
            try:
                stored = os.path.relpath(folder, project_root)
            except ValueError:            # Windows: another drive (D:, a USB stick)
                stored = os.path.abspath(folder)
            config.set_images_dir(stored)
            images_dir = folder
        else:
            # Open without images; the project file and masks stay as they are.
            logger.warning("Images folder %s not found; project opened without images", images_dir)
            main_window.project_root = project_root
            main_window.config = config
            return

    try:
        image_paths = loader(images_dir) or []
    except OSError as exc:                # e.g. a network drive that dropped out
        logger.exception("Could not list %s", images_dir)
        QMessageBox.warning(main_window, "Images Folder",
                            f"The images folder could not be read:\n{images_dir}\n\n{exc}")
        main_window.project_root = project_root
        main_window.config = config
        return

    if not _sync_seproj_images(main_window, project_root, image_paths):
        return  # User canceled on missing-images dialog
    main_window.project_root = project_root
    main_window.config = config

    if image_paths:
        logger.info("Loaded %d image(s) from %s", len(image_paths), images_dir)
        main_window.project_root = project_root
        main_window.config = config
        main_window.state_manager.set_image_paths(image_paths)
        # Back to the image the user was on last time (services/session_state.py)
        place = session_state.load(project_root)
        names = [os.path.basename(p) for p in image_paths]
        if place.get("image") in names:
            main_window.state_manager.image_index = names.index(place["image"])
            logger.info("Reopening at image %s (%d of %d)", place["image"],
                        main_window.state_manager.image_index + 1, len(names))
        main_window.slider.set_image_count(len(image_paths))
        # read on a worker thread: a scan on a cloud drive can take seconds
        main_window.image_display.show_image(main_window.state_manager.current_image_path)
        QTimer.singleShot(0, main_window.image_display.fit_to_view)

        # Sync classes from DB into .SEproj on every open
        rows = main_window.state_manager.class_manager.list_classes()
        seproj_update_classes(project_root, [{"name": r[1], "color": r[2]} for r in rows])

        # ... and to the tool and class used last time
        if hasattr(main_window, "restore_place"):
            main_window.restore_place(place)
    elif hasattr(main_window, "restore_place"):
        main_window.restore_place({})     # no images yet: start remembering from here
