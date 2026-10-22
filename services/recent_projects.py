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
    Returns a *clean* list of existing db paths.
    Also rewrites recent.json to drop missing entries.
    """
    f = get_recent_file()
    items = _read_json_safely(f)
    existing = []
    changed = False
    for p in items:
        if os.path.exists(p):
            existing.append(p)
        else:
            changed = True
    if changed:
        _write_json_safely(f, existing)
    return existing

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

def delete_project_and_remove_recent(db_path: str) -> bool:
    """
    Deletes the project folder (…/<project>) if it exists and removes the entry.
    Returns True if folder was deleted or didn’t exist; False on failure.
    """
    try:
        root = _project_root_from_db(db_path)
        remove_recent_project(db_path)
        if root.exists():
            shutil.rmtree(root)
        logger.info("Deleted project folder %s", root)
        return True
    except Exception:
        logger.exception("Failed to delete project folder for %s", db_path)
        return False

_CHUNK = 900  # stay under SQLite's 999-variable limit

def _sync_seproj_images(main_window, project_root, image_paths) -> bool:
    """
    Compare the image list in the .SEproj file against what's on disk.
    If images are missing: show a popup, delete their masks from the DB
    via the app's own connection (no stale snapshot), emit masks_updated,
    and update the .SEproj. Returns False if the user cancels.
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
        f"Do you want to continue?",
        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
    )
    if reply == QMessageBox.StandardButton.Cancel:
        return False

    # The DB stores image_name WITHOUT extension (e.g. "fish1"), but .SEproj
    # stores filenames WITH extension (e.g. "fish1.jpg") — strip before deleting.
    missing_db = [os.path.splitext(n)[0] for n in missing]

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
    if not os.path.exists(images_dir):
        reply = QMessageBox.question(
            main_window, "Images Folder Not Found",
            f"The folder '{images_dir}' does not exist.\nWould you like to locate it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            folder = QFileDialog.getExistingDirectory(main_window, "Locate Images Folder")
            if folder:
                relative = os.path.relpath(folder, project_root)
                config.set_images_dir(relative)
                images_dir = folder
            else:
                return  # User canceled

    image_paths = loader(images_dir)

    if not _sync_seproj_images(main_window, project_root, image_paths):
        return  # User canceled on missing-images dialog

    if image_paths:
        logger.info("Loaded %d image(s) from %s", len(image_paths), images_dir)
        main_window.project_root = project_root
        main_window.config = config
        main_window.state_manager.set_image_paths(image_paths)
        main_window.slider.set_image_count(len(image_paths))
        # read on a worker thread: a scan on a cloud drive can take seconds
        main_window.image_display.show_image(main_window.state_manager.current_image_path)
        QTimer.singleShot(0, main_window.image_display.fit_to_view)

        # Sync classes from DB into .SEproj on every open
        rows = main_window.state_manager.class_manager.list_classes()
        seproj_update_classes(project_root, [{"name": r[1], "color": r[2]} for r in rows])
