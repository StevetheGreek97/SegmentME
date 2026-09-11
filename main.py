import multiprocessing

# Must run before anything else is imported: in a frozen (PyInstaller)
# build, the helper processes that multiprocessing spawns -- e.g. the
# DataLoader workers ultralytics starts while training -- re-execute this
# very executable and must be handed off here instead of starting the GUI.
multiprocessing.freeze_support()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

# Re-invoked as a plain subprocess (see services.app_process)? Handle it
# before any PyQt6/GUI import so the worker stays lightweight and works
# identically whether this is `python main.py` or a frozen exe. The imports
# must stay literal: PyInstaller only bundles modules it can see imported.
_worker_flag = sys.argv[1] if len(sys.argv) > 1 else ""
if _worker_flag == "--inference-worker":
    from core.inference_worker import main as _worker_main
    sys.exit(_worker_main())
if _worker_flag == "--training-worker":
    from core.trainer.training_worker import main as _worker_main
    sys.exit(_worker_main())

import torch
from ui.main_window import MainApp
from ui.dialogs.project_dialog import ProjectStartupDialog
from services.recent_projects import save_recent_project, initialize_project
from services.logger import get_logger

from pathlib import Path
from PyQt6.QtWidgets import QApplication

logger = get_logger(__name__)

app = QApplication(sys.argv)

if len(sys.argv) > 1 and sys.argv[1].endswith(".SEproj"):
    seproj_file = Path(sys.argv[1]).resolve()
    project_path = seproj_file.parent
    db_path = project_path / ".segmentme" / "masks.db"

    if db_path.exists():
        save_recent_project(str(db_path))
        window = MainApp(db_path=str(db_path))
        initialize_project(window, str(db_path))
        window.show()
        sys.exit(app.exec())
    else:
        logger.error("Cannot open %s: expected project database at %s", seproj_file, db_path)
        sys.exit(1)
else:
    dialog = ProjectStartupDialog()
    if dialog.exec() == dialog.Accepted:
        save_recent_project(dialog.selected_project_path)
        window = MainApp(db_path=dialog.selected_project_path)
        initialize_project(window, dialog.selected_project_path)
        window.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)
