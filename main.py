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
# torch must be imported before PyQt6 (below) is: PyQt6 ships its own,
# older copy of the MSVC++ runtime (MSVCP140.dll et al.) inside
# PyQt6/Qt6/bin, unmangled. If QApplication() loads that first, Windows'
# DLL search can hand torch's own loader that older, incompatible copy
# instead of the correct one, and torch fails with "DLL initialization
# routine failed" (WinError 1114) loading c10.dll -- reproduced by
# swapping this order while adding the splash screen below. The packaged
# build strips those stale DLLs at build time (see SegmentME.spec) so it
# isn't exposed there, but a source checkout's venv still has them.

from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMessageBox

app = QApplication(sys.argv)
# Lets the desktop match our windows to segmentme.desktop (registered by
# install-desktop.sh on Linux): the application name becomes the X11
# WM_CLASS, the desktop file name the Wayland app id.
app.setApplicationName("SegmentME")
QApplication.setDesktopFileName("segmentme")

# Show the splash screen before the remaining slow imports below
# (ultralytics, the rest of the Qt UI) run, and before the startup dialog
# or main window is built -- those take a couple of seconds in a frozen
# build, and without this the app looks hung after a double-click.
from ui.splash_screen import SplashScreen
splash = SplashScreen()
splash.show()
app.processEvents()

from ui.main_window import MainApp
app.processEvents()
from ui.dialogs.project_dialog import ProjectStartupDialog
from services.recent_projects import save_recent_project, initialize_project
from services.logger import get_logger
app.processEvents()

logger = get_logger(__name__)

# Started with a project file (double-click in a file manager, or given
# on the command line)? Open it directly instead of the startup dialog.
if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".seproj"):
    seproj_file = Path(sys.argv[1]).resolve()
    project_path = seproj_file.parent
    db_path = project_path / ".segmentme" / "masks.db"

    if db_path.exists():
        save_recent_project(str(db_path))
        window = MainApp(db_path=str(db_path))
        initialize_project(window, str(db_path))
        splash.finish(window)
        window.show()
        sys.exit(app.exec())
    else:
        splash.close()
        logger.error("Cannot open %s: expected project database at %s", seproj_file, db_path)
        QMessageBox.critical(
            None, "SegmentME",
            f"Cannot open {seproj_file.name}:\nno project database found at\n{db_path}",
        )
        sys.exit(1)
else:
    splash.close()
    dialog = ProjectStartupDialog()
    if dialog.exec() == dialog.Accepted:
        save_recent_project(dialog.selected_project_path)
        window = MainApp(db_path=dialog.selected_project_path)
        initialize_project(window, dialog.selected_project_path)
        window.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)
