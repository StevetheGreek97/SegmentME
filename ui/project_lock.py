"""
One project, one SegmentME window.

Two windows on the same project each show their own view of the masks and
overwrite each other's project file, which looks like lost work. A lock
file in <project>/.segmentme/ notices a second window and asks first.
A lock left behind by a crash is cleared automatically (its process is
gone), and a project on a read-only drive simply opens.
"""
from pathlib import Path

from PyQt6.QtCore import QLockFile
from PyQt6.QtWidgets import QMessageBox

from services.logger import get_logger

logger = get_logger(__name__)

LOCK_NAME = "segmentme.lock"


def lock_project(db_path, parent=None, ask=True):
    """Returns (go_on, lock). Keep `lock` alive while the project is open.
    go_on is False when the user chose not to open it a second time."""
    path = Path(db_path).parent / LOCK_NAME
    lock = QLockFile(str(path))
    lock.setStaleLockTime(0)              # stale only when its process is gone
    if lock.tryLock(100):
        return True, lock
    if lock.error() != QLockFile.LockError.LockFailedError:
        logger.info("Project lock not available (%s); opening anyway", lock.error())
        return True, None                 # read-only drive etc.
    ok, pid, host, _app = lock.getLockInfo()
    logger.warning("Project %s is already open (pid %s on %s)", db_path, pid, host)
    if not ask:
        return False, None
    box = QMessageBox(QMessageBox.Icon.Warning, "Project Already Open",
                      "This project is already open in another SegmentME window"
                      + (f" (on {host})" if ok and host else "") + ".", parent=parent)
    box.setInformativeText("Working on it in two windows at once mixes up the changes. "
                           "Switch to the other window instead.")
    cancel = box.addButton(QMessageBox.StandardButton.Cancel)
    anyway = box.addButton("Open Anyway", QMessageBox.ButtonRole.AcceptRole)
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    box.exec()
    return box.clickedButton() is anyway, None
