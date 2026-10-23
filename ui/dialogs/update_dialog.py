"""
Updates: "SegmentME X is available" with Update / Later / Skip This Version.

* At start (packaged app only) SegmentME quietly asks GitHub, at most once
  a day, whether there is a newer release; Settings > Check for Updates
  Automatically turns that off. Help > Check for Updates… asks at any time.
* The automatic check shows a button in the status bar ("Update to X…")
  rather than opening a window in the middle of someone's work.
* Update downloads the file built for this computer (services/updates.py)
  in the background (progress and Cancel in the status bar), then:
    Windows  - starts the installer and closes SegmentME,
    macOS    - opens the .dmg (drag SegmentME into Applications),
    Linux    - shows the downloaded file and the command that installs it
               (a .deb needs sudo, which the app cannot ask for).
* Skip This Version: no more reminders until a newer version than that.

The question to GitHub runs on a worker thread, so a slow or missing
connection never holds up the window.
"""
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QSettings, QStandardPaths, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTextBrowser,
    QVBoxLayout,
)

from services import model_store, updates
from services.logger import get_logger
from version import __version__

logger = get_logger(__name__)

SETTINGS_ORG = "SegmentME"
KEY_AUTO = "updates/auto_check"
KEY_LAST = "updates/last_check"
KEY_SKIPPED = "updates/skipped_version"
CHECK_EVERY_SECONDS = 24 * 3600

UPDATE, LATER, SKIP = "update", "later", "skip"


class UpdateDialog(QDialog):
    """Shows what is new; `choice` is UPDATE, LATER or SKIP afterwards."""

    def __init__(self, parent, release, download):
        super().__init__(parent)
        self.choice = LATER
        self.setWindowTitle("Update Available")
        self.resize(560, 460)
        v = QVBoxLayout(self)

        title = QLabel(f"<h3>SegmentME {release.version} is available</h3>"
                       f"You have version {__version__}.")
        v.addWidget(title)

        notes = QTextBrowser()
        notes.setOpenExternalLinks(True)
        notes.setMarkdown(updates.notes_for_app(release.notes, release.page)
                          or "_No release notes._")
        v.addWidget(notes, 1)

        if download is not None:
            where = QLabel(f"Download: {download.name} ({model_store.format_size(download.size)})")
        else:
            where = QLabel("There is no download for this computer here; "
                           "Update opens the download page instead.")
        where.setWordWrap(True)
        where.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(where)

        row = QHBoxLayout()
        self.skip_button = QPushButton("Skip This Version")
        self.later_button = QPushButton("Later")
        self.update_button = QPushButton("Update" if download is not None else "Open Download Page")
        # No default button: a key pressed while the window pops up (Enter,
        # Space) must not start a 2 GB download. Esc = Later.
        for button in (self.skip_button, self.later_button, self.update_button):
            button.setAutoDefault(False)
            button.setDefault(False)
        self.later_button.setFocus()
        self.skip_button.clicked.connect(lambda: self._done(SKIP))
        self.later_button.clicked.connect(lambda: self._done(LATER))
        self.update_button.clicked.connect(lambda: self._done(UPDATE))
        row.addWidget(self.skip_button)
        row.addStretch(1)
        row.addWidget(self.later_button)
        row.addWidget(self.update_button)
        v.addLayout(row)

    def _done(self, choice):
        self.choice = choice
        self.accept() if choice == UPDATE else self.reject()


class _Answer(QObject):
    """Carries the worker thread's result to the window's thread."""
    arrived = pyqtSignal(object, object, bool)   # release, error, manual


class Updater(QObject):
    """Owns the update check for one window (window.updater)."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_ORG)
        self.installation = updates.current_installation()
        self.checking = False
        self.notice = None            # status-bar button after the automatic check
        self._noticed = None
        self.downloading = None       # DownloadJob while an update downloads
        self._answer = _Answer(self)
        self._answer.arrived.connect(self._on_answer)

    # ---------- settings ----------
    @property
    def auto_check(self) -> bool:
        return self.settings.value(KEY_AUTO, True, type=bool)

    @auto_check.setter
    def auto_check(self, on):
        self.settings.setValue(KEY_AUTO, bool(on))

    def skipped_version(self) -> str:
        return self.settings.value(KEY_SKIPPED, "", type=str)

    # ---------- checking ----------
    def check_on_start(self):
        """The quiet daily check. Only the packaged app does it: from source
        there is nothing to install, and tests must not go online."""
        if self.installation.kind == updates.SOURCE or not self.auto_check:
            return False
        last = self.settings.value(KEY_LAST, 0.0, type=float)
        since = time.time() - last
        if 0 <= since < CHECK_EVERY_SECONDS:      # negative: the clock was set back
            return False
        return self.check(manual=False)

    def check_now(self):
        """Help > Check for Updates…"""
        return self.check(manual=True)

    def check(self, manual):
        if self.downloading is not None:
            if manual:
                QMessageBox.information(self.window, "Check for Updates",
                                        "An update is already downloading (see the bottom "
                                        "of the window).")
            return False
        if self.checking:
            return False
        self.checking = True
        if manual:
            self._status("Checking for updates…")

        def work():
            release = error = None
            try:
                release = updates.fetch_latest()
            except updates.UpdateError as exc:
                error = exc
            except Exception as exc:              # never let a check crash the app
                logger.exception("Update check failed")
                error = exc
            try:
                self._answer.arrived.emit(release, error, manual)
            except RuntimeError:                  # the window closed meanwhile
                pass

        threading.Thread(target=work, name="update-check", daemon=True).start()
        return True

    def _on_answer(self, release, error, manual):
        self.checking = False
        if error is not None:
            logger.info("Update check: %s", error)
            if manual:
                self._status("")
                QMessageBox.warning(self.window, "Check for Updates",
                                    f"Could not check for updates.\n\n{error}")
            return
        self.settings.setValue(KEY_LAST, time.time())
        newer = updates.is_newer(release.version)
        logger.info("Update check: latest %s, running %s", release.version, __version__)
        if manual:
            self._status("")
        if not newer:
            if manual:
                QMessageBox.information(self.window, "Check for Updates",
                                        f"SegmentME {__version__} is the latest version.")
            return
        if not manual and release.version == self.skipped_version():
            return
        if manual:
            self.offer(release)
        else:
            self.show_notice(release)

    def show_notice(self, release):
        """A button in the status bar ("Update to X…"): the automatic check
        never opens a window while someone is drawing."""
        bar = self._bar()
        if bar is None:
            return self.offer(release)
        if self.notice is None:
            self.notice = QPushButton()
            self.notice.setFlat(True)
            self.notice.setCursor(Qt.CursorShape.PointingHandCursor)
            self.notice.setStyleSheet("QPushButton { color: palette(link); font-weight: bold; }")
            self.notice.clicked.connect(lambda: self.offer(self._noticed))
            bar.addPermanentWidget(self.notice)
        self._noticed = release
        self.notice.setText(f"⬆ Update to {release.version}…")
        self.notice.setToolTip(f"SegmentME {release.version} is available (you have {__version__})")
        self.notice.show()
        return None

    def _hide_notice(self):
        if self.notice is not None:
            self.notice.hide()

    # ---------- offering ----------
    def offer(self, release):
        download = updates.pick_download(release, self.installation)
        dialog = UpdateDialog(self.window, release, download)
        dialog.exec()
        if dialog.choice == SKIP:
            self.settings.setValue(KEY_SKIPPED, release.version)
            self._hide_notice()
        elif dialog.choice == UPDATE:
            self._hide_notice()
            if download is None:
                open_url(release.page)
            else:
                self.download_and_install(release, download)
        return dialog.choice

    def download_and_install(self, release, download):
        """Download in the background (the app stays usable; slow
        connections can take a long time), shown at the bottom of the
        window with a Cancel button; then install."""
        if self.downloading is not None:
            return None
        folder = download_folder()
        need = updates.space_needed(download, folder)
        free = updates.free_space(folder)
        if need and free < need + 200_000_000:
            QMessageBox.warning(
                self.window, "Update",
                f"There is not enough free disk space for the update.\n\n"
                f"Needed: {model_store.format_size(need)}\n"
                f"Free on that drive: {model_store.format_size(free)}\n"
                f"Folder: {folder}\n\nFree some space and try again.")
            return None
        self.downloading = DownloadJob(self, release, download, folder)
        self.downloading.start()
        return self.downloading

    def _download_finished(self, job):
        self.downloading = None
        if job.cancelled:
            self._status("Update download stopped; next time it continues where it left off.")
            return
        if job.error is not None:
            answer = QMessageBox.warning(
                self.window, "Update",
                f"The update could not be downloaded.\n\n{job.error}\n\n"
                "Open the download page instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if answer == QMessageBox.StandardButton.Yes:
                open_url(job.release.page)
            return
        self.install(Path(job.result), job.release)

    def stop(self):
        """The window is closing: stop a download (it resumes next time)."""
        if self.downloading is not None:
            self.downloading.cancel.set()

    def install(self, path, release):
        kind = self.installation.kind
        if kind == updates.WINDOWS_INSTALLER:
            answer = QMessageBox.question(
                self.window, "Install Update",
                f"SegmentME {release.version} is downloaded.\n\nSegmentME will close and the "
                "installer will start. Your masks are already saved.",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Ok:
                show_in_folder(path)
                return
            if not self._launch(path):
                return
            self.window.close()
            QApplication.quit()
        elif kind == updates.MACOS_DMG:
            if not self._launch(path):
                return
            QMessageBox.information(
                self.window, "Install Update",
                f"SegmentME {release.version} is downloaded and opened.\n\nQuit SegmentME, then "
                "drag SegmentME from the opened window into Applications, replacing the old one.")
        else:
            self._linux_instructions(path, release)

    def _launch(self, path):
        try:
            launch(path)
            return True
        except Exception as exc:          # antivirus, file removed, ...
            logger.warning("Could not open %s: %r", path, exc)
            QMessageBox.warning(
                self.window, "Install Update",
                f"The update could not be opened ({exc}).\n\nIt is saved here; open it "
                f"by hand:\n{path}")
            show_in_folder(path)
            return False

    def _linux_instructions(self, path, release):
        if self.installation.kind == updates.LINUX_DEB:
            command = f'sudo apt install "{path}"'
            text = (f"SegmentME {release.version} is downloaded to:\n{path}\n\n"
                    f"Close SegmentME and install it in a terminal with:\n\n{command}")
        else:
            command = f'tar -xzf "{path}" -C "{path.parent}"'
            text = (f"SegmentME {release.version} is downloaded to:\n{path}\n\n"
                    f"Close SegmentME, extract it with:\n\n{command}\n\nand use the new "
                    f"folder in place of the old one:\n{model_store.app_dir()}\n"
                    "(keep its models folder).")
        box = QMessageBox(QMessageBox.Icon.Information, "Install Update", text, parent=self.window)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        copy = box.addButton("Copy Command", QMessageBox.ButtonRole.ActionRole)
        folder = box.addButton("Show in Folder", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is copy:
            QGuiApplication.clipboard().setText(command)
            self._status("Install command copied.")
        elif box.clickedButton() is folder:
            show_in_folder(path)

    def _bar(self):
        return self.window.statusBar() if hasattr(self.window, "statusBar") else None

    def _status(self, text):
        bar = self._bar()
        if bar is not None:
            bar.showMessage(text, 8000) if text else bar.clearMessage()


class DownloadJob:
    """One update download on a worker thread, shown in the status bar."""

    def __init__(self, updater, release, download, folder):
        self.updater = updater
        self.release = release
        self.download = download
        self.folder = folder
        self.cancel = threading.Event()
        self.shared = {"done": 0, "total": download.size, "text": "", "finished": False}
        self.result = self.error = None
        self.cancelled = False
        self.widget = None
        self.timer = None

    def start(self):
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QProgressBar, QWidget
        bar = self.updater._bar()
        if bar is not None:
            self.widget = QWidget()
            row = QHBoxLayout(self.widget)
            row.setContentsMargins(0, 0, 0, 0)
            self.label = QLabel("Downloading update…")
            self.progress = QProgressBar()
            self.progress.setRange(0, 1000)
            self.progress.setMaximumWidth(160)
            self.progress.setTextVisible(False)
            stop = QPushButton("Cancel")
            stop.setFlat(True)
            stop.clicked.connect(self.cancel.set)
            row.addWidget(self.label)
            row.addWidget(self.progress)
            row.addWidget(stop)
            bar.addPermanentWidget(self.widget)
        self.timer = QTimer(self.updater)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._poll)
        threading.Thread(target=self._work, name="update-download", daemon=True).start()
        self.timer.start()

    def _work(self):
        def progress(done, total, text=None):
            self.shared.update(done=done, total=total, text=text or self.shared["text"])
        try:
            self.result = updates.download(self.download, self.folder, progress, self.cancel.is_set)
        except model_store.DownloadCancelled:
            self.cancelled = True
        except Exception as exc:
            if not isinstance(exc, updates.UpdateError):
                logger.exception("Update download failed")
            self.error = exc
        finally:
            self.shared["finished"] = True

    def _poll(self):
        done, total = self.shared["done"], self.shared["total"]
        if self.widget is not None:
            if self.cancel.is_set():
                self.label.setText("Stopping…")
            else:
                size = (f"{model_store.format_size(done)} of {model_store.format_size(total)}"
                        if total else model_store.format_size(done))
                self.label.setText(f"Downloading update {self.release.version}: {size}")
            self.progress.setValue(min(1000, int(1000 * done / total)) if total else 0)
        if self.shared["finished"]:
            self.timer.stop()
            if self.widget is not None:
                self.widget.deleteLater()
                self.widget = None
            self.updater._download_finished(self)

    def wait(self, timeout=30):
        """Tests: run until the download is over."""
        end = time.time() + timeout
        while self.timer.isActive() and time.time() < end:
            QApplication.processEvents()
            time.sleep(0.01)


# ---------- small helpers (replaced in tests) ----------
def download_folder() -> Path:
    """The Downloads folder, or another one SegmentME can really write to."""
    import tempfile
    found = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
    for folder in ([Path(found)] if found else []) + [
            model_store.user_data_dir() / "updates", Path(tempfile.gettempdir()) / "segmentme-updates"]:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".segmentme-write-test"
            probe.write_bytes(b"")
            probe.unlink()
            return folder
        except OSError:
            continue
    return Path(tempfile.gettempdir())


def open_url(url):
    QDesktopServices.openUrl(QUrl(url))


def show_in_folder(path):
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))


def launch(path):
    """Start an installer (Windows) or open a disk image (macOS), detached
    from SegmentME so it keeps running after the app closes."""
    path = str(path)
    if platform.system() == "Windows":
        os.startfile(path)        # through the shell: asks for admin rights if needed
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=dict(os.environ))
