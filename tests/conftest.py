"""
Shared test setup. Run from the repository root:

    python -m pip install pytest
    python -m pytest

Tests run without models, a GPU or network access. Qt runs headless
(offscreen). Tests that need an optional package (sam2, hireseg) skip
themselves when it isn't installed.
"""
import os
import sys
from pathlib import Path

# torch before anything loads PyQt6, as in main.py: PyQt6 carries an older
# copy of the MSVC++ runtime, and if Qt loads it first, torch's c10.dll
# fails to initialise on Windows ("WinError 1114").
try:
    import importlib
    importlib.import_module("torch")
except ImportError:
    pass

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Keep the app's log of test runs out of the user's real log folder.
import tempfile  # noqa: E402
os.environ.setdefault("SEGMENTME_LOG_DIR", os.path.join(tempfile.gettempdir(), "segmentme-test-logs"))

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Tests never read or change the user's real settings (display look, update
# checks, tour offered, SAM model...): QSettings("Org", "App") -- registry on
# Windows, plist on macOS, ~/.config on Linux -- becomes an .ini file in an
# empty folder of this run. Done before any app module imports QSettings.
import PyQt6.QtCore as _qtcore  # noqa: E402

_SETTINGS_DIR = tempfile.mkdtemp(prefix="segmentme-test-settings-")


class _TestSettings(_qtcore.QSettings):
    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], str) and (len(args) == 1 or isinstance(args[1], str)):
            name = "-".join(a for a in args[:2] if a) or "default"
            super().__init__(os.path.join(_SETTINGS_DIR, name + ".ini"),
                             _qtcore.QSettings.Format.IniFormat, *args[2:], **kwargs)
        else:
            super().__init__(*args, **kwargs)


_qtcore.QSettings = _TestSettings


_APP = None   # kept until the process ends (see qapp)
_EXIT_STATUS = 0


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole run, never freed by Python (see
    pytest_unconfigure)."""
    global _APP
    from PyQt6.QtWidgets import QApplication
    _APP = QApplication.instance() or QApplication([])
    yield _APP


def pytest_sessionfinish(session, exitstatus):
    global _EXIT_STATUS
    _EXIT_STATUS = int(exitstatus)


def pytest_unconfigure(config):
    """End the test process right after pytest has reported, skipping Python's
    final clean-up: there PyQt walks every Qt object the tests left behind
    (many tests only close their windows) and crashed now and then -- after
    all tests had passed. The app itself exits cleanly
    (test_app_quits_without_crashing_with_everything_open)."""
    if _APP is None:
        return
    import logging
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS)


@pytest.fixture
def no_popups(monkeypatch):
    """Record QMessageBox popups instead of blocking on them."""
    from PyQt6.QtWidgets import QMessageBox
    shown = []

    def fake(kind):
        def show(parent, title, text, *args, **kwargs):
            shown.append((kind, title, text))
            return QMessageBox.StandardButton.Ok
        return staticmethod(show)

    for kind in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, kind, fake(kind))
    return shown


def make_image(path, size=(64, 48), color=(10, 120, 200)):
    """Write a small RGB image (format from the extension); returns the path."""
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def can_symlink(tmp_path) -> bool:
    """Symbolic links need Developer Mode or admin rights on Windows."""
    target = tmp_path / "_probe_target"
    target.write_bytes(b"x")
    link = tmp_path / "_probe_link"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        return False
    finally:
        if os.path.lexists(link):
            os.unlink(link)
        target.unlink()
    return True


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def wait_for_image(window, timeout=10):
    """Opening a project / Next read the image on a worker thread: wait
    until it is on screen."""
    import time
    from PyQt6.QtWidgets import QApplication
    end = time.time() + timeout
    while time.time() < end and (window.image_display.loading() or window.image_display.busy()):
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()
