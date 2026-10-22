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

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Keep the app's log of test runs out of the user's real log folder.
import tempfile  # noqa: E402
os.environ.setdefault("SEGMENTME_LOG_DIR", os.path.join(tempfile.gettempdir(), "segmentme-test-logs"))

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


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
