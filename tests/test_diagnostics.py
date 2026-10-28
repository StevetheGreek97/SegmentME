"""
The detailed log: system report, user actions, messages, Qt warnings,
freezes, database errors, native crashes, the diagnostic report.
"""
import logging
import os
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def text(self):
        return "\n".join(f"{r.levelname} {r.name}: {r.getMessage()}" for r in self.records)


@pytest.fixture
def captured():
    handler = Capture()
    app_logger = logging.getLogger("segmentme")
    app_logger.addHandler(handler)
    yield handler
    app_logger.removeHandler(handler)


def test_system_report_has_the_basics(qapp):
    from ui import diagnostics
    report = diagnostics.system_report()
    for key in ("SegmentME", "OS", "Python", "Qt / PyQt", "RAM", "CPU", "Log file", "Screen 0"):
        assert key in report, key


def test_clicks_menus_and_messages_are_recorded(qapp, captured):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QDialog, QMessageBox, QPushButton, QVBoxLayout
    from ui import diagnostics
    recorder = diagnostics.ActionRecorder()
    qapp.installEventFilter(recorder.filter)
    try:
        dialog = QDialog()
        dialog.setWindowTitle("Export Morphometrics")
        button = QPushButton("Export", dialog)
        QVBoxLayout(dialog).addWidget(button)
        dialog.show()
        QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
        dialog.accept()
        box = QMessageBox(QMessageBox.Icon.Warning, "Update", "The disk is full.")
        box.show()
        qapp.processEvents()
        box.close()
    finally:
        qapp.removeEventFilter(recorder.filter)
    text = captured.text()
    assert "Opened window 'Export Morphometrics'" in text
    assert "Clicked 'Export' in 'Export Morphometrics'" in text
    assert "Closed window 'Export Morphometrics' (OK)" in text
    assert "WARNING segmentme.actions: Message shown" in text and "The disk is full." in text
    if sys.platform != "darwin":       # macOS message boxes have no title
        assert "Message shown 'Update': The disk is full." in text
    assert "Clicked 'Export'" in diagnostics.recent_actions()


def test_crash_message_includes_what_happened_before(qapp, monkeypatch, captured):
    from ui import crash_handler, diagnostics
    monkeypatch.setattr(crash_handler, "show_error_message", lambda *a: None)
    diagnostics.action("Menu: Run Inference")
    try:
        raise RuntimeError("model file is broken")
    except RuntimeError:
        crash_handler.CrashHandler()(*sys.exc_info())
    critical = [r for r in captured.records if r.levelno == logging.CRITICAL][-1].getMessage()
    assert "model file is broken" in critical and "Menu: Run Inference" in critical


def test_database_errors_show_the_sql(tmp_path, captured):
    import sqlite3
    from core.database.connection import DatabaseConnection
    db = DatabaseConnection(str(tmp_path / "m.db"))
    with pytest.raises(sqlite3.Error):
        db.execute_query("INSERT INTO masks (image_name) VALUES (?)", ("a",))   # NOT NULL fails
    db.conn.close()
    text = captured.text()
    assert "Database error in m.db" in text and "INSERT INTO masks" in text and "('a')" in text


def test_freeze_is_logged_with_the_stack(qapp, captured):
    from ui import diagnostics
    dog = diagnostics.FreezeWatchdog(seconds=0.5).start()
    try:
        qapp.processEvents()
        def busy_doing_something_slow():
            time.sleep(1.2)            # the main thread does not handle events
        busy_doing_something_slow()
        end = time.time() + 2
        while time.time() < end and "responds again" not in captured.text():
            qapp.processEvents()
            time.sleep(0.05)
    finally:
        dog.stop()
    text = captured.text()
    assert "has not responded" in text and "busy_doing_something_slow" in text
    assert "responds again" in text


def test_qt_warnings_reach_the_log(captured):
    from PyQt6.QtCore import QtMsgType
    from ui import diagnostics

    class Context:
        file, line = "qwidget.cpp", 12
    diagnostics._qt_message_handler(QtMsgType.QtWarningMsg, Context(), "QPainter::begin: bad device")
    diagnostics._qt_message_handler(QtMsgType.QtWarningMsg, Context(), "does not support propagateSizeHints()")
    text = captured.text()
    assert "WARNING segmentme.qt: QPainter::begin: bad device (qwidget.cpp:12)" in text
    assert "propagateSizeHints" not in text


def test_damaged_project_file_is_kept_and_logged(tmp_path, captured):
    from services import seproj
    (tmp_path / "p.SEproj").write_text("{ this is not json")
    assert seproj.read_seproj(tmp_path) == {}
    assert list(tmp_path.glob("p.SEproj.damaged-*"))
    assert "could not be read" in captured.text()
    seproj.write_seproj(tmp_path, {"name": "p"})
    assert not list(tmp_path.glob("*.tmp"))


def test_diagnostic_report_zip(qapp, tmp_path, monkeypatch):
    from ui import diagnostics
    monkeypatch.setattr(diagnostics, "log_dir", lambda: tmp_path)
    (tmp_path / "segmentme.log").write_text("line 1\n")
    (tmp_path / "segmentme.log.1").write_text("older\n")
    diagnostics.action("Clicked 'Save'")
    path = diagnostics.save_report(tmp_path / "report.zip")
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        assert {"system.txt", "recent-actions.txt", "logs/segmentme.log", "logs/segmentme.log.1"} <= names
        assert "Clicked 'Save'" in z.read("recent-actions.txt").decode()


def test_help_menu_has_log_items(qapp, tmp_path, no_popups):
    import json
    from PyQt6.QtWidgets import QMenu
    from tests.conftest import make_image
    from ui.main_window import MainApp
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    make_image(root / "images" / "a.png")
    (root / "proj.SEproj").write_text(json.dumps({"name": "proj", "images": ["a.png"]}))
    win = MainApp(db_path=str(root / ".segmentme" / "masks.db"))
    try:
        texts = [a.text() for m in win.menu_bar.findChildren(QMenu) for a in m.actions()]
        assert "Open Log Folder" in texts and "Save Diagnostic Report…" in texts
    finally:
        win.state_manager.db.conn.close()


def _run_session(log_dir, body):
    """A separate Python process that starts a logging session."""
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(REPO)!r})
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt6.QtWidgets import QApplication
        app = QApplication([])
        from ui import diagnostics
        diagnostics.start_session(app)
    """) + textwrap.dedent(body)
    env = dict(os.environ, SEGMENTME_LOG_DIR=str(log_dir), QT_QPA_PLATFORM="offscreen")
    return subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                          text=True, timeout=120)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGSEGV simulation differs on Windows")
def test_native_crash_is_reported_on_the_next_start(tmp_path):
    """torch/OpenCV/Qt crashing in C++ kills the app without a Python error;
    the next start finds the dump and logs it."""
    crashed = _run_session(tmp_path, """
        import faulthandler
        def deep_inside_native_code():
            faulthandler._sigsegv()
        deep_inside_native_code()
    """)
    assert crashed.returncode != 0
    _run_session(tmp_path, "app.aboutToQuit.emit()\n")
    log = (tmp_path / "segmentme.log").read_text(encoding="utf-8")
    assert "did not end normally" in log
    assert "Native crash of the previous session" in log and "deep_inside_native_code" in log
    assert "System:" in log and "Session ended normally" in log
    assert not list(tmp_path.glob("crash-*.log")) and not list(tmp_path.glob("session-*.json"))


def test_app_quits_without_crashing_with_everything_open(tmp_path):
    """Logging's window watcher, the Display window, a tour and a tool all
    active at exit. The watcher used to touch dialogs while Qt destroyed
    them, and PyQt crashed while shutting down (most exits)."""
    import json
    from tests.conftest import make_image
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    make_image(root / "images" / "a.png", (300, 200))
    (root / "proj.SEproj").write_text(json.dumps({"name": "proj", "images": ["a.png"]}))
    body = f"""
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor
import numpy as np
from ui.main_window import MainApp
from services.recent_projects import initialize_project
from ui import tour
w = MainApp(db_path={str(root / ".segmentme" / "masks.db")!r})
w.state_manager.class_manager.add_class("Debris", QColor("#0000ff"))
initialize_project(w, w.db_path)
w.show()
w.display_settings.set_sharpen(30)
w.display_settings.set_class_visible("Debris", False)
w.show_display_settings()
w.sidebar.select_tool("manual_mask")
overlay = tour.start_tour(w, "table")
overlay.show_step(3)
QTimer.singleShot(300, lambda: overlay.close_tour(False))
QTimer.singleShot(600, w.close)
QTimer.singleShot(800, app.quit)
code = app.exec()
sys.exit(code)
"""
    for attempt in range(3):
        result = _run_session(tmp_path / f"logs{attempt}", body)
        assert result.returncode == 0, result.stderr[-2000:]
