"""
Everything the log needs to tell what went wrong, beyond the modules'
own messages:

* A system report when a session starts (version, build, OS, Python, Qt,
  torch/CUDA/GPU, CPU, RAM, disks, screens, settings) and a project report
  when a project opens.
* Hard crashes: Python's faulthandler writes the stack of every thread to
  crash-<pid>.log when native code (torch, OpenCV, Qt) brings the process
  down; the next start copies it into the log. A session that ended without
  saying goodbye (killed, power cut) is reported as such.
* Breadcrumbs: every button and menu item clicked, shortcut used, window
  opened and closed, and every message the user was shown -- so an error
  comes with what led to it (also kept in memory for the crash message).
* Qt's own warnings, which otherwise go nowhere in the packaged app.
* Freezes: when the window stops responding for FREEZE_SECONDS, the stack
  of the main thread is logged (what it was busy with), and again how long
  it lasted.
* Help > Save Diagnostic Report…: the logs plus these reports in one zip
  the user can send.
"""
import collections
import faulthandler
import json
import os
import platform
import sys
import threading
import time
import traceback
import zipfile
from datetime import datetime
from importlib import metadata
from pathlib import Path

import psutil

from services.logger import get_logger, log_dir, log_file
from version import __version__

logger = get_logger("diagnostics")
actions_logger = get_logger("actions")

FREEZE_SECONDS = 3.0
BREADCRUMBS = collections.deque(maxlen=60)       # (time, text)
_state = {"crash_file": None, "marker": None, "watchdog": None, "filter": None}


# --------------------------------------------------------------------------
# Breadcrumbs
# --------------------------------------------------------------------------
def action(text, *args, level=20):
    """Record something the user did (also used by the window's shortcuts).
    level=10 (debug) for frequent things like clicks on the image: kept in
    the file, not on the console."""
    try:
        if args:
            text = text % args
    except Exception:
        text = f"{text} {args}"
    BREADCRUMBS.append((time.time(), text))
    actions_logger.log(level, "%s", text)


def recent_actions(limit=25):
    """The last user actions as text, oldest first."""
    rows = list(BREADCRUMBS)[-limit:]
    return "\n".join(f"{datetime.fromtimestamp(t):%H:%M:%S}  {text}" for t, text in rows)


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def _version_of(dist):
    try:
        return metadata.version(dist)
    except Exception:
        return "not installed"


def _gb(n):
    return f"{n / 1e9:.1f} GB"


def _disk(path):
    try:
        p = Path(path)
        while not p.exists() and p != p.parent:
            p = p.parent
        usage = psutil.disk_usage(str(p))
        return f"{_gb(usage.free)} free of {_gb(usage.total)}"
    except Exception as exc:
        return f"unknown ({exc})"


def _cpu_name():
    """platform.processor() is empty on Linux and terse on macOS."""
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            import subprocess
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=2).stdout.strip()
            if out:
                return out
    except Exception:
        pass
    return platform.processor() or platform.machine()


def system_report() -> str:
    """Plain-text facts about this computer and this copy of the app."""
    lines = []

    def add(key, value):
        lines.append(f"{key:<22} {value}")

    add("SegmentME", __version__)
    try:
        from services import updates
        inst = updates.current_installation()
        add("Build", f"{inst.kind}, {inst.flavor}, {inst.system} {inst.arch}")
    except Exception as exc:
        add("Build", f"unknown ({exc})")
    add("Executable", sys.executable)
    add("Arguments", " ".join(sys.argv))
    add("Working folder", os.getcwd())
    add("Started", datetime.now().isoformat(timespec="seconds"))
    add("OS", f"{platform.platform()} ({platform.machine()})")
    add("Python", sys.version.replace("\n", " "))
    try:
        from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
        add("Qt / PyQt", f"{QT_VERSION_STR} / {PYQT_VERSION_STR}")
    except Exception:
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        add("torch", torch.__version__)
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    add(f"GPU {i}", f"{props.name}, {_gb(props.total_memory)} "
                                    f"(CUDA {torch.version.cuda})")
            else:
                add("GPU", "none usable (CPU only)")
        except Exception as exc:
            add("GPU", f"check failed ({exc})")
    for dist in ("numpy", "opencv-python", "opencv-python-headless", "ultralytics", "shapely"):
        version = _version_of(dist)
        if version != "not installed" or dist in ("numpy", "ultralytics"):
            add(dist, version)
    add("CPU", f"{_cpu_name()}, "
               f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads")
    mem = psutil.virtual_memory()
    add("RAM", f"{_gb(mem.available)} free of {_gb(mem.total)}")
    add("Disk (home)", _disk(Path.home()))
    add("Log file", log_file())
    try:
        from PyQt6.QtGui import QGuiApplication
        for i, screen in enumerate(QGuiApplication.screens()):
            g = screen.geometry()
            add(f"Screen {i}", f"{g.width()}x{g.height()} at scale {screen.devicePixelRatio():g}")
        add("Qt platform", QGuiApplication.platformName())
    except Exception:
        pass
    for key in sorted(os.environ):
        if key.startswith(("SEGMENTME_", "QT_", "CUDA_VISIBLE")):
            add(f"env {key}", os.environ[key])
    try:
        from PyQt6.QtCore import QSettings
        settings = QSettings("SegmentME", "SegmentME")
        for key in settings.allKeys():
            add(f"setting {key}", settings.value(key))
    except Exception:
        pass
    return "\n".join(lines)


def project_report(window) -> str:
    """Facts about the open project (no image or mask content)."""
    lines = []

    def add(key, value):
        lines.append(f"{key:<22} {value}")
    try:
        state = window.state_manager
        root = Path(getattr(window, "project_root", None) or state.project_root or "")
        db = Path(state.db.db_path)
        add("Project", root)
        add("Database", f"{db} ({db.stat().st_size / 1e6:.1f} MB)" if db.exists() else db)
        config = getattr(window, "config", None)
        if config is not None:
            add("Images folder", config.get_images_dir())
        paths = state.image_paths or []
        links = sum(1 for p in paths[:2000] if os.path.islink(p))
        add("Images", f"{len(paths)} ({links} linked of the first {min(len(paths), 2000)})")
        add("Masks", state.mask_manager.count())
        add("Classes", ", ".join(state.class_manager.get_all_class_names()) or "none")
        add("Disk (project)", _disk(root))
        try:
            scale = window.project_scale
            add("Scale", getattr(scale, "describe", lambda: scale)())
        except Exception:
            pass
    except Exception as exc:
        add("Project report failed", repr(exc))
    return "\n".join(lines)


def log_project_opened(window):
    logger.info("Project opened:\n%s", project_report(window))


# --------------------------------------------------------------------------
# Crashes and unexpected exits
# --------------------------------------------------------------------------
def _pid_alive(pid):
    try:
        return psutil.pid_exists(int(pid)) and int(pid) != os.getpid()
    except Exception:
        return False


def _report_previous_sessions(folder):
    """Sessions that did not end normally, and native crash dumps."""
    for marker in folder.glob("session-*.json"):
        try:
            info = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            info = {}
        pid = info.get("pid") or marker.stem.split("-", 1)[-1]
        if _pid_alive(pid):
            continue                       # another SegmentME running right now
        logger.warning("The previous session (started %s, version %s, pid %s) did not end "
                       "normally: it crashed, was killed, or the computer lost power. "
                       "Its last messages are above in this log.",
                       info.get("started", "?"), info.get("version", "?"), pid)
        crash = folder / f"crash-{pid}.log"
        if crash.exists() and crash.stat().st_size:
            try:
                text = crash.read_text(encoding="utf-8", errors="replace")
                logger.error("Native crash of the previous session (stack of every thread):\n%s",
                             text.strip())
            except OSError:
                pass
        crash.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
    for crash in folder.glob("crash-*.log"):            # dumps without a marker
        pid = crash.stem.split("-", 1)[-1]
        if _pid_alive(pid) or str(pid) == str(os.getpid()):
            continue
        if crash.stat().st_size:
            logger.error("Native crash report found (%s):\n%s", crash.name,
                         crash.read_text(encoding="utf-8", errors="replace").strip())
        crash.unlink(missing_ok=True)


def _start_crash_capture(folder):
    try:
        handle = open(folder / f"crash-{os.getpid()}.log", "w", encoding="utf-8")
        faulthandler.enable(file=handle, all_threads=True)
        _state["crash_file"] = handle
    except Exception as exc:
        logger.warning("Could not set up crash capture: %s", exc)
    marker = folder / f"session-{os.getpid()}.json"
    try:
        marker.write_text(json.dumps({"pid": os.getpid(), "version": __version__,
                                      "started": datetime.now().isoformat(timespec="seconds")}),
                          encoding="utf-8")
        _state["marker"] = marker
    except OSError:
        pass


def end_session():
    """Normal exit: say goodbye and remove the crash capture."""
    watchdog = _state.get("watchdog")
    if watchdog is not None:
        watchdog.stop()
    logger.info("Session ended normally")
    handle = _state.get("crash_file")
    if handle is not None:
        try:
            faulthandler.disable()
            handle.close()
            Path(handle.name).unlink(missing_ok=True)
        except Exception:
            pass
        _state["crash_file"] = None
    marker = _state.get("marker")
    if marker is not None:
        marker.unlink(missing_ok=True)
        _state["marker"] = None


# --------------------------------------------------------------------------
# Qt messages
# --------------------------------------------------------------------------
_QT_IGNORED = (
    "propagateSizeHints",             # harmless, printed by some platforms for every window
)


def _qt_message_handler(mode, context, message):
    from PyQt6.QtCore import QtMsgType
    if any(text in message for text in _QT_IGNORED):
        return
    level = {QtMsgType.QtDebugMsg: 10, QtMsgType.QtInfoMsg: 20, QtMsgType.QtWarningMsg: 30,
             QtMsgType.QtCriticalMsg: 40, QtMsgType.QtFatalMsg: 50}.get(mode, 30)
    where = f" ({context.file}:{context.line})" if context.file else ""
    get_logger("qt").log(level, "%s%s", message, where)


# --------------------------------------------------------------------------
# What the user does
# --------------------------------------------------------------------------
def _label(widget):
    from PyQt6.QtWidgets import QAbstractButton
    text = ""
    if isinstance(widget, QAbstractButton):
        text = widget.text().replace("&", "")
    text = text or widget.toolTip().split("\n")[0] or widget.objectName() or type(widget).__name__
    return " ".join(text.split())[:80]


def _window_title(widget):
    window = widget.window()
    return window.windowTitle() if window is not None else ""


class ActionRecorder:
    """An application-wide event filter that only watches (never consumes)."""

    def __init__(self):
        from PyQt6.QtCore import QObject

        class _Filter(QObject):
            def eventFilter(inner, obj, event):
                try:
                    self.observe(obj, event)
                except Exception:          # watching must never break the app
                    pass
                return False
        self.filter = _Filter()

    def observe(self, obj, event):
        from PyQt6.QtCore import QEvent, Qt
        from PyQt6.QtWidgets import QAbstractButton, QDialog, QMenu, QMessageBox
        kind = event.type()
        if kind == QEvent.Type.MouseButtonRelease:
            if event.button() != Qt.MouseButton.LeftButton:
                return
            if isinstance(obj, QMenu):
                act = obj.activeAction()
                if act is not None and act.isEnabled() and not act.isSeparator() and act.menu() is None:
                    action("Menu: %s", act.text().replace("&", ""))
            elif isinstance(obj, QAbstractButton) and obj.isEnabled() \
                    and obj.rect().contains(event.position().toPoint()):
                state = ""
                if obj.isCheckable():
                    state = " -> off" if obj.isChecked() else " -> on"
                action("Clicked '%s'%s in '%s'", _label(obj), state, _window_title(obj))
        elif kind == QEvent.Type.KeyPress and isinstance(obj, QMenu):
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and obj.activeAction():
                action("Menu (keyboard): %s", obj.activeAction().text().replace("&", ""))
        elif kind == QEvent.Type.Show and isinstance(obj, QDialog) and obj.isWindow():
            if isinstance(obj, QMessageBox):
                level = {QMessageBox.Icon.Critical: 40, QMessageBox.Icon.Warning: 30}.get(obj.icon(), 20)
                text = " ".join(f"{obj.text()} {obj.informativeText()}".split())
                actions_logger.log(level, "Message shown '%s': %s", obj.windowTitle(), text[:600])
                BREADCRUMBS.append((time.time(), f"Message '{obj.windowTitle()}': {text[:120]}"))
            else:
                action("Opened window '%s' (%s)", obj.windowTitle(), type(obj).__name__)
        elif kind == QEvent.Type.Hide and isinstance(obj, QDialog) and obj.isWindow() \
                and not isinstance(obj, QMessageBox):
            result = {QDialog.DialogCode.Accepted: "OK"}.get(obj.result(), "closed")
            action("Closed window '%s' (%s)", obj.windowTitle(), result)


# --------------------------------------------------------------------------
# Freezes
# --------------------------------------------------------------------------
class FreezeWatchdog:
    """Notices when the main thread stops handling events and logs where."""

    def __init__(self, seconds=FREEZE_SECONDS):
        from PyQt6.QtCore import QTimer
        self.seconds = seconds
        self.last_beat = time.monotonic()
        self.main_id = threading.main_thread().ident
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.beat)
        self.running = True
        self.thread = threading.Thread(target=self._watch, name="freeze-watchdog", daemon=True)

    def start(self):
        self.timer.start()
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        self.timer.stop()

    def beat(self):
        self.last_beat = time.monotonic()

    def _watch(self):
        frozen_since = None
        while self.running:
            time.sleep(0.25)
            silent = time.monotonic() - self.last_beat
            if silent >= self.seconds and frozen_since is None:
                frozen_since = self.last_beat
                frame = sys._current_frames().get(self.main_id)
                stack = "".join(traceback.format_stack(frame)) if frame is not None else "(unknown)"
                logger.warning("The window has not responded for %.1f s. The main thread is "
                               "busy here:\n%s", silent, stack)
            elif silent < self.seconds and frozen_since is not None:
                logger.warning("The window responds again after %.1f s",
                               self.last_beat - frozen_since)
                frozen_since = None


# --------------------------------------------------------------------------
# Start / report
# --------------------------------------------------------------------------
def start_session(app):
    """Call once, right after QApplication exists (main.py)."""
    folder = log_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    logger.info("=" * 70)
    logger.info("SegmentME %s starting (pid %d)", __version__, os.getpid())
    _report_previous_sessions(folder)
    _start_crash_capture(folder)
    logger.info("System:\n%s", system_report())

    from PyQt6.QtCore import qInstallMessageHandler
    qInstallMessageHandler(_qt_message_handler)

    recorder = ActionRecorder()
    app.installEventFilter(recorder.filter)
    _state["filter"] = recorder
    _state["watchdog"] = FreezeWatchdog().start()
    app.aboutToQuit.connect(end_session)
    return recorder


def save_report(path, window=None) -> Path:
    """A zip with the logs, the system and project reports and the recent
    actions -- what to send when something went wrong."""
    path = Path(path)
    folder = log_dir()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("system.txt", system_report())
        if window is not None:
            z.writestr("project.txt", project_report(window))
        z.writestr("recent-actions.txt", recent_actions(limit=len(BREADCRUMBS)) or "(none)")
        for f in sorted(folder.glob("segmentme.log*")) + sorted(folder.glob("crash-*.log")):
            try:
                z.write(f, arcname=f"logs/{f.name}")
            except OSError as exc:
                z.writestr(f"logs/{f.name}.unreadable.txt", str(exc))
    logger.info("Saved diagnostic report %s", path)
    return path
