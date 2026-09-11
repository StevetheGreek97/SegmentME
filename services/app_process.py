"""Re-launching this application as a helper subprocess.

The inference and training workers are the app itself started with a
flag (`--inference-worker`, `--training-worker`) that main.py dispatches
before importing anything GUI-related. Spawning ourselves -- rather than
a console script such as `yolo` -- is what makes the same code work both
for `python main.py` and for a frozen PyInstaller executable, where no
console scripts exist and no separate Python interpreter is installed.
"""
import subprocess
import sys
from pathlib import Path


def self_command(flag: str):
    """(program, arguments) that re-launch this app in worker mode."""
    if getattr(sys, "frozen", False):
        return sys.executable, [flag]
    main_py = Path(__file__).resolve().parent.parent / "main.py"
    return sys.executable, [str(main_py), flag]


def popen_kwargs() -> dict:
    """Extra Popen keyword arguments for helper processes.

    On Windows, keep a helper started from the (windowed) app from
    flashing a console window of its own.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}
