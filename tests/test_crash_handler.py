"""The safety net: an unexpected error shows one message and the app
keeps running, instead of PyQt6 closing it."""
import sys

import pytest


def _raise(exc):
    try:
        raise exc
    except Exception:
        return sys.exc_info()


def test_error_is_logged_and_shown_once(qapp, monkeypatch):
    from ui import crash_handler
    shown = []
    monkeypatch.setattr(crash_handler, "show_error_message",
                        lambda t, e, details: shown.append((t, str(e), details)))
    handler = crash_handler.CrashHandler()
    info = _raise(ValueError("bad value"))
    handler(*info)
    assert len(shown) == 1 and shown[0][1] == "bad value"
    assert "Traceback" in shown[0][2]
    # The same error again (e.g. on every mouse move): logged, not shown
    for _ in range(20):
        handler(*info)
    assert len(shown) == 1 and handler.count == 21
    # A different error is shown
    handler(*_raise(KeyError("other")))
    assert len(shown) == 2


def test_no_second_message_while_one_is_open(qapp, monkeypatch):
    from ui import crash_handler
    handler = crash_handler.CrashHandler()
    inner = []

    def show(t, e, details):
        inner.append(str(e))
        handler(*_raise(RuntimeError("while showing")))   # error inside the message's loop
    monkeypatch.setattr(crash_handler, "show_error_message", show)
    handler(*_raise(ValueError("first")))
    assert inner == ["first"]


def test_handler_never_fails_itself(qapp, monkeypatch):
    from ui import crash_handler

    def broken(*args):
        raise OSError("no display")
    monkeypatch.setattr(crash_handler, "show_error_message", broken)
    crash_handler.CrashHandler()(*_raise(ValueError("x")))   # must not raise


def test_slot_error_keeps_app_running(qapp, monkeypatch):
    """An error in a Qt slot goes to the handler instead of aborting."""
    from PyQt6.QtWidgets import QPushButton
    from ui import crash_handler
    seen = []
    monkeypatch.setattr(crash_handler, "show_error_message", lambda t, e, d: seen.append(str(e)))
    monkeypatch.setattr(sys, "excepthook", crash_handler.CrashHandler())
    button = QPushButton()
    button.clicked.connect(lambda: 1 / 0)
    button.click()
    assert seen == ["division by zero"]


def test_install_sets_hooks(monkeypatch):
    import threading
    from ui import crash_handler
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    handler = crash_handler.install()
    assert sys.excepthook is handler
    assert threading.excepthook is crash_handler._thread_hook
