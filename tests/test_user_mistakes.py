"""
Things a hurried user does by mistake, and the app surviving them:
deleting a project, images deleted by hand, closing long jobs, two windows
on one project, unplugged drives, training before exporting -- plus the
automatic backups that undo the rest.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_image


def _project(tmp_path, names=("a.png", "b.png"), name="proj"):
    root = tmp_path / name
    (root / ".segmentme").mkdir(parents=True)
    for n in names:
        make_image(root / "images" / n)
    (root / f"{name}.SEproj").write_text(json.dumps({"name": name, "images": list(names)}))
    return root


def _square(x=5):
    return np.array([[x, x], [x + 10, x], [x + 10, x + 10], [x, x + 10]], dtype=np.float32)


@pytest.fixture
def window(qapp, tmp_path, no_popups):
    from ui.main_window import MainApp
    from services.recent_projects import initialize_project
    root = _project(tmp_path)
    db = str(root / ".segmentme" / "masks.db")
    win = MainApp(db_path=db)
    initialize_project(win, db)
    qapp.processEvents()
    win.popups = no_popups
    yield win
    win.state_manager.db.conn.close()


# ---------- 1. Delete project ----------
def test_delete_never_touches_a_folder_that_is_not_a_project(tmp_path, monkeypatch):
    from services import recent_projects as rp
    monkeypatch.setattr(rp, "remove_recent_project", lambda db_path: None)
    # An old masks.db lying in a folder full of other work
    work = tmp_path / "Projects"
    (work / "other_project").mkdir(parents=True)
    (work / "other_project" / "thesis.docx").write_text("years of work")
    (work / "masks.db").write_bytes(b"SQLite format 3\0")
    assert rp.project_root_to_delete(str(work / "masks.db")) is None
    assert rp.delete_project(str(work / "masks.db"), permanently=True) == rp.NOT_A_PROJECT
    assert (work / "other_project" / "thesis.docx").exists()

    # .segmentme without a .SEproj next to it: not trusted either
    odd = tmp_path / "odd"
    (odd / ".segmentme").mkdir(parents=True)
    (odd / ".segmentme" / "masks.db").write_bytes(b"x")
    (odd / "keep.txt").write_text("x")
    assert rp.delete_project(str(odd / ".segmentme" / "masks.db"), permanently=True) == rp.NOT_A_PROJECT
    assert (odd / "keep.txt").exists()

    assert rp.delete_project(str(tmp_path / "gone" / ".segmentme" / "masks.db")) == rp.ALREADY_GONE


def test_delete_moves_to_trash_and_never_deletes_silently(tmp_path, monkeypatch):
    from PyQt6.QtCore import QFile
    from services import recent_projects as rp
    monkeypatch.setattr(rp, "remove_recent_project", lambda db_path: None)
    root = _project(tmp_path)
    (root / ".segmentme" / "masks.db").write_bytes(b"x")
    db = str(root / ".segmentme" / "masks.db")

    monkeypatch.setattr(QFile, "moveToTrash", staticmethod(lambda path: False))
    assert rp.delete_project(db) == rp.NO_TRASH          # nothing deleted; the UI asks
    assert root.exists()

    trashed = []
    monkeypatch.setattr(QFile, "moveToTrash", staticmethod(lambda path: trashed.append(path) or True))
    assert rp.delete_project(db) == rp.TRASHED
    assert trashed == [str(root)]


# ---------- 2. Images deleted by hand ----------
def test_deleted_image_removes_its_masks_after_a_warning(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from services import backups
    from services.recent_projects import initialize_project
    state = window.state_manager
    state.mask_manager.save_mask(_square(), "a", "Object")
    state.mask_manager.save_mask(_square(30), "b", "Object")
    root = tmp_path / "proj"
    (root / "images" / "b.png").unlink()                # deleted by hand

    # Cancel: nothing removed
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel))
    initialize_project(window, state.db.db_path)
    assert state.mask_manager.count() == 2

    # OK: the deleted image's masks go, after a backup
    shown = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda p, title, text, *a, **k:
                                     shown.append(text) or QMessageBox.StandardButton.Ok))
    initialize_project(window, state.db.db_path)
    assert "b.png" in shown[0] and "removed from the database" in shown[0]
    assert state.mask_manager.count() == 1
    assert state.mask_manager.class_of(state.mask_manager.load_masks("a")[0][0]) == "Object"
    saved = backups.list_backups(state.db.db_path)
    assert saved[0].reason == "before removing missing images" and saved[0].masks == 2
    assert len(state.image_paths) == 1


def test_images_folder_gone_does_not_crash(window, tmp_path):
    from services.recent_projects import initialize_project
    root = tmp_path / "proj"
    (root / "images").rename(root / "images_moved")
    # The fake question answers OK (not Yes): "don't locate it"
    initialize_project(window, window.state_manager.db.db_path)
    assert window.project_root == str(root)


# ---------- 3./4. Long jobs are not stopped by a stray click ----------
def test_training_window_asks_before_stopping(qapp, monkeypatch):
    from ui.dialogs import confirm
    from ui.dialogs.training_monitor import TrainingMonitor
    monitor = TrainingMonitor()
    stopped = []
    monitor.set_stop_callback(lambda: stopped.append(True))
    monkeypatch.setattr(monitor, "training", lambda: True)
    answers = []
    monkeypatch.setattr(confirm, "ask_to_stop", lambda *a, **k: answers.append(a[1]) or False)
    monitor.show()
    monitor.reject()                       # Esc
    monitor.close()                        # the X button
    monitor.stop_btn.click()               # Stop
    assert stopped == [] and monitor.isVisible()
    assert answers == ["Stop Training?"] * 3
    monkeypatch.setattr(confirm, "ask_to_stop", lambda *a, **k: True)
    monitor.stop_btn.click()
    assert stopped == [True]
    monitor.timer.stop()
    monitor.deleteLater()


def test_inference_window_asks_before_cancelling(qapp, monkeypatch):
    from ui.dialogs import confirm
    from ui.dialogs.progress import InferenceProgressDialog
    dialog = InferenceProgressDialog(None, 10, "Running")
    cancelled = []
    dialog.canceled.connect(lambda: cancelled.append(True))
    dialog.set_progress(3)
    texts = []
    monkeypatch.setattr(confirm, "ask_to_stop", lambda parent, title, text, **k: texts.append(text) or False)
    dialog.show()
    dialog.reject()                        # Esc
    dialog.cancel_button.click()
    assert cancelled == [] and dialog.isVisible()
    assert "3 of 10" in texts[0]
    monkeypatch.setattr(confirm, "ask_to_stop", lambda *a, **k: True)
    dialog.cancel_button.click()
    assert cancelled == [True]


def test_closing_the_window_during_inference_asks(window, monkeypatch):
    from PyQt6.QtGui import QCloseEvent
    from ui.dialogs import confirm

    class Running:
        running = True

        def progress_text(self):
            return "5 of 9 image(s) done"

        def stop_inference(self):
            pass
    window.inference.manager = Running()
    asked = []
    monkeypatch.setattr(confirm, "ask_to_stop", lambda parent, title, text, **k: asked.append(text) or False)
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted() and "5 of 9" in asked[0]
    window.inference.manager = None


# ---------- 5. Backups ----------
def test_backup_and_restore_undo_a_deleted_class(window, monkeypatch):
    from services import backups
    state = window.state_manager
    from PyQt6.QtGui import QColor
    state.class_manager.add_class("Daphnia", QColor("#ff0000"))
    for i in range(3):
        state.mask_manager.save_mask(_square(5 + i * 20), "a", "Daphnia")
    db_path = state.db.db_path

    # Deleting the class (answering Yes) makes a backup first
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Yes)
    window.sidebar.class_dropdown.setCurrentText("Daphnia")
    window.sidebar.remove_selected_class()
    assert state.mask_manager.count() == 0
    saved = backups.list_backups(db_path)
    assert saved and saved[0].reason == "before deleting a class" and saved[0].masks == 3

    backups.restore(saved[0], state.db.conn, db_path)
    window.after_restore()
    assert state.mask_manager.count() == 3
    assert "Daphnia" in state.class_manager.get_all_class_names()
    reasons = [b.reason for b in backups.list_backups(db_path)]
    assert "before restore" in reasons                     # the restore can be undone


def test_backups_are_pruned_and_skip_a_full_disk(tmp_path, monkeypatch):
    import shutil
    import sqlite3
    from services import backups
    db = tmp_path / "proj" / ".segmentme" / "masks.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE masks (id INTEGER PRIMARY KEY, image_name TEXT)")
    conn.execute("CREATE TABLE classes (id INTEGER PRIMARY KEY, class_name TEXT)")
    conn.commit()
    conn.close()
    for _ in range(backups.KEEP + 3):
        assert backups.make_backup(db) is not None
    assert len(backups.list_backups(db)) == backups.KEEP

    class Full:
        free = 1000
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Full())
    assert backups.make_backup(db) is None                 # never fills the disk
    assert backups.make_backup(tmp_path / "missing.db") is None


def test_window_backs_up_when_opened_and_after_changes(window, monkeypatch, qapp):
    from services import backups
    made = []
    monkeypatch.setattr(backups, "make_backup", lambda db, reason="x": made.append(reason))
    window._backup_if_changed()
    window.state_manager.mask_manager.save_mask(_square(), "a", "Object")
    window._backup_if_changed()
    window._backup_if_changed()                            # nothing new since
    time.sleep(0.2)
    assert made == ["automatic"]
    assert any(a.text() == "Restore Backup…" for a in window.menuBar().actions()[0].menu().actions())


# ---------- 6. Same project twice ----------
def test_second_window_on_the_same_project_is_noticed(qapp, tmp_path):
    from ui.project_lock import lock_project
    db = _project(tmp_path) / ".segmentme" / "masks.db"
    ok, first = lock_project(db, ask=False)
    assert ok and first is not None
    ok2, second = lock_project(db, ask=False)
    assert not ok2
    first.unlock()
    ok3, third = lock_project(db, ask=False)
    assert ok3
    third.unlock()


# ---------- 7. Unplugged drive ----------
def test_recent_projects_on_an_unplugged_drive_stay_listed(tmp_path, monkeypatch):
    from services import recent_projects as rp
    monkeypatch.setattr(rp, "get_recent_file", lambda: tmp_path / "recent.json")
    there = _project(tmp_path, name="here") / ".segmentme" / "masks.db"
    there.write_bytes(b"x")
    usb = "/media/usb-stick/proj/.segmentme/masks.db"
    (tmp_path / "recent.json").write_text(json.dumps([str(there), usb]))
    assert rp.load_recent_projects() == [str(there), usb]
    assert json.loads((tmp_path / "recent.json").read_text()) == [str(there), usb]


def test_recent_item_shows_not_found(qapp):
    from ui.recent_item import RecentItemWidget
    item = RecentItemWidget("proj", "/media/usb/proj/.segmentme/masks.db",
                            lambda p: None, lambda p: None, missing=True)
    assert item.del_btn.text() == "Remove"
    assert "Not found" in item.toolTip()


# ---------- 8. Training before exporting ----------
def test_training_notices_an_old_export(window, tmp_path):
    from ui.dialogs.training_dialog import TrainingDialog
    root = tmp_path / "proj"
    dialog = TrainingDialog(window)
    assert not dialog.export_is_outdated()                 # never exported
    (root / "data.yaml").write_text("names: {}")
    old = time.time() - 3600
    os.utime(root / "data.yaml", (old, old))
    window.state_manager.mask_manager.save_mask(_square(), "a", "Object")
    assert dialog.export_is_outdated()
    (root / "data.yaml").write_text("names: {}")           # exported again just now
    assert not dialog.export_is_outdated()


def test_training_without_export_offers_to_export(window, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from ui.dialogs.training_dialog import TrainingDialog
    from ui.flows import data_io
    dialog = TrainingDialog(window)
    exported = []
    monkeypatch.setattr(data_io, "export_annotations_dataset", lambda w: exported.append(w))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: self.buttons()[0])  # Export Now
    accepted = []
    monkeypatch.setattr(dialog, "accept", lambda: accepted.append(True))
    dialog.validate_and_accept()
    assert exported == [window]
    assert accepted == []          # the fake export wrote nothing: still not trainable
