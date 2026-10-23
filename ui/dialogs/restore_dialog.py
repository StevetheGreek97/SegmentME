"""
File > Restore Backup…: the project's automatic database copies (see
services/backups.py), newest first, with how many masks each holds.
Restoring replaces the current masks and classes; the current state is
copied first, so a restore can be undone the same way.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QVBoxLayout,
)

from services import backups
from services.logger import get_logger

logger = get_logger(__name__)

REASONS = {
    "opened": "project opened",
    "automatic": "while working",
    "before deleting a class": "before deleting a class",
    "before import": "before importing annotations",
    "before restore": "before a restore",
    "before removing missing images": "before removing masks of deleted images",
}


class RestoreBackupDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window_ = window
        self.setWindowTitle("Restore Backup")
        self.resize(560, 420)
        self.db_path = window.state_manager.db.db_path
        self.restored = None

        v = QVBoxLayout(self)
        state = window.state_manager
        now_masks = state.mask_manager.count()
        intro = QLabel(
            "SegmentME keeps copies of this project's masks and classes: when it opens, "
            "every 20 minutes while you work, and before risky actions.\n\n"
            f"Now: {now_masks:,} masks. Pick a copy to go back to:")
        intro.setWordWrap(True)
        v.addWidget(intro)

        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _item: self.restore_selected())
        v.addWidget(self.list, 1)
        self.backups = backups.list_backups(self.db_path)
        for b in self.backups:
            reason = REASONS.get(b.reason, b.reason or "")
            item = QListWidgetItem(
                f"{b.label}   ·   {b.masks:,} masks on {b.images:,} images, {b.classes} classes"
                + (f"   ({reason})" if reason else ""))
            item.setData(Qt.ItemDataRole.UserRole, b)
            self.list.addItem(item)
        if not self.backups:
            self.list.addItem("No backups yet: the first is made when the project opens.")
            self.list.setEnabled(False)

        row = QHBoxLayout()
        self.restore_button = QPushButton("Restore…")
        self.restore_button.setAutoDefault(False)
        self.restore_button.setEnabled(bool(self.backups))
        self.restore_button.clicked.connect(self.restore_selected)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        row.addStretch(1)
        row.addWidget(self.restore_button)
        row.addWidget(close)
        v.addLayout(row)
        if self.backups:
            self.list.setCurrentRow(0)

    def selected(self):
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def restore_selected(self):
        backup = self.selected()
        if backup is None:
            return False
        now = self.window_.state_manager.mask_manager.count()
        box = QMessageBox(QMessageBox.Icon.Warning, "Restore Backup?",
                          f"Go back to the copy from {backup.label}?", parent=self)
        box.setInformativeText(
            f"The project now has {now:,} masks; the copy has {backup.masks:,}. "
            "Masks and classes are replaced by the copy's.\n\n"
            "The current state is saved as a backup first, so you can come back to it.")
        go = box.addButton("Restore", QMessageBox.ButtonRole.DestructiveRole)
        keep = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(keep)
        box.setEscapeButton(keep)
        box.exec()
        if box.clickedButton() is not go:
            return False
        try:
            backups.restore(backup, self.window_.state_manager.db.conn, self.db_path)
        except Exception as exc:
            logger.exception("Restore failed")
            QMessageBox.critical(self, "Restore Failed",
                                 f"The backup could not be restored; nothing was changed.\n\n{exc}")
            return False
        self.restored = backup
        self.window_.after_restore()
        QMessageBox.information(self, "Restored",
                                f"Restored the copy from {backup.label} ({backup.masks:,} masks).")
        self.accept()
        return True
