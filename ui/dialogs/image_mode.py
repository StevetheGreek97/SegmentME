"""
The "Copy into project / Link to original files" choice used by every way of
adding images (New Project, Import Images).
See services/image_files.py for what a link is on each OS.
"""
import sys

from PyQt6.QtWidgets import (
    QButtonGroup, QDialog, QDialogButtonBox, QLabel, QRadioButton, QVBoxLayout, QWidget,
)

from services import image_files

_LINK_HINT = (
    "Linked images stay where they are and use no extra disk space. Don't move, "
    "rename or delete the originals, and the project can't be moved to another "
    "computer on its own.")
if sys.platform == "win32":
    _LINK_HINT += (" On Windows, images on another drive than the project are "
                   "copied instead.")


class ImageModeChoice(QWidget):
    """Two radio buttons plus a hint; mode() gives image_files.COPY or LINK."""

    # Remembered for the session, so the next import offers the same choice.
    _last_mode = image_files.COPY

    def __init__(self, parent=None):
        super().__init__(parent)
        self.copy_radio = QRadioButton("Copy images into the project folder")
        self.link_radio = QRadioButton("Link to the original files (no copy)")
        self._group = QButtonGroup(self)
        self._group.addButton(self.copy_radio)
        self._group.addButton(self.link_radio)
        self.hint = QLabel(_LINK_HINT)
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: gray;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.copy_radio)
        layout.addWidget(self.link_radio)
        layout.addWidget(self.hint)

        self.link_radio.toggled.connect(self.hint.setVisible)
        self.set_mode(ImageModeChoice._last_mode)
        self._group.buttonToggled.connect(
            lambda *_: setattr(ImageModeChoice, "_last_mode", self.mode()))

        self._fit_hint(440)

    def mode(self) -> str:
        return image_files.LINK if self.link_radio.isChecked() else image_files.COPY

    def set_mode(self, mode: str):
        (self.link_radio if mode == image_files.LINK else self.copy_radio).setChecked(True)
        self.hint.setVisible(mode == image_files.LINK)

    def _fit_hint(self, width):
        # A word-wrapped label doesn't report the height it needs, so the
        # dialog could squeeze it (and overlap its neighbours) otherwise.
        self.hint.setMinimumHeight(self.hint.heightForWidth(max(width, 200)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_hint(event.size().width())


class ImageModeDialog(QDialog):
    """Asks how to add `count` images; exec() then mode()."""

    def __init__(self, parent, count: int):
        super().__init__(parent)
        self.setWindowTitle("Import Images")
        self.choice = ImageModeChoice(self)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"How should the {count} image(s) be added?"))
        layout.addWidget(self.choice)
        layout.addWidget(buttons)
        self.setMinimumWidth(460)

    def mode(self) -> str:
        return self.choice.mode()
