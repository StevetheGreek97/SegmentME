"""
Image navigation above the image: a counter ("12 / 480"), a searchable
dropdown of the image names, and a slider.

Typing in the dropdown filters the names by any part (ignoring case);
picking one moves the slider there, so every way of changing the image goes
through the same path (the slider's short delay, then display_image).
"""
import os

import qtawesome as qta
from PyQt6.QtCore import QStringListModel, Qt, QTimer
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QComboBox, QCompleter, QHBoxLayout, QLabel, QLineEdit, QSlider, QVBoxLayout, QWidget,
)


class ImageSlider(QWidget):
    LOAD_DELAY_MS = 150   # wait for the slider to settle before loading a (big) image

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self._state = parent.state_manager

        # Names in the dropdown belong to this image list (identity check:
        # a new project / import sets a new list).
        self._names_for = None
        self._index_of = {}           # image name -> index

        # --- counter + searchable dropdown --------------------------------
        self.label = QLabel(self)
        self.label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.label.setMinimumWidth(self.label.fontMetrics().horizontalAdvance("00000 / 00000"))

        self._names = QStringListModel(self)
        self.image_picker = QComboBox(self)
        self.image_picker.setModel(self._names)
        self.image_picker.setEditable(True)
        self.image_picker.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.image_picker.setMaxVisibleItems(20)
        self.image_picker.setMinimumContentsLength(28)
        self.image_picker.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.image_picker.view().setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.image_picker.setToolTip("Pick an image, or type part of its name to search")
        edit = self.image_picker.lineEdit()
        edit.setPlaceholderText("Search images…")
        edit.addAction(qta.icon("fa5s.search", color="gray"), QLineEdit.ActionPosition.LeadingPosition)

        completer = QCompleter(self._names, self.image_picker)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setMaxVisibleItems(20)
        self.image_picker.setCompleter(completer)

        self.image_picker.activated.connect(self._go_to_index)
        completer.activated[str].connect(self._go_to_name)
        edit.editingFinished.connect(self._restore_name)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch(1)
        row.addWidget(self.label)
        row.addWidget(self.image_picker)
        row.addStretch(1)

        # --- slider -------------------------------------------------------
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, 0)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)

        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(self.slider)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_image)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self._state.image_changed.connect(self.update_slider)

        self.update_label(0)

    # ------------------------------------------------------------- names
    def _sync_names(self):
        """Refill the dropdown when the project's image list changed."""
        paths = self._state.image_paths
        if paths is self._names_for and len(paths) == len(self._index_of):
            return
        names = [os.path.basename(p) for p in paths]
        self._names_for = paths
        self._index_of = {name: i for i, name in enumerate(names)}
        self.image_picker.blockSignals(True)
        self._names.setStringList(names)      # one model reset, not one per row
        self.image_picker.blockSignals(False)

    def _go_to_index(self, index):
        """Show image `index` (as moving the slider there does)."""
        if 0 <= index < len(self._state.image_paths):
            if index == self.slider.value():
                self.update_label(index)      # same image: just tidy the text
            else:
                self.slider.setValue(index)
        self.parent.image_display.setFocus()  # keyboard shortcuts work again

    def _go_to_name(self, name):
        index = self._index_of.get(name)
        if index is not None:
            self._go_to_index(index)

    def _restore_name(self):
        """Typed text that isn't an image name: show the current name again."""
        if self.image_picker.currentText() not in self._index_of:
            self.update_label(self.slider.value())

    # ------------------------------------------------------------- slider
    def set_image_count(self, count):
        """The image list changed: new range, names and position."""
        self.slider.setMaximum(max(count - 1, 0))
        self._sync_names()
        self.update_slider(self._state.current_image_path)

    def on_slider_changed(self, value):
        """Slider moved: update the text now, load the image once it settles."""
        self.update_label(value)
        self.timer.start(self.LOAD_DELAY_MS)

    def update_image(self):
        """Load the image at the slider position."""
        if self._state.image_paths:
            self._state.image_index = self.slider.value()
            self.parent.image_display.display_image(self._state.current_image_path)

    def update_slider(self, image_path):
        """The image changed elsewhere (next/previous, import): follow it."""
        paths = self._state.image_paths
        if image_path == self._state.current_image_path:
            index = self._state.image_index           # no search through the list
        elif image_path in paths:
            index = paths.index(image_path)
        else:
            return
        self.slider.blockSignals(True)  # no reload: the image is already shown
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self.update_label(index)

    def update_label(self, index):
        """Counter "n / N" and the image's name in the dropdown."""
        self._sync_names()
        total = len(self._state.image_paths)
        self.label.setText(f"{index + 1} / {total}" if total else "0 / 0")
        self.image_picker.blockSignals(True)
        if 0 <= index < total:
            self.image_picker.setCurrentIndex(index)
        else:
            self.image_picker.setCurrentIndex(-1)
            self.image_picker.setEditText("")
        self.image_picker.blockSignals(False)
