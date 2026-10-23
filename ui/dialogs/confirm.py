"""Questions before stopping long work. The safe answer is the default, so
an Enter or Esc pressed without reading keeps the work going."""
from PyQt6.QtWidgets import QMessageBox


def ask_to_stop(parent, title, text, stop_label="Stop", keep_label="Keep Running") -> bool:
    """True only when the user clicks `stop_label`."""
    box = QMessageBox(QMessageBox.Icon.Question, title, text, parent=parent)
    stop = box.addButton(stop_label, QMessageBox.ButtonRole.DestructiveRole)
    keep = box.addButton(keep_label, QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(keep)
    box.setEscapeButton(keep)
    box.exec()
    return box.clickedButton() is stop
