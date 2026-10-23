# ui/widgets/recent_item.py
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QSizePolicy
from PyQt6.QtCore import Qt
import os

class RecentItemWidget(QWidget):
    def __init__(self, project_name: str, db_path: str, on_open, on_delete, parent=None,
                 missing: bool = False):
        super().__init__(parent)
        self.db_path = db_path
        self.missing = missing
        project_root = os.path.dirname(os.path.dirname(db_path))
        self.setToolTip(project_root + ("\n\nNot found: moved, deleted, or on a drive "
                                        "that is not connected." if missing else ""))

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(8)

        title = QLabel(project_name + ("  (not found)" if missing else ""))
        title.setStyleSheet("font-weight:600;" + (" color: gray;" if missing else ""))
        title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        open_btn = QPushButton("Open")
        # A project that is not found can only be taken off the list
        del_btn  = QPushButton("Remove" if missing else "Delete")
        self.open_btn, self.del_btn = open_btn, del_btn
        open_btn.setObjectName("action")
        del_btn.setObjectName("action")


        # Let the buttons be wide enough for text (padding is 14px left/right)
        for b in (open_btn, del_btn):
            b.setMinimumWidth(90)                  # <<< was 20
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        open_btn.clicked.connect(lambda: on_open(self.db_path))
        del_btn.clicked.connect(lambda: on_delete(self.db_path))

        text_box = QWidget()
        tl = QHBoxLayout(text_box)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(20)
        tl.addWidget(title)

        lay.addWidget(text_box, 1)   # <<< give text area stretch so buttons don’t get squeezed
        lay.addWidget(open_btn, 0)
        lay.addWidget(del_btn, 0)
        self.setMinimumHeight(50)

