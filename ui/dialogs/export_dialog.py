from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QComboBox,
    QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt
from ui.custom_components.custom_slider import ColorRangeSlider 


class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📤 Export Annotations")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Export format
        layout.addWidget(QLabel("Export format:"))
        self.format_box = QComboBox()
        self.format_box.addItems(["YOLO", "COCO", "Pascal VOC"])
        layout.addWidget(self.format_box)

        # Data split
        layout.addWidget(QLabel("Train / Val / Test split (%):"))
        self.slider = ColorRangeSlider()
        layout.addWidget(self.slider)

        self.split_label = QLabel()
        self.split_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.split_label)
        self.update_split_label()
        self.slider.splitChanged.connect(self.update_split_label)



        # Action buttons
        btn_layout = QHBoxLayout()
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.accept)
        btn_layout.addWidget(self.export_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_button)

        layout.addLayout(btn_layout)

    def update_split_label(self):
        left, right = self.slider.get_split()
        train = left
        val = right - left
        test = 100 - right
        self.split_label.setText(f"Train: {train}% | Val: {val}% | Test: {test}%")



    def get_settings(self):
        fmt = self.format_box.currentText().lower()
        train, val_end = self.slider.get_split()
        val = val_end - train
        test = 100 - val_end
        return {
            "format": fmt,
            "train": train,
            "val": val,
            "test": test,

        }
