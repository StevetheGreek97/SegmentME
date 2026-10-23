from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QSpinBox, QCheckBox, QPushButton,
    QLineEdit, QTabWidget, QComboBox, QHBoxLayout, QDoubleSpinBox,
    QMessageBox, QScrollArea, QWidget, QFormLayout, QGroupBox
)
from core.trainer.trainer_settings import TrainingSettings
from services.hardware import auto_detect_device 
from pathlib import Path

from services.logger import get_logger

logger = get_logger(__name__)

class TrainingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("🧠 Train Custom Model")
        self.setMinimumSize(650, 650)
        self.project_root = Path(self.parent.state_manager.project_root)

        self.models_dir = str(self.project_root / ".models")

        self.tabs = QTabWidget(self)
        self.training_tab = QWidget()
        self.augment_tab = QWidget()
        self.tabs.addTab(self.training_tab, "Training Settings")
        self.tabs.addTab(self.augment_tab, "Augmentation Settings")

        self._build_training_tab()
        self._build_augment_tab()

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

        # Buttons
        btn_layout = QHBoxLayout()
        self.train_button = QPushButton("🚀 Train")
        self.train_button.clicked.connect(self.validate_and_accept)
        btn_layout.addWidget(self.train_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_button)

        layout.addLayout(btn_layout)

    def _build_training_tab(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vbox = QVBoxLayout(container)

        form = QFormLayout()

        # Dropdown for selecting model version and type
        self.model_selector = QComboBox()
        self.available_models = [
            "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt","yolov8l-seg.pt", "yolov8x-seg.pt", 
            "yolov9c-seg.pt", "yolov9e-seg.pt",
            "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt" 


        ]
        self.model_selector.addItems(self.available_models)
        self.model_selector.setCurrentText("yolov8n-seg.pt")
        form.addRow("Select Model:", self.model_selector)



        self.data_path_input = QLabel()
        self.data_path_input.setText(str(self.project_root / "data.yaml"))
        form.addRow("Dataset YAML path:", self.data_path_input)

        self.epoch_box = QSpinBox()
        self.epoch_box.setRange(1, 1000)
        self.epoch_box.setValue(100)
        form.addRow("Epochs:", self.epoch_box)

        self.batch_box = QDoubleSpinBox()
        self.batch_box.setRange(-1.0, 1024.0)
        self.batch_box.setValue(-1)
        form.addRow("Batch size:", self.batch_box)

        self.imgsz_box = QSpinBox()
        self.imgsz_box.setRange(64, 20048)
        self.imgsz_box.setValue(640)
        form.addRow("Image size:", self.imgsz_box)

        self.device_label = QLabel()
        self.device_label.setStyleSheet("color: green; font-weight: bold")
        self.device_label.setText("Detecting...")

        form.addRow("Device:", self.device_label)

        # Immediately detect on dialog load
        device = auto_detect_device(self)
        self.device_label.setText(str(device))
        self.detected_device = device


        self.optimizer_input = QComboBox()
        self.optimizer_input.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        form.addRow("Optimizer:", self.optimizer_input)


    
        self.project_name = QLineEdit()
        self.project_name.setText('MyCooltTrainingProject')
        form.addRow("Project name:", self.project_name)

  
        self.run_name = QLineEdit()
        self.run_name.setText('runs')
        form.addRow("Run name:", self.run_name)
        
        self.output_dir_input = QLabel()
        self.output_dir_input.setText(str(self.project_root / 'trainings' / self.project_name.text() / self.run_name.text()))
        self.project_name.textChanged.connect(self.update_output_path)
        self.run_name.textChanged.connect(self.update_output_path)

        form.addRow("Output directory:", self.output_dir_input)


        vbox.addLayout(form)

        # Advanced
        self.advanced_training_checkbox = QCheckBox("Show Advanced Settings")
        self.advanced_training_checkbox.toggled.connect(lambda checked: advanced_box.setVisible(checked))
        vbox.addWidget(self.advanced_training_checkbox)

        advanced_box = QGroupBox("Advanced Training Settings")
        advanced_box.setVisible(False)
        advanced_form = QFormLayout()

        self.time_limit = QDoubleSpinBox()
        self.time_limit.setRange(0, 168)
        self.time_limit.setSingleStep(0.5)
        self.time_limit.setSuffix(" hrs")
        advanced_form.addRow("Max training time:", self.time_limit)

        self.patience_box = QSpinBox()
        self.patience_box.setRange(0, 300)
        self.patience_box.setValue(100)
        advanced_form.addRow("Patience:", self.patience_box)

        self.lr0 = QDoubleSpinBox()
        self.lr0.setRange(0.00001, 1.0)
        self.lr0.setValue(0.01)
        advanced_form.addRow("Initial LR:", self.lr0)

        self.lrf = QDoubleSpinBox()
        self.lrf.setRange(0.00001, 1.0)
        self.lrf.setValue(0.01)
        advanced_form.addRow("Final LR Fraction:", self.lrf)

        self.momentum = QDoubleSpinBox()
        self.momentum.setRange(0.0, 1.0)
        self.momentum.setValue(0.937)
        advanced_form.addRow("Momentum:", self.momentum)

        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setRange(0.0, 0.1)
        self.weight_decay.setValue(0.0005)
        advanced_form.addRow("Weight Decay:", self.weight_decay)

        advanced_box.setLayout(advanced_form)
        vbox.addWidget(advanced_box)

        scroll.setWidget(container)
        layout.addWidget(scroll)
        self.training_tab.setLayout(layout)
    def update_output_path(self):
        self.output_dir_input.setText(str(self.project_root / 'trainings' /  self.project_name.text() / self.run_name.text()))

    def _build_augment_tab(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vbox = QVBoxLayout(container)
        form = QFormLayout()

        self.hsv_h = QDoubleSpinBox()
        self.hsv_h.setRange(0.0, 1.0)
        self.hsv_h.setValue(0.015)
        form.addRow("Hue (hsv_h):", self.hsv_h)

        self.hsv_s = QDoubleSpinBox()
        self.hsv_s.setRange(0.0, 1.0)
        self.hsv_s.setValue(0.7)
        form.addRow("Saturation (hsv_s):", self.hsv_s)

        self.hsv_v = QDoubleSpinBox()
        self.hsv_v.setRange(0.0, 1.0)
        self.hsv_v.setValue(0.4)
        form.addRow("Value/Brightness (hsv_v):", self.hsv_v)

        self.degrees = QDoubleSpinBox()
        self.degrees.setRange(0.0, 180.0)
        form.addRow("Rotation (degrees):", self.degrees)

        self.translate = QDoubleSpinBox()
        self.translate.setRange(0.0, 1.0)
        self.translate.setValue(0.1)
        form.addRow("Translation (translate):", self.translate)

        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.0, 10.0)
        self.scale.setValue(0.5)
        form.addRow("Scale (scale):", self.scale)

        self.shear = QDoubleSpinBox()
        self.shear.setRange(-180.0, 180.0)
        form.addRow("Shear (shear):", self.shear)

        self.perspective = QDoubleSpinBox()
        self.perspective.setRange(0.0, 0.001)
        form.addRow("Perspective:", self.perspective)

        self.flipud = QDoubleSpinBox()
        self.flipud.setRange(0.0, 1.0)
        form.addRow("Vertical Flip Prob (flipud):", self.flipud)

        self.fliplr = QDoubleSpinBox()
        self.fliplr.setRange(0.0, 1.0)
        self.fliplr.setValue(0.5)
        form.addRow("Horizontal Flip Prob (fliplr):", self.fliplr)

        self.bgr = QDoubleSpinBox()
        self.bgr.setRange(0.0, 1.0)
        form.addRow("BGR Flip Prob (bgr):", self.bgr)

        self.mosaic = QDoubleSpinBox()
        self.mosaic.setRange(0.0, 1.0)
        self.mosaic.setValue(1.0)
        form.addRow("Mosaic:", self.mosaic)

        vbox.addLayout(form)

        # Advanced
        self.advanced_aug_checkbox = QCheckBox("Show Advanced Settings")
        self.advanced_aug_checkbox.toggled.connect(lambda checked: advanced_box.setVisible(checked))
        vbox.addWidget(self.advanced_aug_checkbox)

        advanced_box = QGroupBox("Advanced Augmentations")
        advanced_box.setVisible(False)
        advanced_form = QFormLayout()

        self.mixup = QDoubleSpinBox()
        self.mixup.setRange(0.0, 1.0)
        advanced_form.addRow("Mixup:", self.mixup)

        self.cutmix = QDoubleSpinBox()
        self.cutmix.setRange(0.0, 1.0)
        advanced_form.addRow("CutMix:", self.cutmix)

        self.copy_paste = QDoubleSpinBox()
        self.copy_paste.setRange(0.0, 1.0)
        advanced_form.addRow("CopyPaste:", self.copy_paste)

        self.copy_paste_mode = QComboBox()
        self.copy_paste_mode.addItems(["flip", "mixup"])
        advanced_form.addRow("CopyPaste Mode:", self.copy_paste_mode)

        self.auto_augment = QComboBox()
        self.auto_augment.addItems(["none", "randaugment", "autoaugment", "augmix"])
        advanced_form.addRow("AutoAugment:", self.auto_augment)

        self.erasing = QDoubleSpinBox()
        self.erasing.setRange(0.0, 0.9)
        self.erasing.setValue(0.4)
        advanced_form.addRow("Random Erasing:", self.erasing)

        advanced_box.setLayout(advanced_form)
        vbox.addWidget(advanced_box)

        scroll.setWidget(container)
        layout.addWidget(scroll)
        self.augment_tab.setLayout(layout)

    def validate_and_accept(self):
        """Training learns from the exported dataset (File > Export
        annotations). Offer to export when that was never done, or when the
        masks changed after the last export (it would train on old masks)."""
        missing = self.validate_project_structure(self.project_root)
        if missing:
            logger.info("Training needs an export first; missing %s", missing)
            if not self._ask_export(
                    "Export Needed",
                    "Training learns from your annotations exported as a dataset, and they "
                    "have not been exported yet.\n\nExport them now?",
                    "Export Now", None):
                return
        elif self.export_is_outdated():
            answer = self._ask_export(
                "Annotations Changed",
                "You changed masks or classes after the last export, so training would use "
                "the old annotations.\n\nExport them again first?",
                "Export Again", "Train on Old Export")
            if answer is None:                       # Cancel
                return
        if self.validate_project_structure(self.project_root):
            return                                   # export cancelled or failed
        self.accept()

    def _ask_export(self, title, text, export_label, skip_label):
        """True after exporting, False when skipped (skip_label), None on Cancel."""
        box = QMessageBox(QMessageBox.Icon.Question, title, text, parent=self)
        export = box.addButton(export_label, QMessageBox.ButtonRole.AcceptRole)
        skip = box.addButton(skip_label, QMessageBox.ButtonRole.ActionRole) if skip_label else None
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(export)
        box.setEscapeButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is export:
            from ui.flows import data_io
            data_io.export_annotations_dataset(self.parent)
            return True
        if skip is not None and clicked is skip:
            return False
        return None

    def export_is_outdated(self) -> bool:
        """The project database changed after data.yaml was written."""
        exported = self.project_root / "data.yaml"
        db = Path(self.parent.state_manager.db.db_path)
        try:
            export_time = exported.stat().st_mtime
            changed = max(p.stat().st_mtime for p in (db, db.with_name(db.name + "-wal"))
                          if p.exists())
        except (OSError, ValueError):
            return False
        return changed > export_time + 5

    def validate_project_structure(self, root: Path):
        required = [
            "autosplit_train.txt",
            "autosplit_val.txt",
            "autosplit_test.txt",
            "data.yaml",
            "images",
            "labels"
        ]
        missing = []
        for item in required:
            path = root / item
            if item in ["images", "labels"]:
                if not path.is_dir():
                    missing.append(item)
            else:
                if not path.is_file():
                    missing.append(item)
        return missing


    def get_settings(self):
        # Step 1: Compute base output directory using pathlib (OS-independent)
        base_output_dir = self.project_root / "trainings" / self.project_name.text()
        run_base = self.run_name.text()
        full_output_dir = base_output_dir / run_base

        # Step 2: Ensure unique output directory (e.g., run, run1, run2, ...)
        counter = 1
        while full_output_dir.exists():
            full_output_dir = base_output_dir / f"{run_base}{counter}"
            counter += 1

        # Step 3: Update the output_dir label in UI
        self.output_dir_input.setText(str(full_output_dir))

        # Step 4: Return a clean TrainingSettings object
        return TrainingSettings(
            model=str(Path(self.models_dir) / self.model_selector.currentText()),
            data=str(Path(self.data_path_input.text())),
            epochs=self.epoch_box.value(),
            batch=int(self.batch_box.value()),
            imgsz=self.imgsz_box.value(),
            device=self.detected_device,
            optimizer=self.optimizer_input.currentText(),
            project=self.project_name.text(),
            name=full_output_dir.name,  # e.g., "run", "run1", etc.
            time=self.time_limit.value() or None,
            patience=self.patience_box.value(),
            lr0=self.lr0.value(),
            lrf=self.lrf.value(),
            momentum=self.momentum.value(),
            weight_decay=self.weight_decay.value(),
            hsv_h=self.hsv_h.value(),
            hsv_s=self.hsv_s.value(),
            hsv_v=self.hsv_v.value(),
            degrees=self.degrees.value(),
            translate=self.translate.value(),
            scale=self.scale.value(),
            shear=self.shear.value(),
            perspective=self.perspective.value(),
            flipud=self.flipud.value(),
            fliplr=self.fliplr.value(),
            bgr=self.bgr.value(),
            mosaic=self.mosaic.value(),
            mixup=self.mixup.value(),
            cutmix=self.cutmix.value(),
            copy_paste=self.copy_paste.value(),
            copy_paste_mode=self.copy_paste_mode.currentText(),
            auto_augment=self.auto_augment.currentText(),
            erasing=self.erasing.value(),
            output_dir=str(full_output_dir)
        )
