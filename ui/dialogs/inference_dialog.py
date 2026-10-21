import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QDoubleSpinBox,
    QFileDialog, QComboBox, QSpinBox, QDialogButtonBox,
    QHBoxLayout, QCheckBox, QMessageBox
)

from core.tools import sam_registry

_BROWSE_TEXT = "\U0001F4C1 Browse for custom model..."

# Sentinel itemData values for the combos' non-model entries.
_SAM_BROWSE_DATA = "__browse_sam__"
_SAM_CUSTOM_DATA = "__custom_sam__"
_YOLO_BROWSE_DATA = "__browse_yolo__"
_YOLO_CUSTOM_DATA = "__custom_yolo__"


class InferenceDialog(QDialog):
    def __init__(self, model_paths, parent=None):
        """`model_paths`: the YOLO .pt files to offer (see
        services.model_store.installed_yolo_models / trained_models)."""
        super().__init__(parent)
        self.setWindowTitle("Run Inference")
        self.setMinimumWidth(420)
        self.model_paths = [Path(p) for p in model_paths]
        self.custom_model_path = None

        root = QVBoxLayout(self)
        root.setSpacing(14)

        # --- Model group -----------------------------------------------------
        model_box = QGroupBox("Model")
        model_form = QFormLayout()
        model_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.model_type_selector = QComboBox()
        self.model_type_selector.addItem("YOLO", "yolo")
        self.model_type_selector.addItem("HiReS (tiled YOLO for large images)", "hires")
        self.model_type_selector.addItem("SAM (auto-segment every object)", "sam")
        self.model_type_selector.setToolTip(
            "YOLO runs your trained detection/segmentation model.\n"
            "HiReS runs the same YOLO model on overlapping tiles of each image\n"
            "and merges the results, for high-resolution scans.\n"
            "SAM auto-segments every object in each image (SAM2/2.1 only)."
        )
        self.model_type_selector.currentIndexChanged.connect(self._on_model_type_changed)
        model_form.addRow("Model type:", self.model_type_selector)

        # -- YOLO model picker --
        self.model_selector = QComboBox()
        self.model_selector.setToolTip(
            "YOLO weights from this project's training runs and from the\n"
            "models/yolo folder (Settings -> Models -> Open folder)."
        )
        self.model_selector.activated.connect(self._handle_yolo_selection)
        self.populate_models()

        browse_hint = QLabel("Or choose a custom .pt file from disk.")
        browse_hint.setStyleSheet("color: #666; font-size: 11px;")
        self.yolo_row = QVBoxLayout()
        self.yolo_row.setSpacing(4)
        self.yolo_row.addWidget(self.model_selector)
        self.yolo_row.addWidget(browse_hint)
        model_form.addRow("Select model:", self.yolo_row)
        self._yolo_row_index = model_form.rowCount() - 1

        # -- SAM2/2.1 variant picker (auto-segment) --
        self.custom_sam_path = None
        self.sam_variant_selector = QComboBox()
        self.sam_variant_selector.setToolTip(
            "Downloaded SAM2/2.1 variants, or browse for your own .pt model:\n"
            "SAM2/2.1 (official or fine-tuned, size auto-detected) and\n"
            "Cellpose-SAM (cpsam; needs 'pip install cellpose'). SAM3 has no\n"
            "automatic \"segment everything\" mode yet; use the sidebar SAM\n"
            "tool for SAM3."
        )
        self.sam_variant_selector.activated.connect(self._handle_sam_selection)
        self.populate_sam_variants()
        model_form.addRow("SAM model:", self.sam_variant_selector)
        self._sam_row_index = model_form.rowCount() - 1

        self._model_form = model_form
        model_box.setLayout(model_form)
        root.addWidget(model_box)

        # --- Parameters group ------------------------------------------------
        params_box = QGroupBox("Parameters")
        params_form = QFormLayout()
        params_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Confidence
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.0, 1.0)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setDecimals(2)
        self.conf_spinbox.setValue(0.25)
        self.conf_spinbox.setToolTip("Detection confidence threshold (0–1).")
        params_form.addRow("Confidence:", self.conf_spinbox)

        # Image dimensions
        dims_row = QHBoxLayout()
        dims_row.setSpacing(8)

        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(16, 32768)
        self.width_spinbox.setValue(1024)
        self.width_spinbox.setSuffix(" px")
        self.width_spinbox.setToolTip("Input image width in pixels.")

        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(16, 32768)
        self.height_spinbox.setValue(1024)
        self.height_spinbox.setSuffix(" px")
        self.height_spinbox.setToolTip("Input image height in pixels.")

        self.lock_aspect = QCheckBox("Lock aspect ratio")
        self.lock_aspect.setChecked(False)
        self.lock_aspect.setToolTip("Keep height proportional when width changes.")

        # when locked, adjust height on width changes
        self._aspect_ratio = self.height_spinbox.value() / max(1, self.width_spinbox.value())

        def update_ratio():
            # store ratio when user edits both fields while unlocked
            w = max(1, self.width_spinbox.value())
            h = self.height_spinbox.value()
            self._aspect_ratio = h / w

        def on_width_changed():
            if self.lock_aspect.isChecked():
                w = self.width_spinbox.value()
                self.height_spinbox.blockSignals(True)
                self.height_spinbox.setValue(int(round(w * self._aspect_ratio)))
                self.height_spinbox.blockSignals(False)

        self.width_spinbox.valueChanged.connect(on_width_changed)
        self.width_spinbox.editingFinished.connect(update_ratio)
        self.height_spinbox.editingFinished.connect(update_ratio)

        dims_row.addWidget(self.width_spinbox, 1)
        dims_row.addWidget(self.height_spinbox, 1)

        dims_col = QVBoxLayout()
        dims_col.addLayout(dims_row)
        dims_col.addWidget(self.lock_aspect)

        params_form.addRow("Image size:", dims_col)
        self._params_form = params_form
        self._dims_row_index = params_form.rowCount() - 1

        params_box.setLayout(params_form)
        root.addWidget(params_box)

        # --- HiReS group (only for the HiReS model type) ---------------------
        self.hires_box = self._build_hires_box()
        root.addWidget(self.hires_box)

        # --- Buttons ---------------------------------------------------------
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Run")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self._on_model_type_changed(self.model_type_selector.currentIndex())

        # Focus the model selector first for smooth UX
        self.model_selector.setFocus()

    # ------------------------ HiReS -----------------------------------------
    def _build_hires_box(self):
        """Tiling / merging parameters, defaults as in the hires CLI."""
        box = QGroupBox("HiReS tiling")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def spin(lo, hi, value, suffix="", tip=""):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(value)
            s.setSuffix(suffix)
            s.setToolTip(tip)
            return s

        def dspin(lo, hi, value, step, decimals, tip):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(decimals)
            s.setSingleStep(step)
            s.setValue(value)
            s.setToolTip(tip)
            return s

        self.hires_tile_w = spin(64, 16384, 1024, " px", "Tile width in pixels.")
        self.hires_tile_h = spin(64, 16384, 1024, " px",
                                 "Tile height in pixels. Images must be taller than one tile.")
        tile_row = QHBoxLayout()
        tile_row.setSpacing(8)
        tile_row.addWidget(self.hires_tile_w, 1)
        tile_row.addWidget(self.hires_tile_h, 1)
        form.addRow("Tile size:", tile_row)

        self.hires_overlap = spin(0, 8192, 150, " px",
                                  "Overlap between neighbouring tiles. Should be larger than\n"
                                  "the biggest object, so every object is whole in some tile.")
        form.addRow("Overlap:", self.hires_overlap)

        self.hires_imgsz = spin(32, 8192, 1024, " px",
                                "Size YOLO resizes each tile to (ultralytics 'imgsz').")
        form.addRow("Inference size:", self.hires_imgsz)

        self.hires_edge = dspin(0.0, 0.2, 0.01, 0.005, 3,
                                "Drop polygons within this fraction of a tile's border\n"
                                "(cut-off objects), and of the image border after merging.")
        form.addRow("Edge filter:", self.hires_edge)

        self.hires_iou = dspin(0.05, 1.0, 0.7, 0.05, 2,
                               "Polygons from overlapping tiles with IoU above this\n"
                               "are duplicates; the higher-confidence one is kept.")
        form.addRow("NMS IoU:", self.hires_iou)

        hint = QLabel("Uses the YOLO model selected above. Masks are added to the "
                      "existing ones; nothing is deleted.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        form.addRow(hint)

        box.setLayout(form)
        return box

    def get_hires_params(self):
        """HiReS parameters as sent to the inference worker."""
        return {
            "chunk_size": [int(self.hires_tile_w.value()), int(self.hires_tile_h.value())],
            "overlap": int(self.hires_overlap.value()),
            "imgsz": int(self.hires_imgsz.value()),
            "edge_threshold": float(self.hires_edge.value()),
            "iou_thresh": float(self.hires_iou.value()),
        }

    def accept(self):
        if self.get_mode() == "hires":
            smallest = min(self.hires_tile_w.value(), self.hires_tile_h.value())
            if self.hires_overlap.value() >= smallest:
                QMessageBox.warning(
                    self, "Run Inference",
                    f"The overlap ({self.hires_overlap.value()} px) must be smaller "
                    f"than the tile size ({self.hires_tile_w.value()} x "
                    f"{self.hires_tile_h.value()} px).")
                return
        super().accept()

    # ------------------------ helpers ---------------------------------------
    def populate_models(self):
        self.model_selector.clear()
        for path in self.model_paths:
            label = path.name
            if path.parent.name == "weights":
                # <project>/trainings/<training project>/<run>/weights/best.pt
                label = f"{path.name}  ({path.parents[2].name} / {path.parents[1].name})"
            self.model_selector.addItem(label, str(path))
        self.model_selector.addItem(_BROWSE_TEXT, _YOLO_BROWSE_DATA)

    def populate_sam_variants(self):
        self.sam_variant_selector.clear()
        current_key = sam_registry.get_selected_key()
        default_index = 0
        for variant in sam_registry.SAM_VARIANTS.values():
            if variant.family != "sam2":
                continue  # SAM3 has no automatic "segment everything" mode yet
            if not sam_registry.is_available(variant):
                continue
            self.sam_variant_selector.addItem(variant.label, variant.key)
            if variant.key == current_key:
                default_index = self.sam_variant_selector.count() - 1

        self.sam_variant_selector.addItem(_BROWSE_TEXT, _SAM_BROWSE_DATA)
        self.sam_variant_selector.setCurrentIndex(default_index)

    def _handle_sam_selection(self, index):
        if self.sam_variant_selector.itemData(index) != _SAM_BROWSE_DATA:
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom SAM2 Checkpoint", "", "PyTorch Model (*.pt)"
        )
        if file_path:
            self.custom_sam_path = file_path
            base = os.path.basename(file_path)
            custom_index = self.sam_variant_selector.findData(_SAM_CUSTOM_DATA)
            if custom_index == -1:
                self.sam_variant_selector.insertItem(0, base, _SAM_CUSTOM_DATA)
                custom_index = 0
            else:
                self.sam_variant_selector.setItemText(custom_index, base)
            self.sam_variant_selector.setCurrentIndex(custom_index)
        else:
            # Cancelled: fall back to the first entry instead of leaving
            # the "Browse..." row selected.
            self.sam_variant_selector.setCurrentIndex(0)

    def _on_model_type_changed(self, _index):
        mode = self.get_mode()
        is_sam = mode == "sam"
        is_hires = mode == "hires"

        self.model_selector.setVisible(not is_sam)
        for i in range(self.yolo_row.count()):
            self.yolo_row.itemAt(i).widget().setVisible(not is_sam)
        self.sam_variant_selector.setVisible(is_sam)
        self._model_form.setRowVisible(self._yolo_row_index, not is_sam)
        self._model_form.setRowVisible(self._sam_row_index, is_sam)

        # SAM auto-segmentation has no confidence knob.
        self.conf_spinbox.setEnabled(not is_sam)
        # HiReS has its own per-tile inference size instead of width x height.
        self._params_form.setRowVisible(self._dims_row_index, not is_hires)
        self.hires_box.setVisible(is_hires)
        self.adjustSize()

    def _handle_yolo_selection(self, index):
        if self.model_selector.itemData(index) != _YOLO_BROWSE_DATA:
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom Model", "", "PyTorch Model (*.pt)"
        )
        if file_path:
            self.custom_model_path = file_path
            base = os.path.basename(file_path)
            custom_index = self.model_selector.findData(_YOLO_CUSTOM_DATA)
            if custom_index == -1:
                self.model_selector.insertItem(0, base, _YOLO_CUSTOM_DATA)
                custom_index = 0
            else:
                self.model_selector.setItemText(custom_index, base)
            self.model_selector.setCurrentIndex(custom_index)
        else:
            # Cancelled: don't leave the "Browse..." row selected.
            self.model_selector.setCurrentIndex(0)

    # ------------------------ getters ---------------------------------------
    def get_mode(self):
        return self.model_type_selector.currentData()

    def get_selected_model(self):
        """Absolute path of the chosen YOLO weights, or None if nothing usable
        is selected (no models installed and none browsed for)."""
        data = self.model_selector.currentData()
        if data == _YOLO_CUSTOM_DATA:
            return self.custom_model_path
        if data in (None, _YOLO_BROWSE_DATA):
            return None
        return data

    def get_sam_variant_key(self):
        data = self.sam_variant_selector.currentData()
        if data in (_SAM_BROWSE_DATA, _SAM_CUSTOM_DATA):
            return None
        return data

    def get_sam_custom_path(self):
        """Path of a user-browsed SAM2 checkpoint, or None if a registry
        variant is selected."""
        if self.sam_variant_selector.currentData() == _SAM_CUSTOM_DATA:
            return self.custom_sam_path
        return None

    def get_threshold(self):
        return float(self.conf_spinbox.value())

    def get_image_dimensions(self):
        """Returns (width, height) as ints."""
        return int(self.width_spinbox.value()), int(self.height_spinbox.value())
