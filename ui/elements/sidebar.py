from PyQt6.QtWidgets import (
    QVBoxLayout, QPushButton, QHBoxLayout, QWidget, QComboBox, QColorDialog,
    QInputDialog, QMessageBox, QFrame, QLabel, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor
import qtawesome as qta
import re
from core.tools import sam_registry
from services.file_handlers import get_tooltip
from services.seproj import update_classes as seproj_update_classes
from services.logger import get_logger

logger = get_logger(__name__)

class Sidebar(QWidget):
    """
    Polished Sidebar: grouped sections, elegant buttons, subtle separators,
    and consistent spacing/typography.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # --- Root layout ----------------------------------------------------
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Card container for a modern look
        self.card = QFrame(self)
        self.card.setObjectName("SidebarCard")
        self.card.setFrameShape(QFrame.Shape.NoFrame)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(12)





        root.addWidget(self.card)

        # --- Sections --------------------------------------------------------
        self._add_section_title("Navigation", card_layout)
        self._init_navigation_buttons(card_layout)
        self._add_separator(card_layout)

        self._add_section_title("Tools", card_layout)
        self._init_tools(card_layout)
        self._add_separator(card_layout)

        self._add_section_title("Classes", card_layout)
        self._init_class_management(card_layout)

        # Push content to the top; keep nice breathing room at bottom
        card_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Apply styles last
        self._apply_styles()

    # ----------------------------------------------------------------------
    # Sections & helpers
    # ----------------------------------------------------------------------
    def _add_section_title(self, text: str, layout: QVBoxLayout):
        row = QHBoxLayout()
        row.setSpacing(6)

        dot = QLabel(self.card)
        dot.setObjectName("SectionDot")
        dot.setFixedSize(8, 8)
        row.addWidget(dot, alignment=Qt.AlignmentFlag.AlignVCenter)

        lbl = QLabel(text, self.card)
        lbl.setObjectName("SectionTitle")
        row.addWidget(lbl, 1)

        layout.addLayout(row)

    def _add_separator(self, layout: QVBoxLayout):
        sep = QFrame(self.card)
        sep.setObjectName("Separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

    # ----------------------------------------------------------------------
    # Navigation
    # ----------------------------------------------------------------------
    def _init_navigation_buttons(self, layout: QVBoxLayout):
        row = QHBoxLayout()
        row.setSpacing(8)

        prev_button = self._create_icon_button(
            fa_icon_name="fa5s.arrow-left",
            callback=self.parent.previous_image,
            tooltip="prev_image",
            label="Prev"
        )
        next_button = self._create_icon_button(
            fa_icon_name="fa5s.arrow-right",
            callback=self.parent.next_image,
            tooltip="next_image",
            label="Next"
        )

        prev_button.setProperty("variant", "secondary")
        next_button.setProperty("variant", "primary")

        row.addWidget(prev_button, 1)
        row.addWidget(next_button, 1)
        layout.addLayout(row)

    # ----------------------------------------------------------------------
    # Tools
    # ----------------------------------------------------------------------
    def _init_tools(self, layout: QVBoxLayout):
        # A tidy grid of equally-styled toggle buttons
        grid = QHBoxLayout()
        grid.setSpacing(8)

        # Manual mask
        self.manual_mask = self._create_toggle(
            icon="fa5s.pencil-alt",
            label="Manual",
            tooltip="manual_mask",
            on_toggle=self.toggle_manual_mask
        )

        # DEXTR
        self.dextr = self._create_toggle(
            icon="fa5s.magic",  # “smart” selection vibe
            label="DEXTR",
            tooltip="dextr",
            on_toggle=self.toggle_dextr
        )

        # SAM (model variant chosen in Settings -> SAM Model)
        self.sam = self._create_toggle(
            icon="fa5s.robot",
            label="SAM",
            tooltip="sam",
            on_toggle=self.toggle_sam
        )
        self.update_sam_tooltip()

        # Intelligent Scissors
        self.intelligent_scissors = self._create_toggle(
            icon="fa5s.cut",
            label="Scissors",
            tooltip="intelligent_scissors",
            on_toggle=self.toggle_intelligent_scissors
        )

        # Arrange as two rows for balance
        row1 = QHBoxLayout(); row1.setSpacing(8)
        row1.addWidget(self.manual_mask, 1)
        row1.addWidget(self.sam, 1)
        row1.addWidget(self.dextr, 1)

        row2 = QHBoxLayout(); row2.setSpacing(8)
        row2.addWidget(self.intelligent_scissors, 1)
        row2.addStretch(2)

        layout.addLayout(row1)
        layout.addLayout(row2)

    # ----------------------------------------------------------------------
    # Classes
    # ----------------------------------------------------------------------
    def _init_class_management(self, layout: QVBoxLayout):
        # Dropdown
        self.class_dropdown = QComboBox(self.card)
        self.class_dropdown.setObjectName("ClassDropdown")
        layout.addWidget(self.class_dropdown)

        # Add / Remove row
        row = QHBoxLayout()
        row.setSpacing(8)

        add_btn = self._pill_button("Add Class", "fa5s.plus-circle", variant="primary")
        add_btn.clicked.connect(self.add_class)

        remove_btn = self._pill_button("Remove Selected", "fa5s.trash-alt", variant="danger")
        remove_btn.clicked.connect(self.remove_selected_class)

        row.addWidget(add_btn, 1)
        row.addWidget(remove_btn, 1)

        layout.addLayout(row)

        # Populate at startup
        self.populate_class_dropdown()

    # ----------------------------------------------------------------------
    # Toggle handlers (keep your tool-manager logic, but make toggles exclusive visually)
    # ----------------------------------------------------------------------
    def _uncheck_others(self, keep):
        for btn in [self.manual_mask, self.dextr, self.sam, self.intelligent_scissors]:
            if btn is not keep:
                btn.setChecked(False)

    def toggle_dextr(self):
        if self.dextr.isChecked():
            self._uncheck_others(self.dextr)
            if not self.parent.tool_manager.enable_tool("dextr"):
                self.dextr.setChecked(False)  # model missing/declined or failed to load
        else:
            self.parent.tool_manager.disable_tools()

    def toggle_sam(self):
        if self.sam.isChecked():
            self._uncheck_others(self.sam)
            if not self.parent.tool_manager.enable_tool("sam"):
                self.sam.setChecked(False)
        else:
            self.parent.tool_manager.disable_tools()

    def update_sam_tooltip(self):
        """Show the currently selected model variant in the SAM tooltip."""
        variant = sam_registry.get_selected_variant()
        state = "" if sam_registry.is_available(variant) else ", not downloaded yet"
        self.sam.setToolTip(f"{get_tooltip('sam')}\nCurrent model: {variant.label}{state} "
                            "(change in Settings -> Models)")

    def toggle_intelligent_scissors(self):
        if self.intelligent_scissors.isChecked():
            self._uncheck_others(self.intelligent_scissors)
            self.parent.tool_manager.enable_tool("intelligent_scissors")
        else:
            self.parent.tool_manager.disable_tools()

    def toggle_manual_mask(self):
        if self.manual_mask.isChecked():
            self._uncheck_others(self.manual_mask)
            self.parent.tool_manager.enable_tool("manual_mask")
        else:
            self.parent.tool_manager.disable_tools()

    # ----------------------------------------------------------------------
    # Class ops (unchanged logic, just living here)
    # ----------------------------------------------------------------------
    def get_selected_class_color(self):
        current_index = self.class_dropdown.currentIndex()
        if current_index >= 0:
            text = self.class_dropdown.currentText()
            color = self.class_dropdown.itemData(current_index)
            return text.split(" ")[0], color
        return None, None

    def add_class(self):
        class_name, ok = QInputDialog.getText(self, "Add Class", "Enter class name:")
        if not ok or not class_name.strip():
            logger.debug("Add class cancelled: empty name")
            return

        class_name = class_name.strip()

        # Allow only letters, numbers, underscores
        if not re.fullmatch(r"[A-Za-z0-9_]+", class_name):
            QMessageBox.warning(
                self,
                "Invalid Class Name",
                "Class name must contain only letters, numbers, and underscores (no spaces or special characters)."
            )
            return

        # Prevent duplicate class names (case-insensitive)
        existing_classes = [c.lower() for c in self.parent.state_manager.class_manager.get_all_class_names()]
        if class_name.lower() in existing_classes:
            QMessageBox.warning(
                self,
                "Duplicate Class",
                f"The class '{class_name}' already exists."
            )
            return

        color = QColorDialog.getColor()
        if not color.isValid():
            logger.debug("Add class cancelled: no color selected")
            return

        self.parent.state_manager.class_manager.add_class(class_name, color)
        logger.info("Added class %r (%s)", class_name, color.name())
        self.populate_class_dropdown()
        self._sync_classes_to_seproj()


    def remove_selected_class(self):
        current_index = self.class_dropdown.currentIndex()
        if current_index < 0:
            logger.debug("Remove class requested with no class selected")
            return

        class_name = self.class_dropdown.currentText().split(" ")[0]
        count = self.parent.state_manager.mask_manager.count_masks_by_class(class_name)

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Confirm Deletion")
        msg_box.setText(f"Are you sure you want to delete the class '{class_name}'?")
        msg_box.setInformativeText(f"This will also delete {count} mask(s) associated with this class.")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)

        result = msg_box.exec()
        if result == QMessageBox.StandardButton.No:
            logger.debug("Class deletion cancelled by user")
            return

        self.parent.state_manager.class_manager.remove_class(class_name)
        self.parent.state_manager.mask_manager.delete_masks_by_class(class_name)
        self.parent.state_manager.class_manager.reindex_classes()

        self.class_dropdown.removeItem(current_index)
        logger.info("Removed class %r and its %d mask(s)", class_name, count)
        self._sync_classes_to_seproj()
        self.parent.image_display.display_image(self.parent.state_manager.current_image_path, preserve_zoom=True)

    def pick_class_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.selected_color = color
            logger.debug("Selected color %s", color.name())

    def _sync_classes_to_seproj(self):
        project_root = getattr(self.parent.state_manager, "project_root", None)
        if not project_root:
            return
        rows = self.parent.state_manager.class_manager.list_classes()
        classes = [{"name": row[1], "color": row[2]} for row in rows]
        seproj_update_classes(project_root, classes)

    def populate_class_dropdown(self):
        class_manager = self.parent.state_manager.class_manager
        class_names = class_manager.get_all_class_names()
        self.class_dropdown.clear()
        if not class_names:
            logger.debug("No classes defined yet; dropdown left empty")
            return
        self.class_dropdown.addItems(class_names)
        logger.debug("Loaded %d class(es) into the dropdown", len(class_names))

    def has_valid_class_selection(self):
        if self.class_dropdown.count() == 0:
            QMessageBox.warning(self, "No Classes", "⚠️ No classes defined yet. Please add a class first.")
            return False
        if self.class_dropdown.currentIndex() < 0:
            QMessageBox.warning(self, "No Class Selected", "⚠️ Please select a class before saving the mask.")
            return False
        return True

    # ----------------------------------------------------------------------
    # UI element factories
    # ----------------------------------------------------------------------
    def _create_icon_button(self, fa_icon_name, callback, tooltip=None, label=None):
        btn = QPushButton(label or "", self.card)
        if fa_icon_name:
            btn.setIcon(qta.icon(fa_icon_name))
        btn.setIconSize(QSize(16, 16))
        btn.setFixedHeight(36)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setProperty("variant", btn.property("variant") or "secondary")

        if tooltip:
            btn.setToolTip(get_tooltip(tooltip))
        if callback:
            btn.clicked.connect(callback)
        return btn

    def _create_toggle(self, icon: str, label: str, tooltip: str, on_toggle):
        btn = QPushButton(label, self.card)
        btn.setCheckable(True)
        btn.setIcon(qta.icon(icon))
        btn.setIconSize(QSize(16, 16))
        btn.setFixedHeight(36)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setProperty("variant", "toggle")

        if tooltip:
            btn.setToolTip(get_tooltip(tooltip))
        if on_toggle:
            btn.clicked.connect(on_toggle)
        return btn

    def _pill_button(self, text: str, icon: str, variant: str = "secondary"):
        btn = QPushButton(text, self.card)
        btn.setIcon(qta.icon(icon))
        btn.setIconSize(QSize(16, 16))
        btn.setFixedHeight(34)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setProperty("variant", variant)
        return btn

    # ----------------------------------------------------------------------
    # Styles
    # ----------------------------------------------------------------------
    def _apply_styles(self):
        self.setStyleSheet("""
        QWidget { font-size: 12.5px; }

        /* Buttons (existing) */
        QPushButton {
            border: 1px solid palette(Mid);
            border-radius: 10px;
            padding: 6px 10px;
            color: palette(ButtonText);
            background-color: palette(Button);
        }
        QPushButton:hover { background-color: palette(Midlight); }
        QPushButton:pressed { background-color: palette(Dark); }

        /* Variants (existing) */
        QPushButton[variant="primary"] {
            background-color: rgba(56,132,255,220);
            border: 1px solid rgba(56,132,255,220);
            color: white;
        }
        QPushButton[variant="primary"]:hover {
            background-color: rgba(56,132,255,240);
        }
        QPushButton[variant="danger"] {
            background-color: rgba(220,40,60,200);
            border: 1px solid rgba(220,40,60,220);
            color: white;
        }
        QPushButton[variant="danger"]:hover {
            background-color: rgba(220,40,60,220);
        }

        /* === Toggle tools (NEW): default + checked (blue) === */
        QPushButton[variant="toggle"] {
            /* default/unchecked look */
            background-color: palette(Button);
            color: palette(ButtonText);
            border: 1px solid palette(Mid);
        }
        QPushButton[variant="toggle"]:hover {
            background-color: palette(Midlight);
        }

        /* Active tool turns blue */
        QPushButton[variant="toggle"]:checked {
            background-color: rgba(56,132,255,220);
            border: 1px solid rgba(56,132,255,220);
            color: white;
        }
        QPushButton[variant="toggle"]:checked:hover {
            background-color: rgba(56,132,255,240);
        }
        """)
