# ui/custom_components/mask_context_menu.py
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QPalette
import qtawesome as qta


class MaskContextMenu(QMenu):
    """
    Context menu operating on a selection of masks.

    Both the annotations table and the image view create one of these and
    exec() it; all mask-selection actions live here, in one place. The menu
    also opens with an empty selection — selection-dependent actions are
    simply disabled.

    Args:
        main_window: The MainApp instance (hub to state_manager, image_display,
                     annotations dock).
        rows: The selection to operate on, as (image_name, mask_id) tuples.
        parent: Optional QWidget parent for the menu.

    Extending:
        Sections are built by dedicated _build_*_section methods, assembled in
        _build(). A subclass can override _build() to add, remove, or reorder
        sections without touching the existing ones.
    """

    def __init__(self, main_window, rows, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.state = main_window.state_manager
        self.rows = list(rows)
        self._apply_styles()
        self._build()

    # ------------------------------ building --------------------------------
    def _build(self):
        self._build_header()
        self.addSeparator()
        self._build_actions_section()
        self.addSeparator()
        self._build_selection_section()

    def _build_actions_section(self):
        """'Actions' submenu grouping the operations on the selected masks."""
        actions_menu = self.addMenu(qta.icon("fa5s.bolt"), "Actions")
        if not self.rows:
            actions_menu.setEnabled(False)
        self._build_class_section(actions_menu)
        self._build_split_section(actions_menu)
        self._build_brush_section(actions_menu)
        actions_menu.addSeparator()
        self._build_delete_section(actions_menu)

    def _build_header(self):
        n = len(self.rows)
        text = "No masks selected" if n == 0 else f"{n} mask{'s' if n > 1 else ''} selected"
        header = self.addAction(qta.icon("fa5s.mouse-pointer", color="#3884ff"), text)
        header.setEnabled(False)

    def _build_class_section(self, parent_menu):
        class_menu = parent_menu.addMenu(qta.icon("fa5s.tags"), "Set class")

        class_names = self.state.class_manager.get_all_class_names()
        if not self.rows or not class_names:
            class_menu.setEnabled(False)
        for cls in class_names:
            action = class_menu.addAction(self._class_icon(cls), cls)
            action.triggered.connect(lambda _, c=cls: self._assign_class(c))

    def _build_selection_section(self):
        select_all = self.addAction(qta.icon("fa5s.object-group"), "Select all masks on image")
        select_all.triggered.connect(self._select_all)
        select_all.setProperty("keep_open", True)
        if not self.main_window.image_display._mask_index:
            select_all.setEnabled(False)

        clear = self.addAction(qta.icon("fa5s.times-circle"), "Clear selection")
        clear.triggered.connect(self._clear_selection)
        clear.setProperty("keep_open", True)
        if not self.rows:
            clear.setEnabled(False)

    def _build_split_section(self, parent_menu):
        split = parent_menu.addAction(qta.icon("fa5s.cut"), "Split mask (draw a stroke)")
        split.setToolTip("Drag a freehand stroke across the mask to cut it")
        split.triggered.connect(self._start_split)
        if len(self.rows) != 1:  # only meaningful for a single mask
            split.setEnabled(False)

    def _build_brush_section(self, parent_menu):
        brush = parent_menu.addAction(qta.icon("fa5s.paint-brush"), "Brush edit (paint / erase)")
        brush.setToolTip("Left-drag paints, right-drag erases, scroll resizes, "
                         "Ctrl+scroll zooms, S applies, Esc cancels")
        brush.triggered.connect(self._start_brush)
        if len(self.rows) != 1:  # only meaningful for a single mask
            brush.setEnabled(False)

    def _build_delete_section(self, parent_menu):
        n = len(self.rows)
        label = "Delete selected" if n == 0 else f"Delete {n} mask{'s' if n > 1 else ''}"
        delete = parent_menu.addAction(qta.icon("fa5s.trash-alt", color="#dc283c"), label)
        delete.triggered.connect(self._delete_selection)
        if not self.rows:
            delete.setEnabled(False)

    def _class_icon(self, class_name: str) -> QIcon:
        """Rounded color swatch for a class."""
        color = self.state.class_manager.get_class_color(class_name)
        pm = QPixmap(14, 14)
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(color)
        painter.setPen(QColor(0, 0, 0, 60))
        painter.drawRoundedRect(QRectF(0.5, 0.5, 13, 13), 4, 4)
        painter.end()
        return QIcon(pm)

    # ------------------------------ keep-open behavior ----------------------
    def mouseReleaseEvent(self, event):
        """Actions marked keep_open trigger without closing the menu."""
        action = self.actionAt(event.position().toPoint())
        if action is not None and action.isEnabled() and action.property("keep_open"):
            action.trigger()
            self._refresh()
            return  # skip super() so the menu stays open
        super().mouseReleaseEvent(event)

    def _refresh(self):
        """Rebuild the menu in place after the selection changed."""
        display = self.main_window.image_display
        image_name = self.state.current_image_name
        self.rows = [(image_name, int(mid)) for mid in display.highlighted_mask_ids]
        self.clear()
        self._build()

    # ------------------------------ actions ---------------------------------
    def _assign_class(self, new_class: str):
        """Assign new_class to all rows in a single transaction."""
        # one transaction, one masks_updated: the image, table and
        # statistics then redraw themselves
        self.state.mask_manager.assign_class(self.rows, new_class)

    def _start_split(self):
        _, mask_id = self.rows[0]
        self.main_window.image_display.start_split_mode(mask_id)

    def _start_brush(self):
        _, mask_id = self.rows[0]
        self.main_window.image_display.start_brush_mode(mask_id)

    def _select_all(self):
        display = self.main_window.image_display
        mask_ids = list(display._mask_index.keys())
        if mask_ids:
            display._select_masks(mask_ids, skip_already_selected=True)

    def _clear_selection(self):
        self.main_window.image_display.clear_selection()

    def _delete_selection(self):
        by_image: dict[str, list[int]] = {}
        for image_name, mask_id in self.rows:
            by_image.setdefault(image_name, []).append(mask_id)

        for image_name, mask_ids in by_image.items():
            self.state.mask_manager.delete_masks(image_name, mask_ids, profile=False)

        self._clear_selection()   # the image redraws itself (masks_updated)

    # ------------------------------ styling ---------------------------------
    def _apply_styles(self):
        # palette(Mid) can vanish against a dark base, so derive the muted
        # color from the theme's text color instead (dimmed via alpha).
        text = self.palette().color(QPalette.ColorRole.Text)
        muted = f"rgba({text.red()},{text.green()},{text.blue()},140)"

        # Stylesheet is inherited by submenus created via addMenu()
        self.setStyleSheet(f"""
        QMenu {{
            background-color: palette(Base);
            color: palette(Text);
            border: 1px solid palette(Mid);
            border-radius: 10px;
            padding: 6px;
        }}
        QMenu::item {{
            padding: 7px 24px 7px 10px;
            border-radius: 6px;
        }}
        QMenu::item:selected {{
            background-color: palette(Highlight);
            color: palette(HighlightedText);
        }}
        QMenu::item:disabled {{
            color: {muted};
            background: transparent;
        }}
        QMenu::separator {{
            height: 1px;
            background-color: palette(Mid);
            margin: 5px 8px;
        }}
        QMenu::icon {{ padding-left: 6px; }}
        """)
