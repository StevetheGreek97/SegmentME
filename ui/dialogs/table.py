# ui/dialogs/table.py

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSizePolicy,
    QTableView, QComboBox, QAbstractItemView, QStyledItemDelegate
)
from ui.custom_components.mask_context_menu import MaskContextMenu
from PyQt6.QtCore import (Qt, pyqtSignal, QTimer, QModelIndex, QSortFilterProxyModel,
                          QItemSelection, QItemSelectionModel)
from PyQt6.QtGui import QIcon, QPixmap, QColor

import time
from contextlib import contextmanager

from services.logger import get_logger
from core.measure.cache import measure_cached
from core.measure.morphometrics import to_units

logger = get_logger(__name__)

# --- Back-compat shim for code expecting QTableWidget.selectedItems() ----
class _FakeItem:
    """Minimal stand-in for QTableWidgetItem with column() and text()."""
    def __init__(self, col: int, text: str):
        self._col = col
        self._text = text
    def column(self) -> int:
        return self._col
    def text(self) -> str:
        return self._text

class SelectionView(QTableView):
    """
    QTableView with a back-compat .selectedItems(), .rowCount(), and .item()
    so existing QTableWidget-style callers keep working.
    We only really need col=1 (Mask ID), but item() works for any column.
    """
    def __init__(self, dock, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dock = dock  # access to _proxy and _model

    # --- Back-compat: QTableWidget.selectedItems() ---
    def selectedItems(self):
        items = []
        sel = self.selectionModel().selectedRows()  # indexes in proxy
        for proxy_idx in sel:
            src_idx = self._dock._proxy.mapToSource(proxy_idx)
            row = self._dock._model.row_at(src_idx.row())
            items.append(_FakeItem(1, str(row.mask_id)))
        return items

    # --- Back-compat: QTableWidget.rowCount() ---
    def rowCount(self):
        m = self.model()
        return m.rowCount() if m is not None else 0

    # --- Back-compat: QTableWidget.item(row, col) ---
    def item(self, row: int, column: int):
        m = self.model()
        if m is None or row < 0 or row >= m.rowCount():
            return None
        idx = m.index(row, column)
        # fetch display data from proxy model (since view is using proxy)
        val = m.data(idx, Qt.ItemDataRole.DisplayRole)
        return _FakeItem(column, "" if val is None else str(val))

# ----------------------------- tiny timing helper -----------------------------
@contextmanager
def _t(section, bucket: dict):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        bucket[section] = bucket.get(section, 0.0) + (time.perf_counter() - t0)


# ----------------------------- Table Model -----------------------------------
SORT_ROLE = Qt.ItemDataRole.UserRole + 1
class MaskTableModelData:
    """
    Simple row container to keep the code readable.
    """
    __slots__ = ("image_name", "mask_id", "surface_area", "klass", "length", "width")

    def __init__(self, image_name: str, mask_id: int, surface_area: float | None, klass: str | None,
                 length: float | None = None, width: float | None = None):
        self.image_name = image_name
        self.mask_id = int(mask_id)
        # area, length and width are in the table's unit (mm / µm, or px without a scale)
        self.surface_area = 0.0 if surface_area is None else float(surface_area)
        self.klass = "" if klass is None else str(klass)
        self.length = length
        self.width = width


from PyQt6.QtCore import QAbstractTableModel


class MaskTableModel(QAbstractTableModel):
    """
    Lightweight model for annotations.
    Edits are allowed only for the 'Class' column.
    """
    classChanged = pyqtSignal(str, int, str)  # image_name, mask_id, new_class

    HEADERS = ["Image Name", "Mask ID", "Area", "Class", "Length", "Width"]

    COL_IMAGE = 0
    COL_ID = 1
    COL_AREA = 2
    COL_CLASS = 3
    COL_LENGTH = 4
    COL_WIDTH = 5
    NUMERIC = (COL_ID, COL_AREA, COL_LENGTH, COL_WIDTH)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[MaskTableModelData] = []
        self._unit = "px"     # "mm", "µm" or "px" (no scale for this image)

    def set_unit(self, unit: str):
        if unit != self._unit:
            self._unit = unit
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(self.HEADERS) - 1)

    def _number(self, value, area=False):
        if value is None:
            return ""
        if self._unit == "px":
            return f"{value:.0f}" if area else f"{value:.1f}"
        return f"{value:.4g}"

    # ---------------- required overrides -----------------
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            name = self.HEADERS[section]
            if section == self.COL_AREA:
                return f"{name} ({self._unit}²)"
            if section in (self.COL_LENGTH, self.COL_WIDTH):
                return f"{name} ({self._unit})"
            return name
        return section + 1

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        row = self._rows[r]

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if c == self.COL_IMAGE:
                return row.image_name
            elif c == self.COL_ID:
                return str(row.mask_id)
            elif c == self.COL_AREA:
                return self._number(row.surface_area, area=True)
            elif c == self.COL_CLASS:
                return row.klass
            elif c == self.COL_LENGTH:
                return self._number(row.length)
            elif c == self.COL_WIDTH:
                return self._number(row.width)

        # Sorting: numbers as numbers (the proxy sorts on SORT_ROLE)
        if role == SORT_ROLE:
            values = {self.COL_IMAGE: row.image_name, self.COL_ID: row.mask_id,
                      self.COL_AREA: row.surface_area, self.COL_CLASS: row.klass.lower(),
                      self.COL_LENGTH: row.length, self.COL_WIDTH: row.width}
            value = values.get(c)
            return -1.0 if value is None else value

        # Center numeric columns visually
        if role == Qt.ItemDataRole.TextAlignmentRole and c in self.NUMERIC:
            return int(Qt.AlignmentFlag.AlignCenter)

        return None

    # Built once: Qt asks for every cell when selecting (90,000 calls for "select
    # all" on 5,000 rows), and combining enum flags in Python each time added up.
    _FLAGS = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
    _FLAGS_EDITABLE = _FLAGS | Qt.ItemFlag.ItemIsEditable

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return self._FLAGS_EDITABLE if index.column() == self.COL_CLASS else self._FLAGS

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        if index.column() != self.COL_CLASS:
            return False

        r = index.row()
        row = self._rows[r]
        new_class = str(value)
        if row.klass == new_class:
            return True

        row.klass = new_class
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
        # Notify listeners (dock) to persist to DB & refresh overlays/plots.
        self.classChanged.emit(row.image_name, int(row.mask_id), new_class)
        return True

    # ---------------- convenience API -------------------
    def set_rows(self, rows: list[MaskTableModelData]):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def row_at(self, row_idx: int) -> MaskTableModelData:
        return self._rows[row_idx]


# ----------------------------- Filter/Sort Proxy ------------------------------
class MaskFilterProxy(QSortFilterProxyModel):
    """
    Text search across all columns + class filter.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self._class = None  # None => all classes, else exact match
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def setSearchText(self, text: str):
        t = (text or "").strip()
        if t == self._text:
            return
        self._text = t
        self.invalidateFilter()

    def setClassFilter(self, klass: str | None):
        v = None if (not klass or klass == "All classes") else klass
        if v == self._class:
            return
        self._class = v
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model: MaskTableModel = self.sourceModel()  # type: ignore
        idx_img = model.index(source_row, MaskTableModel.COL_IMAGE, source_parent)
        idx_id = model.index(source_row, MaskTableModel.COL_ID, source_parent)
        idx_area = model.index(source_row, MaskTableModel.COL_AREA, source_parent)
        idx_class = model.index(source_row, MaskTableModel.COL_CLASS, source_parent)

        # Class filter
        if self._class is not None:
            klass = model.data(idx_class, Qt.ItemDataRole.DisplayRole) or ""
            if klass != self._class:
                return False

        # Text search across all textual columns
        if not self._text:
            return True

        txt = self._text.lower()
        vals = [
            (model.data(idx_img) or "").lower(),
            (model.data(idx_id) or "").lower(),
            (model.data(idx_area) or "").lower(),
            (model.data(idx_class) or "").lower(),
        ]
        return any(txt in v for v in vals)


# ----------------------------- delegate for Class col -------------------------
class ClassComboDelegate(QStyledItemDelegate):
    """
    Provides a QComboBox editor only while editing the 'Class' column.
    No per-row widget overhead.
    """
    def __init__(self, class_manager, get_icon_cache, parent=None):
        super().__init__(parent)
        self.cm = class_manager
        self._get_icon_cache = get_icon_cache  # callable — always returns the dock's current cache

    def createEditor(self, parent, option, index: QModelIndex):
        # Only column 3 gets a combo editor
        if index.column() != MaskTableModel.COL_CLASS:
            return super().createEditor(parent, option, index)

        combo = QComboBox(parent)
        icon_cache = self._get_icon_cache()
        for cls in self.cm.get_all_class_names():
            combo.addItem(icon_cache.get(cls), cls)
        QTimer.singleShot(0, combo.showPopup)
        return combo

    def setEditorData(self, editor, index):
        if isinstance(editor, QComboBox):
            current = index.data(Qt.ItemDataRole.EditRole) or index.data()
            i = editor.findText(current)
            editor.setCurrentIndex(max(i, 0))
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
        else:
            super().setModelData(editor, model, index)


# --------------------------------- main dock ----------------------------------
class MaskResultsDock(QDockWidget):
    masks_selected = pyqtSignal(list)  # list of dicts with image_name, mask_id, class

    def __init__(self, parent):
        super().__init__("Annotations", parent)
        self.parent = parent
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Throttle/Coalesce refresh
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._populate_table_impl)

        # --- Root container
        self.card = QWidget()
        self.card_layout = QVBoxLayout(self.card)
        self.card.setObjectName("DockCard")
        self.card_layout.setContentsMargins(10, 10, 10, 10)
        self.card_layout.setSpacing(8)
        self.setWidget(self.card)

        # --- Toolbar (filter + class filter)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.search = QLineEdit(placeholderText="Search…")
        self.search.textChanged.connect(self.apply_filters)
        self.search.setClearButtonEnabled(True)
        self.search.setMinimumWidth(160)

        self.class_filter = QComboBox()
        self.class_filter.addItem("All classes")
        self.class_filter.currentIndexChanged.connect(self.apply_filters)

        toolbar.addWidget(self.search, 2)
        toolbar.addWidget(self.class_filter, 1)
        self.card_layout.addLayout(toolbar)

        # --- Table (Model/View)
        self.table = SelectionView(self, self.card) 
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setWordWrap(False)                           # NEW
        self.table.verticalHeader().setDefaultSectionSize(22)  
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)

        # Model + Proxy
        self._model = MaskTableModel(self.table)
        self._proxy = MaskFilterProxy(self.table)
        self._proxy.setSourceModel(self._model)
        self._proxy.setSortRole(SORT_ROLE)
        self.table.setModel(self._proxy)

        # Delegate for column 3 (Class) — uses a lambda so it always reads the current cache
        self._class_icon_cache = self._build_class_icon_cache()
        self.table.setItemDelegateForColumn(
            MaskTableModel.COL_CLASS,
            ClassComboDelegate(
                class_manager=self.parent.state_manager.class_manager,
                get_icon_cache=lambda: self._class_icon_cache,
                parent=self.table
            )
        )

        # Persist edits coming from the model
        self._model.classChanged.connect(self._commit_class_change)

        # Selection signal
        self.table.selectionModel().selectionChanged.connect(self.on_selection_changed)

        # Context menu for bulk class assignment
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

        self.card_layout.addWidget(self.table)

        # --- Status line
        self.status = QLabel("0 rows")
        self.status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.card_layout.addWidget(self.status)

        # Signals (use throttled refresh)
        self.parent.state_manager.image_changed.connect(self.refresh_table)
        self.parent.state_manager.masks_updated.connect(self.refresh_table)

        # Dock settings
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self._apply_styles()

    # ------------------------------ Helpers / setup ---------------------------
    def _build_class_icon_cache(self):
        cache = {}
        cm = self.parent.state_manager.class_manager
        for cls in cm.get_all_class_names():
            qcol = cm.get_class_color(cls)
            pm = QPixmap(12, 12)
            pm.fill(qcol if isinstance(qcol, QColor) else QColor(qcol))
            cache[cls] = QIcon(pm)
        return cache

    def _commit_class_change(self, image_name: str, mask_id: int, new_class: str):
        # The image and the statistics redraw themselves (masks_updated)
        self.parent.state_manager.mask_manager.rename_mask(image_name, mask_id, new_class)

    def select_mask_ids(self, mask_ids, skip_already_selected=False):
        """Select the rows of these masks (clicked on the image).

        Matching rows are TOGGLED, as selectRow() does in MultiSelection mode,
        so rows that are already selected flip off. Click-selection relies on
        that toggle to deselect; bulk selection (e.g. "Select all") passes
        skip_already_selected=True to leave selected rows untouched. All rows
        change in one selection update, so the outlines refresh once (the
        selection change reports the whole selection: masks_selected)."""
        table = self.table
        table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        model = table.model()
        sel_model = table.selectionModel()
        wanted = {str(mask_id) for mask_id in mask_ids}
        last_column = max(model.columnCount() - 1, 0)

        toggle = QItemSelection()
        last_toggled_row = None
        for row in range(table.rowCount()):
            if table.item(row, 1).text() in wanted:
                already = sel_model.isSelected(model.index(row, 0))
                if not (skip_already_selected and already):
                    toggle.select(model.index(row, 0), model.index(row, last_column))
                    last_toggled_row = row

        if last_toggled_row is not None:
            sel_model.select(toggle, QItemSelectionModel.SelectionFlag.Toggle
                             | QItemSelectionModel.SelectionFlag.Rows)
            # Like selectRow(): the last toggled row becomes current, which
            # scrolls the table to it.
            sel_model.setCurrentIndex(model.index(last_toggled_row, 0),
                                      QItemSelectionModel.SelectionFlag.NoUpdate)

    # ------------------------------ Context menu / bulk class assign ---------
    def _selected_rows_data(self) -> list[tuple[str, int]]:
        """Selected rows as (image_name, mask_id) tuples."""
        rows = []
        for proxy_idx in self.table.selectionModel().selectedRows():
            src_idx = self._proxy.mapToSource(proxy_idx)
            row = self._model.row_at(src_idx.row())
            rows.append((row.image_name, row.mask_id))
        return rows

    def _show_context_menu(self, pos):
        rows = self._selected_rows_data()
        menu = MaskContextMenu(self.parent, rows, parent=self.table)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    # ------------------------------ Public refresh API (throttled) -----------
    def refresh_table(self, *_, delay_ms: int = 50):
        self._refresh_timer.start(delay_ms)

    # ------------------------------ Populate / Refresh (impl) ----------------
    def _populate_table_impl(self):
        stats = {}
        with _t("total", stats):

            selected_ids = self.get_current_selected_mask_ids()

            with _t("class_filter_refresh", stats):
                self._refresh_class_filter()

            self.table.setUpdatesEnabled(False)
            try:
                image_name = self.parent.state_manager.current_image_name
                if not image_name:
                    self._model.set_rows([])
                    self.update_status(0)
                    return

                with _t("load_masks", stats):
                    masks = self.parent.state_manager.mask_manager.load_masks(image_name)
                    # expected tuples: (mask_id, _mask, class_name, surface_area)

                with _t("build_rows", stats):
                    um, unit, unit_um = self._image_scale()
                    rows = []
                    for (mask_id, mask, class_name, surface_area) in masks:
                        px = measure_cached(mask)
                        if px is None:   # no area: keep the stored value
                            area, length, width = surface_area, None, None
                        elif um:
                            real = to_units(px, um, unit_um)
                            area, length, width = real["area"], real["length"], real["width"]
                        else:
                            area, length, width = px["area"], px["length"], px["width"]
                        rows.append(MaskTableModelData(image_name, mask_id, area, class_name,
                                                       length, width))
                    self._model.set_unit(unit if um else "px")
                    self._model.set_rows(rows)

                # Keep sort if user clicked a header previously
                # (QTableView + QSortFilterProxyModel handles this efficiently)

            finally:
                self.table.setUpdatesEnabled(True)

            with _t("restore_selection", stats):
                self.restore_selected_mask_ids(selected_ids)

            with _t("apply_filters", stats):
                self.apply_filters()

        ms = lambda k: stats.get(k, 0.0) * 1000.0
        logger.debug("Table refresh: total=%.2f ms (load=%.2f, build=%.2f, filters=%.2f)",
                     ms('total'), ms('load_masks'), ms('build_rows'), ms('apply_filters'))

    def _image_scale(self):
        """(µm per pixel or None, unit, µm per unit) for the image on screen."""
        try:
            return self.parent.current_image_scale()
        except Exception:
            return None, "px", 1.0

    # ------------------------------ Selection & filters -----------------------
    def _refresh_class_filter(self):
        self._class_icon_cache = self._build_class_icon_cache()
        current = self.class_filter.currentText() if self.class_filter.count() else "All classes"
        self.class_filter.blockSignals(True)
        self.class_filter.clear()
        self.class_filter.addItem("All classes")
        for name in self.parent.state_manager.class_manager.get_all_class_names():
            self.class_filter.addItem(name)
        idx = self.class_filter.findText(current)
        self.class_filter.setCurrentIndex(idx if idx >= 0 else 0)
        self.class_filter.blockSignals(False)

    def get_current_selected_mask_ids(self) -> set[str]:
        ids: set[str] = set()
        sel = self.table.selectionModel().selectedRows()  # indexes in proxy
        for proxy_idx in sel:
            src_idx = self._proxy.mapToSource(proxy_idx)
            row = self._model.row_at(src_idx.row())
            ids.add(str(row.mask_id))
        return ids

    def restore_selected_mask_ids(self, selected_ids: set[str]):
        if not selected_ids:
            return
        self.table.selectionModel().clearSelection()
        # iterate all source rows and select those whose mask_id is in the set
        for src_row in range(self._model.rowCount()):
            row = self._model.row_at(src_row)
            if str(row.mask_id) in selected_ids:
                src_idx = self._model.index(src_row, 0)
                proxy_idx = self._proxy.mapFromSource(src_idx)
                if proxy_idx.isValid():
                    self.table.selectRow(proxy_idx.row())

    def on_selection_changed(self, *_):
        sel = self.table.selectionModel().selectedRows()
        selected_rows = []
        for proxy_idx in sel:
            src_idx = self._proxy.mapToSource(proxy_idx)
            row = self._model.row_at(src_idx.row())
            selected_rows.append({
                "image_name": row.image_name,
                "mask_id": str(row.mask_id),
                "class": row.klass,
            })
        self.masks_selected.emit(selected_rows)

    def apply_filters(self):
        t0 = time.perf_counter()

        text = self.search.text()
        klass = self.class_filter.currentText()

        self._proxy.setSearchText(text)
        self._proxy.setClassFilter(klass)

        self.update_status()
        logger.debug("apply_filters: %.2f ms (rows=%d, visible=%d)",
                     (time.perf_counter() - t0) * 1000,
                     self._model.rowCount(), self._proxy.rowCount())

    def update_status(self):
        total = self._model.rowCount()
        visible = self._proxy.rowCount()
        self.status.setText(f"{visible} / {total} rows")

    # ------------------------------ Styles -----------------------------------
    def _apply_styles(self):
        self.setStyleSheet("""
        QWidget#DockCard {
            background-color: palette(Base);
            border: 1px solid palette(Mid);
            border-radius: 12px;
        }
        QMenu {
            background-color: palette(Base);
            color: palette(Text);
            border: 1px solid palette(Mid);
            padding: 4px 0;
        }
        QMenu::item { padding: 6px 14px; }
        QMenu::item:selected {
            background-color: palette(Highlight);
            color: palette(HighlightedText);
        }
        QMenu::separator {
            height: 1px; margin: 4px 8px;
            background-color: palette(Mid);
        }
        QToolButton {
            border: 1px solid palette(Mid);
            border-radius: 8px;
            padding: 6px 8px;
            background-color: palette(Button);
            color: palette(ButtonText);
        }
        QToolButton:hover { background-color: palette(Midlight); }
        QComboBox {
            border: 1px solid palette(Mid);
            border-radius: 8px;
            padding: 6px 10px;
            background-color: palette(Base);
            color: palette(Text);
        }
        QComboBox:hover { background-color: palette(AlternateBase); }
        QComboBox QAbstractItemView, QComboBox QAbstractItemView::viewport {
            background-color: palette(Base);
            color: palette(Text);
            border: 1px solid palette(Mid);
            outline: none;
        }
        QComboBox QAbstractItemView::item { padding: 6px 10px; }
        QComboBox QAbstractItemView::item:selected {
            background-color: palette(Highlight);
            color: palette(HighlightedText);
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 24px;
            border-left: 1px solid palette(Mid);
            background-color: palette(Midlight);
        }
        QLineEdit {
            border: 1px solid palette(Mid);
            border-radius: 10px;
            padding: 6px 10px;
            background-color: palette(Base);
            color: palette(Text);
        }
        QLineEdit:focus {
            border: 1px solid palette(Highlight);
            background-color: palette(AlternateBase);
        }
        QTableView, QAbstractItemView {
            background-color: palette(Base);
            color: palette(Text);
            selection-background-color: palette(Highlight);
            selection-color: palette(HighlightedText);
            alternate-background-color: palette(AlternateBase);
            gridline-color: palette(Mid);
        }
        QHeaderView::section {
            background-color: palette(Button);
            color: palette(ButtonText);
            border: 0;
            padding: 4px 6px;
        }
        QWidget { font-size: 12.5px; }
        """)
