from collections import defaultdict

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QDockWidget, QVBoxLayout, QWidget, QTabWidget
from PyQt6.QtCore import Qt

from core.measure.cache import measure_cached
from core.measure.morphometrics import to_units
from services.logger import get_logger

logger = get_logger(__name__)


class MaskStatisticsDock(QDockWidget):
    def __init__(self, parent):
        super().__init__("Annotation Statistics", parent)
        self.parent = parent
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)
        self.setWidget(self.widget)

        # Tabbed layout
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Tab 1: All annotations
        self.all_plot_widget = pg.PlotWidget()
        self.all_plot_widget.setBackground('w')
        self.all_plot_widget.showGrid(x=True, y=True)
        self.all_plot_widget.setLabel('left', 'Instance Count')
        self.all_plot_widget.setLabel('bottom', 'Class Name')
        self.tabs.addTab(self.all_plot_widget, "All Images")

        # Tab 2: Current image annotations
        self.current_plot_widget = pg.PlotWidget()
        self.current_plot_widget.setBackground('w')
        self.current_plot_widget.showGrid(x=True, y=True)
        self.current_plot_widget.setLabel('left', 'Instance Count')
        self.current_plot_widget.setLabel('bottom', 'Class Name')
        self.tabs.addTab(self.current_plot_widget, "Current Image")

        # Tab 3: size distribution (length) per class, current image
        self.sizes_plot_widget = pg.PlotWidget()
        self.sizes_plot_widget.setBackground('w')
        self.sizes_plot_widget.showGrid(x=True, y=True)
        self.sizes_plot_widget.setLabel('left', 'Count')
        self.sizes_plot_widget.addLegend(offset=(-10, 10))
        self.tabs.addTab(self.sizes_plot_widget, "Sizes")
        self.tabs.currentChanged.connect(lambda _: self._refresh_sizes())

        self.refresh_plot()

    def refresh_plot(self):
        """
        Refresh both the 'All Images' and 'Current Image' tabs.
        """
        masks = self.parent.state_manager.mask_manager
        self._refresh_tab(self.all_plot_widget, masks.counts_by_class)

        current_image = self.parent.state_manager.current_image_name
        if current_image:
            self._refresh_tab(self.current_plot_widget,
                              lambda: masks.counts_by_class(current_image))
        else:
            self.current_plot_widget.clear()
        self._refresh_sizes()

    def _refresh_sizes(self):
        """Histogram of lengths per class on the current image (in mm etc. when
        the image has a scale). Only computed while its tab is shown."""
        if self.tabs.currentWidget() is not self.sizes_plot_widget:
            return
        plot = self.sizes_plot_widget
        plot.clear()
        state = self.parent.state_manager
        name = state.current_image_name
        if not name:
            return
        try:
            um, unit, unit_um = self.parent.current_image_scale()
        except Exception:
            um, unit, unit_um = None, "px", 1.0
        lengths = defaultdict(list)
        try:
            for _, mask, class_name, _ in state.mask_manager.load_masks(name):
                px = measure_cached(mask)
                if px is not None:
                    lengths[class_name].append(
                        to_units(px, um, unit_um)["length"] if um else px["length"])
        except Exception:
            logger.exception("Failed to compute mask sizes")
            return
        plot.setLabel('bottom', f"Length ({unit if um else 'px'})")
        values = [v for vs in lengths.values() for v in vs]
        if not values:
            return
        edges = np.histogram_bin_edges(values, bins=min(30, max(5, len(values) // 4)))
        for class_name, vs in sorted(lengths.items()):
            counts, _ = np.histogram(vs, bins=edges)
            q = state.class_manager.get_class_color(class_name)
            color = (q.red(), q.green(), q.blue())
            plot.addItem(pg.PlotDataItem(
                edges, counts, stepMode="center", fillLevel=0,
                brush=pg.mkBrush(*color, 90), pen=pg.mkPen(color, width=2),
                name=f"{class_name} (n={len(vs)}, median {np.median(vs):.3g})"))

    def _refresh_tab(self, plot_widget, fetch_counts):
        """
        Helper to populate a given plot with annotation statistics.
        """
        plot_widget.clear()

        try:
            results = fetch_counts()      # [(class name, number of masks), ...]
            if not results:
                return

            class_names = [row[0] for row in results]
            counts = [row[1] for row in results]

            for i, (cls, cnt) in enumerate(zip(class_names, counts)):
                qcolor = self.parent.state_manager.class_manager.get_class_color(cls)
                brush_color = pg.mkBrush(qcolor.red(), qcolor.green(), qcolor.blue())
                bar = pg.BarGraphItem(x=[i], height=[cnt], width=0.6, brush=brush_color)
                plot_widget.addItem(bar)

            axis = plot_widget.getAxis('bottom')
            axis.setTicks([[(i, name) for i, name in enumerate(class_names)]])
        except Exception:
            logger.exception("Failed to refresh annotation statistics plot")
