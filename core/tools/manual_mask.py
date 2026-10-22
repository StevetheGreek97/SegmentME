"""Manual polygon: left click adds a point, right click removes the last
one, S saves (at least 3 points)."""
import numpy as np
from PyQt6.QtGui import QPen, QColor
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem

from core.tools.base import MaskTool
from services.logger import get_logger

logger = get_logger(__name__)

_POINT_PEN = QPen(QColor(0, 255, 0), 2)
_LINE_PEN = QPen(QColor(255, 0, 0), 2)


class ManualMask(MaskTool):
    label = "manual"

    def __init__(self, host):
        super().__init__(host)
        self.temp_points = []  # ellipse items, one per point
        self.temp_lines = []   # line items between consecutive points

    # -------------------------------------------------------------- input
    def left_click(self, point, ctrl=False):
        self.add_point(point)

    def right_click(self, point):
        self.pop_last_point()

    def add_point(self, point):
        """Add a point (x, y) to the polygon."""
        ellipse = QGraphicsEllipseItem(point[0] - 2, point[1] - 2, 4, 4)
        ellipse.setPen(_POINT_PEN)
        ellipse.setBrush(QColor(0, 255, 0))
        self.scene.addItem(ellipse)
        self.temp_points.append(ellipse)

        if len(self.temp_points) > 1:
            a = self.temp_points[-2].rect().center()
            b = ellipse.rect().center()
            line = QGraphicsLineItem(a.x(), a.y(), b.x(), b.y())
            line.setPen(_LINE_PEN)
            self.scene.addItem(line)
            self.temp_lines.append(line)

    def pop_last_point(self):
        """Remove the last point and the line to it."""
        if not self.temp_points:
            logger.debug("Undo requested with no points to remove")
            return
        self.scene.removeItem(self.temp_points.pop())
        if self.temp_lines:
            self.scene.removeItem(self.temp_lines.pop())

    # ------------------------------------------------------------- saving
    def current_polygon(self):
        if not self.temp_points:
            return None
        return np.array([(int(item.rect().center().x()), int(item.rect().center().y()))
                         for item in self.temp_points], dtype=np.float32)

    def clear_temp_items(self):
        for item in self.temp_lines + self.temp_points:
            self.scene.removeItem(item)
        self.temp_lines.clear()
        self.temp_points.clear()
