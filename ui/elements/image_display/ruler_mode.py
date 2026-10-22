"""
Ruler: drag a line along something of known length (a ruler, a scale bar)
to set the project's scale. Used by the Set Scale window.
"""
import math

from PyQt6.QtCore import QLineF, Qt
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import QGraphicsLineItem, QGraphicsSimpleTextItem

from ui.elements.image_display.edit_modes import MaskEditMode


class RulerMode(MaskEditMode):
    """Press, drag, release: `on_done(length in pixels)` is called; Esc or a
    click without dragging calls `on_done(None)`."""

    def __init__(self, display, on_done):
        self.display = display
        self.state = display.parent.state_manager
        self.mask_id = ""
        self._on_done = on_done
        self._start = None
        self._line = None
        self._label = None
        self._result = None

    def start(self):
        pen = QPen(QColor(255, 210, 0))
        pen.setWidthF(2)
        pen.setCosmetic(True)
        self._line = QGraphicsLineItem()
        self._line.setPen(pen)
        self._line.setZValue(30)
        self._label = QGraphicsSimpleTextItem()
        self._label.setBrush(QColor(255, 210, 0))
        self._label.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self._label.setZValue(31)
        self.display.scene.addItem(self._line)
        self.display.scene.addItem(self._label)
        self._line.hide()
        self._label.hide()
        return True

    def stop(self):
        for item in (self._line, self._label):
            if item is not None and item.scene() is not None:
                item.scene().removeItem(item)
        self._line = self._label = None
        callback, self._on_done = self._on_done, None
        if callback is not None:
            callback(self._result)

    def mouse_press(self, point, button):
        if button == Qt.MouseButton.LeftButton:
            self._start = point
            self._update(point)
            self._line.show()
            self._label.show()
        return True  # nothing else reacts to clicks while measuring

    def mouse_move(self, point):
        if self._start is not None:
            self._update(point)
            return True
        return False

    def mouse_release(self, button):
        if button != Qt.MouseButton.LeftButton or self._start is None:
            return True
        length = self._line.line().length()
        self._result = length if length >= 2 else None
        self.finish()
        return True

    def _update(self, point):
        x0, y0 = self._start
        self._line.setLine(QLineF(x0, y0, point[0], point[1]))
        self._label.setText(f"{math.hypot(point[0] - x0, point[1] - y0):.1f} px")
        self._label.setPos(point[0], point[1])
