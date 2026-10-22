"""
Outlines of the selected masks, drawn on top of the image.

Thousands of masks can be selected at once ("select all" on a busy scan).
Drawn as one big path with a 2 px smoothed pen, Qt took ~4 s per repaint
for 5,000 outlines, and every repaint (Next, pan, zoom) paid it. This item
instead draws each outline on its own, skips those outside the area being
repainted, and uses 1 px lines -- Qt's fast path -- drawn as a 2 px white
line with a dark edge, so it stays visible on any background: well under
0.1 s for 5,000 outlines on a whole scan, a few ms zoomed in.
"""
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsItem


class SelectionOutlines(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self._polygons = []        # QPolygonF in image coordinates
        self._boxes = []           # their bounding rects
        self._rect = QRectF()
        self._light = QPen(QColor(255, 255, 255))
        self._light.setWidth(0)    # 0 = cosmetic 1 px: the fast path
        self._dark = QPen(QColor(0, 0, 0, 200))
        self._dark.setWidth(0)
        # paint() gets the area actually being repainted, not the whole item
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)

    def set_polygons(self, polygons):
        """Show these outlines (list of QPolygonF); [] hides them."""
        self.prepareGeometryChange()
        self._polygons = list(polygons)
        self._boxes = [p.boundingRect() for p in self._polygons]
        rect = QRectF()
        for box in self._boxes:
            rect = rect.united(box)
        self._rect = rect.adjusted(-2, -2, 2, 2) if self._boxes else QRectF()
        self.update()

    def count(self) -> int:
        return len(self._polygons)

    def boundingRect(self):
        return self._rect

    def paint(self, painter, option, widget=None):
        if not self._polygons:
            return
        exposed = option.exposedRect
        visible = [p for p, box in zip(self._polygons, self._boxes) if box.intersects(exposed)]
        if not visible:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        one_px = 1.0 / max(abs(painter.transform().m11()), 1e-9)   # a screen pixel in image units
        # dark edge one pixel down-right, then the white line twice (here and
        # one pixel up-left): reads as a 2 px outline, at 1 px drawing cost
        for (dx, dy), pen in (((1, 1), self._dark), ((-1, -1), self._light), ((0, 0), self._light)):
            painter.setPen(pen)
            painter.translate(dx * one_px, dy * one_px)
            for polygon in visible:
                painter.drawPolygon(polygon)
            painter.translate(-dx * one_px, -dy * one_px)
