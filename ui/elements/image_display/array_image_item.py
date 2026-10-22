"""
The image in the view, drawn straight from its numpy array.

A QGraphicsPixmapItem needs a QPixmap: a third full copy of the scan in
memory (~360 MB for a 7610 x 11842 scan, next to the image and the image
with masks) and ~100 ms to make for every image shown. Drawing the array
through a QImage that shares its memory is as fast (measured: ~2 ms per
repaint fitted, <1 ms zoomed in) and needs neither. Masks redrawn in place
in the array only need update() of that area.
"""
import numpy as np
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QGraphicsItem


class ArrayImageItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self._array = None          # keeps the memory the QImage points at alive
        self._qimage = QImage()

    def set_array(self, array):
        """Show this H x W x 3 uint8 RGB array (shared, not copied), or None."""
        if array is not None:
            array = np.ascontiguousarray(array)
        if array is self._array:
            self.update()
            return
        self.prepareGeometryChange()
        self._array = array
        if array is None:
            self._qimage = QImage()
        else:
            h, w = array.shape[:2]
            self._qimage = QImage(array.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.update()

    def array(self):
        return self._array

    def image(self) -> QImage:
        """A QImage of what is shown (shares the array's memory)."""
        return self._qimage

    def isNull(self) -> bool:
        return self._array is None

    def changed(self, rects):
        """The array was changed in place inside these (x0, y0, x1, y1) boxes."""
        for x0, y0, x1, y1 in rects:
            self.update(QRectF(x0, y0, x1 - x0, y1 - y0))

    def boundingRect(self):
        if self._array is None:
            return QRectF()
        h, w = self._array.shape[:2]
        return QRectF(0, 0, w, h)

    def paint(self, painter, option, widget=None):
        if self._array is None:
            return
        # The view clips the painter to the area being repainted; Qt only
        # samples the image there (as QGraphicsPixmapItem does).
        painter.drawImage(0, 0, self._qimage)
