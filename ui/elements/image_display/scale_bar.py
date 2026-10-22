"""
Scale bar in the bottom-left corner of the image view: a round length
(1 mm, 500 µm, ...) drawn at the current zoom. A child widget of the view's
viewport, so it stays put while the image is scrolled.
"""
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from core.measure.morphometrics import format_length, nice_bar_length

_MARGIN = 14
_PAD = 8


class ScaleBar(QWidget):
    def __init__(self, view):
        super().__init__(view.viewport())
        self._view = view
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        font = QFont(self.font())
        font.setPointSizeF(max(font.pointSizeF(), 9.0))
        font.setBold(True)
        self.setFont(font)
        self._state = None
        self.bar_px = 0.0
        self.text = ""
        self.hide()

    def set_state(self, um_per_px, zoom):
        """um_per_px: the image's scale (None = no bar); zoom: screen px per image px."""
        viewport = self._view.viewport().size()
        state = (um_per_px, round(zoom, 6), viewport.width(), viewport.height())
        if state == self._state:
            return
        self._state = state
        if not um_per_px or zoom <= 0:
            self.hide()
            return
        um_per_screen_px = um_per_px / zoom
        bar_um = nice_bar_length(um_per_screen_px)
        self.bar_px = bar_um / um_per_screen_px
        self.text = format_length(bar_um)
        text_w = QFontMetricsF(self.font()).horizontalAdvance(self.text)
        height = QFontMetricsF(self.font()).height()
        self.resize(int(max(self.bar_px, text_w) + 2 * _PAD) + 1, int(height + 6 + 2 * _PAD) + 1)
        self.place()
        self.show()
        self.update()

    def place(self):
        """Back to the bottom-left corner. Scrolling the view moves the
        viewport's child widgets along with the image, so the view calls this
        after every scroll."""
        viewport = self._view.viewport().size()
        self.move(_MARGIN, viewport.height() - self.height() - _MARGIN)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 140))
        p.drawRoundedRect(QRectF(self.rect()), 6, 6)

        fm = QFontMetricsF(self.font())
        p.setPen(QColor(255, 255, 255))
        text_rect = QRectF(_PAD, _PAD - 1, self.width() - 2 * _PAD, fm.height())
        p.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.text)

        y = self.height() - _PAD - 2
        x0 = (self.width() - self.bar_px) / 2
        pen = QPen(QColor(255, 255, 255))
        pen.setWidthF(3)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        p.setPen(pen)
        p.drawLine(QRectF(x0, y, self.bar_px, 0).topLeft(), QRectF(x0, y, self.bar_px, 0).topRight())
        pen.setWidthF(2)
        p.setPen(pen)
        for x in (x0 + 1, x0 + self.bar_px - 1):     # end ticks
            p.drawLine(QRectF(x, y - 5, 0, 8).topLeft(), QRectF(x, y - 5, 0, 8).bottomLeft())
        p.end()
