"""Startup splash screen.

Shows the SegmentME logo (splash.png: the mark plus its own gradient
"SegmentME" wordmark, resources/icons/splash_icon.svg) with a "Loading..."
status line immediately when the app is launched, while the slow imports in
main.py (torch, ultralytics, the rest of the Qt UI) run underneath. Without
this the app just looks hung for a few seconds after a double-click --
worse in a frozen build, where those imports are slower than running from
source.

main.py creates the QApplication and this splash screen first, before doing
any of those imports, then calls .close() once the startup dialog or main
window is ready to take over.
"""
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import QSplashScreen

from services.file_handlers import get_resource_path

_MESSAGE = "Loading..."
_MESSAGE_RECT = QRect(0, 270, 440, 30)  # bottom strip of splash.png (440x310)
_MESSAGE_COLOR = QColor(110, 110, 110)
_SHADOW_COLOR = QColor(255, 255, 255, 235)  # no card behind the text, so it needs one


class SplashScreen(QSplashScreen):
    """Fully transparent (no card/background rectangle behind the logo) --
    splash.png has real alpha and the window itself is translucent, so only
    the logo, wordmark and status text are visible."""

    def __init__(self):
        base_pixmap = QPixmap(get_resource_path("resources/icons/splash.png"))
        super().__init__(base_pixmap, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        composed = QPixmap(base_pixmap)  # keeps the transparent background
        painter = QPainter(composed)
        self._draw_shadowed_text(painter, _MESSAGE_RECT, _MESSAGE, _MESSAGE_COLOR)
        painter.end()
        self.setPixmap(composed)

    @staticmethod
    def _draw_shadowed_text(painter, rect, text, color, point_size=10):
        # No card behind the text (the window is fully transparent), so
        # it's drawn with a soft light shadow for legibility over any
        # desktop background, instead of QSplashScreen's plain showMessage().
        font = QFont(painter.font().family(), point_size)
        painter.setFont(font)
        painter.setPen(_SHADOW_COLOR)
        painter.drawText(rect.translated(0, 1), Qt.AlignmentFlag.AlignCenter, text)
        painter.setPen(color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
