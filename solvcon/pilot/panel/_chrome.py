# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The title bar of a docked panel, which wears the bar the C++
``RChromeStyle`` draws on a sub-window: the title on the left in the
chrome type, an optional muted note after it, and the buttons on the
right.
"""

from PySide6.QtCore import QEvent, QPointF, QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QDockWidget, QHBoxLayout, QLabel,
                               QSizePolicy, QToolButton, QWidget)

from .. import _pilot_core
from .. import _style
from .. import base
from . import _app_widgets

__all__ = [
    'DockChrome',
    'DockTitleBar',
    'set_dock_note',
]

#: The property of a dock that holds the note of its bar.
_NOTE = "chromeNote"


def _glyph(name, color, size, ratio):
    """Return the icon of the button ``name``, 'close' or 'float', drawn
    as the C++ chrome draws the buttons of a sub-window.

    :param size: The side of the button, in device-independent pixels.
    :param ratio: The device pixel ratio of the screen the button is on.
    """
    pixmap = QPixmap(round(size * ratio), round(size * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(color, 1.3)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    c, h = size / 2.0, 4.0
    if 'close' == name:
        painter.drawLine(QPointF(c - h, c - h), QPointF(c + h, c + h))
        painter.drawLine(QPointF(c - h, c + h), QPointF(c + h, c - h))
    else:
        # Two panes, the front one lifted off the back one.
        painter.drawRoundedRect(c - h, c - h + 2, 2 * h - 2, 2 * h - 2, 1, 1)
        painter.drawLine(QPointF(c - h + 2, c - h), QPointF(c + h, c - h))
        painter.drawLine(QPointF(c + h, c - h), QPointF(c + h, c + h - 2))
    painter.end()
    return QIcon(pixmap)


class DockTitleBar(_style.PaletteStyle, QWidget):
    """The title bar of a docked panel.

    It stands in for the title bar of the dock only while the dock is
    docked; a floating dock gets the native frame of its platform back.

    :param dock: The ``QDockWidget`` it is the bar of.
    """

    def __init__(self, dock):
        super().__init__(dock)
        self.dock = dock
        self.setObjectName("dockTitle")
        self.setAttribute(Qt.WA_StyledBackground, True)
        metrics = _style.chrome_metrics()
        self.setFixedHeight(metrics['bar_height'])
        side = metrics['button_side']
        self.title = QLabel()
        self.title.setObjectName("dockTitleText")
        self.note = _app_widgets.ElidedLabel()
        self.note.setObjectName("dockTitleNote")
        self.note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.buttons = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 4, 0)
        layout.setSpacing(6)
        layout.addWidget(self.title)
        layout.addWidget(self.note, 1)
        for name, slot in (('float', self._float), ('close', dock.close)):
            button = QToolButton()
            button.setObjectName("dockButton")
            button.setFixedSize(QSize(side, side))
            button.setIconSize(QSize(side, side))
            button.setFocusPolicy(Qt.TabFocus)
            button.setCursor(Qt.ArrowCursor)
            button.clicked.connect(slot)
            layout.addWidget(button)
            self.buttons[name] = button
        # The labels let the press through, so the dock drags by them.
        for label in (self.title, self.note):
            label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        dock.windowTitleChanged.connect(self._on_title)
        dock.featuresChanged.connect(self._on_features)
        self._on_title(dock.windowTitle())
        self._on_features(dock.features())
        self.set_note(dock.property(_NOTE) or "")
        self._styling = False
        self._restyle()

    def set_note(self, text):
        """Say ``text`` after the title, muted, such as the window a panel
        follows."""
        self.note.set_full_text(text)

    def _on_title(self, title):
        self.title.setText(title)
        self.dock.setAccessibleName(title)
        for name, verb in (('float', "Float"), ('close', "Close")):
            self.buttons[name].setToolTip(f"{verb} the {title} panel")
            self.buttons[name].setAccessibleName(f"{verb} {title}")

    def _on_features(self, features):
        self.buttons['float'].setVisible(
            bool(features & QDockWidget.DockWidgetFloatable))
        self.buttons['close'].setVisible(
            bool(features & QDockWidget.DockWidgetClosable))

    def _float(self):
        self.dock.setFloating(not self.dock.isFloating())

    _styling = True

    def _apply_style(self):
        # The bar may hear of a new palette before its dock has it, so it
        # reads the dock a turn of the loop later.
        if not self._styling:
            QTimer.singleShot(0, self, self._restyle)

    def _restyle(self):
        if self._styling:
            return
        self._styling = True
        try:
            # Read the dock: the sheet of the bar rewrites the palette of
            # the bar itself, so it would read its own last theme back.
            sheet = _style.ChromeRules.sheet(self.dock, "dock_title")
            if sheet != self.styleSheet():
                self.setStyleSheet(sheet)
            shades = _style.Shades(self.dock)
            color = _style.legible(shades, shades.muted, shades.panel,
                                   floor=_style.GRAPHIC_CONTRAST)
            ratio = self.devicePixelRatioF()
            for name, button in self.buttons.items():
                button.setIcon(_glyph(name, color, button.width(), ratio))
        finally:
            self._styling = False


def _bar_of(dock):
    return dock.findChild(DockTitleBar, None, Qt.FindDirectChildrenOnly)


def set_dock_note(dock, text):
    """Say ``text`` after the title of ``dock``, if it wears the chrome.

    :param dock: A ``QDockWidget``, or None.
    :param text: The note, or empty for none.
    """
    if dock is None:
        return
    # Kept on the dock too, for a bar the dock gets later.
    dock.setProperty(_NOTE, text)
    bar = _bar_of(dock)
    if bar is not None:
        bar.set_note(text)


class DockChrome(base.PilotFeature):
    """Dress every dock panel of the main window, now and later, in a
    :class:`DockTitleBar`, when the pilot draws its own chrome (see
    ``chrome_enabled()`` of the pilot core)."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.enabled = _pilot_core.chrome_enabled()
        self._pending = False
        if self.enabled:
            self._mainWindow.installEventFilter(self)
            self.dress_all()

    def populate_menu(self):
        """Nothing goes in the menus."""

    def dress_all(self):
        """Give every dock panel without a bar its bar."""
        self._pending = False
        for dock in self._mainWindow.findChildren(QDockWidget):
            if _bar_of(dock) is None:
                self._dress(dock)

    def eventFilter(self, obj, event):
        # A dock made with the main window as its parent is announced
        # before it is whole, so the dressing waits a turn of the loop.
        if (QEvent.ChildAdded == event.type() and not self._pending
                and obj is self._mainWindow):
            self._pending = True
            QTimer.singleShot(0, self, self.dress_all)
        return False

    @staticmethod
    def _dress(dock):
        bar = DockTitleBar(dock)
        dock.setTitleBarWidget(bar)

        def on_float(floating, dock=dock, bar=bar):
            # The native frame moves and resizes a floating panel as its
            # platform does.
            dock.setTitleBarWidget(None if floating else bar)
            bar.setVisible(not floating)

        dock.topLevelChanged.connect(on_float)
        if dock.isFloating():
            on_float(True)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
