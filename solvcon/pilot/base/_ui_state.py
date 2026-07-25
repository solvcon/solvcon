# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Remember the pilot user-interface state across sessions.
"""


from PySide6 import QtCore, QtGui

from ...config import Config
from . import _gui_common

__all__ = [
    'UiState',
    'WindowGeometry',
    'apply_geometry',
    'capture_geometry',
]


def _is_int(value):
    """Whether ``value`` is a plain integer, excluding ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _on_screen(x, y):
    """Whether the point ``(x, y)`` lies on some connected screen."""
    point = QtCore.QPoint(x, y)
    for screen in QtGui.QGuiApplication.screens():
        if screen.availableGeometry().contains(point):
            return True
    return False


def apply_geometry(window, section):
    """Apply a stored geometry ``section`` to ``window``."""
    if not isinstance(section, dict):
        return
    width, height = section.get("width"), section.get("height")
    if _is_int(width) and _is_int(height) and width > 0 and height > 0:
        window.resize(width, height)
    x, y = section.get("x"), section.get("y")
    if _is_int(x) and _is_int(y) and _on_screen(x, y):
        window.move(x, y)


def capture_geometry(window):
    """Return the current size and location of ``window`` as a section."""
    size, pos = window.size(), window.pos()
    return {
        "width": size.width(),
        "height": size.height(),
        "x": pos.x(),
        "y": pos.y(),
    }


class WindowGeometry(object):
    """The size and location of a window, as one piece of UI state."""

    #: Name of this part within the UI state.
    KEY = "window"

    def __init__(self, window):
        self._window = window

    def apply(self, section):
        """Resize and move the window to match ``section``."""
        apply_geometry(self._window, section)

    def capture(self):
        """Return the window's present size and location."""
        return capture_geometry(self._window)


class UiState(_gui_common.PilotFeature):
    """Persist the pilot user-interface state in the user configuration.

    The state is a set of parts stored under the ``ui`` section. A part
    carries a ``KEY`` naming its section, an ``apply`` taking that section,
    and a ``capture`` returning it. :meth:`add` registers one, and a part
    whose section is absent receives ``None``.
    """

    #: Configuration key holding every UI-state part.
    SECTION = "ui"

    def __init__(self, *args, config=None, parts=None, **kw):
        """
        :param config: Store the state is read from and written to.
            Defaults to the configuration in the user's directory.
        :type config: solvcon.config.Config
        :param parts: The parts to keep, in the order they are applied.
            Defaults to the main window's geometry alone.
        :type parts: iterable
        """
        super(UiState, self).__init__(*args, **kw)
        self._config = Config() if config is None else config
        if parts is None:
            parts = [WindowGeometry(self._mainWindow)]
        self._parts = list(parts)
        self._config.load()
        self.restore()
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.save)

    def add(self, part):
        """Register one more part, returning it for the caller to keep."""
        self._parts.append(part)
        return part

    def restore(self):
        """Apply every stored part to the interface."""
        state = self._config.get(self.SECTION)
        if not isinstance(state, dict):
            state = {}
        for part in self._parts:
            part.apply(state.get(part.KEY))

    def save(self):
        """Capture every part and persist them together."""
        state = {part.KEY: part.capture() for part in self._parts}
        self._config.set(self.SECTION, state)
        self._config.save()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
