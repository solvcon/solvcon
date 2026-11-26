# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The pilot feature of the shock tube."""

from ...pilot import base

from . import _column
from . import _control
from . import _figure
from . import _window

__all__ = [
    'ShockTubeApp',
    'build_app',
]


class ShockTubeApp(base.PilotFeature):
    """The shock-tube app, opened from "App > Shock tube".

    The app is a sub-window, opened maximized, and the control column in
    the pilot's "App config" panel: a second pick of the menu item brings
    the window back instead of building another, and closing the window
    takes the column along.  The feature composes the window, the
    column, and the :class:`~._control.ShockTubeController` that wires
    them.
    """

    def __init__(self, *args, **kw):
        #: The pilot's :class:`~solvcon.pilot.panel.AppConfig` that holds
        #: :attr:`column`.
        self.app_config = kw.pop('app_config')
        super().__init__(*args, **kw)
        #: The open :class:`~._window.ShockTubeWindow`, or None.
        self.window = None
        #: The :class:`~._column.ControlColumn` beside it, or None.
        self.column = None
        #: Their :class:`~._control.ShockTubeController`, or None.
        self.controller = None
        #: The MDI sub-window holding :attr:`window`, or None.
        self.subwindow = None
        #: The :class:`~solvcon.pilot.base.SubWindowDetacher` that moves
        #: :attr:`window` out to a window of its own and back, or None.
        self.detacher = None
        self._close_filter = None

    def populate_menu(self):
        self.add_action(
            "App", "Shock tube",
            "March the 1D Euler solver through Sod's shock tube and compare "
            "it with the exact solution", self.open,
            id="app.shocktube", weight=10)

    def open(self):
        """Open the window maximized, or bring it back."""
        if self.subwindow is not None:
            self.detacher.bring_forward()
            return
        self.window = _window.ShockTubeWindow(_figure.ShockTubeFigure())
        self.column = _column.ControlColumn()
        self.controller = _control.ShockTubeController(self.window,
                                                       self.column)
        self.controller.title_changed = self._set_title
        self.column.key_pressed = self.window.press_key
        self.subwindow = self._mgr.addSubWindow(self.window)
        self.detacher = base.setup_app_subwindow(self.subwindow, self._mgr,
                                                 self.app_config)
        self._set_title(self.controller.setup.title())
        self._close_filter = base.SubWindowCloseFilter(self._on_closed,
                                                       self.subwindow)
        self.subwindow.installEventFilter(self._close_filter)
        self.subwindow.showMaximized()
        self.app_config.add(self.subwindow, self.column)
        self.window.setFocus()

    def close(self):
        """Close the window, which stops its run."""
        if self.subwindow is not None:
            self.subwindow.close()

    def _set_title(self, title):
        if self.subwindow is not None:
            self.subwindow.setWindowTitle(title)

    def _on_closed(self):
        # The sub-window deletes itself and the window inside on the next
        # turn of the event loop, so the timer stops before that.
        if self.controller is not None:
            self.controller.shutdown()
        self.app_config.remove(self.subwindow)
        self.controller = None
        self.window = None
        self.column = None
        self.subwindow = None
        self.detacher = None
        self._close_filter = None


def build_app(controller):
    """Build the app on the pilot.

    :param controller: The pilot controller, which holds the manager and
        the App config panel.
    """
    return ShockTubeApp(mgr=controller.mgr,
                        app_config=controller.app_config)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
