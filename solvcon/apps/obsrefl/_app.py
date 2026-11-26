# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The pilot feature of the oblique-shock reflection."""

from PySide6.QtCore import QTimer

from ...pilot import base

from . import _column
from . import _control
from . import _figure
from . import _window

__all__ = [
    'ObsreflApp',
    'build_app',
]


class ObsreflApp(base.PilotFeature):
    """The reflection app, opened from "App > Oblique shock reflection".

    The app is a sub-window, opened maximized, and the control column in
    the pilot's "App config" panel: a second pick of the menu item brings
    the window back instead of building another, and closing the window
    takes the column along.  The feature composes the window, the
    column, and the :class:`~._control.ReflectionController` that wires
    them, and forwards the inspector nudge to the outer GUI.
    """

    def __init__(self, *args, **kw):
        #: The pilot's :class:`~solvcon.pilot.panel.AppConfig` that holds
        #: :attr:`column`.
        self.app_config = kw.pop('app_config')
        super().__init__(*args, **kw)
        #: The open :class:`~._window.ReflectionWindow`, or None.
        self.window = None
        #: The :class:`~._column.ControlColumn` beside it, or None.
        self.column = None
        #: Their :class:`~._control.ReflectionController`, or None.
        self.controller = None
        #: The MDI sub-window holding :attr:`window`, or None.
        self.subwindow = None
        #: The :class:`~solvcon.pilot.base.SubWindowDetacher` that moves
        #: :attr:`window` out to a window of its own and back, or None.
        self.detacher = None
        #: Owner-supplied callback fired after the figure shows a new mesh, so
        #: the inspector can read it.
        self.mesh_updated = None
        self._close_filter = None

    def populate_menu(self):
        self.add_action(
            "App", "Oblique shock reflection",
            "March the 2D Euler solver through a shock reflecting off a "
            "wall and compare it with the exact solution", self.open,
            id="app.obsrefl", weight=20)

    def open(self):
        """Open the window maximized, or bring it back."""
        if self.subwindow is not None:
            self.detacher.bring_forward()
            return
        figure = _figure.ReflectionFigure(mgr=self._mgr)
        figure.domain.viewer.mesh_updated = self._notify_mesh_updated
        self.window = _window.ReflectionWindow(figure)
        self.column = _column.ControlColumn()
        self.controller = _control.ReflectionController(self.window,
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
        # The mesh went in before the view had its size, so it is framed
        # again once the maximized window is laid out; the inspector reads
        # the view only now that its sub-window is the active one.
        QTimer.singleShot(0, self._frame_domain)
        self._notify_mesh_updated()

    def close(self):
        """Close the window, which stops its run."""
        if self.subwindow is not None:
            self.subwindow.close()

    def _frame_domain(self):
        if self.window is not None:
            domain = self.window.figure.domain
            if domain.available:
                domain.view.fitCameraToScene()

    def _set_title(self, title):
        if self.subwindow is not None:
            self.subwindow.setWindowTitle(title)

    def _notify_mesh_updated(self):
        if self.mesh_updated is not None:
            self.mesh_updated()

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
    """Build the app on the pilot and wire it to the inspector.

    :param controller: The pilot controller, which holds the manager and
        the App config panel.
    """
    app = ObsreflApp(mgr=controller.mgr,
                     app_config=controller.app_config)
    app.mesh_updated = controller.tree_panel.resync
    return app

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
