# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Pilot feature that runs the 2D Euler oblique-shock reflection and draws a
selected solution field as a flat color map.

The feature mirrors the mesh information panel: a toggle in the View
"Panels" submenu owns the control widget from :mod:`._panel`, and the
callbacks that widget fires build the run session from :mod:`._session`,
open the domain viewer sub-window, and pump the session into it on a timer.
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDockWidget

from ...base import _gui_common
from ._field_render import FieldPainter
from ._panel import SolutionPanel
from ._session import ReflectionSession
from ._viewer import DomainViewer

__all__ = [  # noqa: F822
    'ObliqueShockApp',
]


class ObliqueShockApp(_gui_common.PilotFeature):
    """Euler solver panel, toggled from the View "Panels" submenu.

    The panel owns one domain viewer sub-window and one solver run.  The viewer
    control opens and closes the sub-window; starting marches the run into it
    on a timer; closing it stops the march.

    The run itself belongs to :class:`~._session.ReflectionSession`, which
    decides how far a chunk marches and when the run is over.  This feature
    only pumps it: one chunk per timer frame, one frame drawn after each.
    """

    #: Qt timer interval in milliseconds.
    INTERVAL_MS = 50

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Fired after a run sets the viewer mesh, so the inspector can refresh.
        self.viewer_updated = None
        self._action = None
        self._dock = None
        self._panel = None
        self._session = None
        self._painter = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._advance)
        self._viewer = DomainViewer(self._mgr)
        self._viewer.closed = self._on_viewer_closed
        self._viewer.mesh_updated = self._notify_viewer_updated

    def populate_menu(self):
        self._action = self.add_action(
            "View/Panels", "Euler solver", "Toggle the Euler solver panel",
            None, id="panel.euler_solver", weight=20, checkable=True)
        self._action.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked):
        if checked:
            self._ensure_panel()
            self._dock.show()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_panel(self):
        if self._panel is not None:
            return
        self._panel = SolutionPanel()
        self._panel.viewer_toggled = self._on_viewer
        self._panel.start_requested = self._on_start
        self._panel.pause_toggled = self._on_pause
        self._panel.step_requested = self._on_step
        self._panel.field_changed = self._on_field
        self._dock = QDockWidget("euler solver")
        self._dock.setWidget(self._panel)
        self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea, self._dock)
        self._dock.visibilityChanged.connect(self._action.setChecked)

    def _on_viewer(self, open_):
        if open_:
            self._open_viewer()
        else:
            self._viewer.close()

    def _open_viewer(self):
        """Open the domain viewer sub-window if it is not already open."""
        self._viewer.open()
        self._panel.set_viewer_open(True)

    def _on_viewer_closed(self):
        # Reached from the sub-window's close event; stop the run before Qt
        # frees the viewer.
        self._stop_timer()
        self._panel.set_viewer_open(False)

    def _notify_viewer_updated(self):
        # Fired after a run sets the viewer mesh; the wiring in the outer
        # controller points it at the inspector's resync.
        if self.viewer_updated is not None:
            self.viewer_updated()

    def _on_start(self):
        """(Re)build the run session from the controls and march it into the
        viewer, opening the viewer sub-window first if it was closed."""
        self._stop_timer()
        self._open_viewer()
        self._session = ReflectionSession(**self._panel.params())
        shock = self._session.shock
        self._viewer.update_mesh(shock.mesh)
        self._viewer.draw_shock_path(shock)
        self._painter = FieldPainter(shock.mesh)
        self._panel.set_paused(False)
        self._draw_frame()
        self._timer.start(self.INTERVAL_MS)

    def _on_pause(self, paused):
        if self._session is None:
            return
        if paused:
            self._timer.stop()
        elif self._viewer.is_open:
            self._timer.start(self.INTERVAL_MS)

    def _on_step(self):
        if self._session is not None and self._viewer.is_open:
            self._march_frame()

    def _on_field(self, _name):
        if self._session is not None:
            self._draw_frame()

    def _advance(self):
        if not self._viewer.is_open:
            self._stop_timer()
            return
        if self._session.finished:
            self._stop_timer()
            self._panel.set_paused(True)
            return
        self._march_frame()

    def _march_frame(self):
        """March one chunk and draw what it left behind.

        The chunk length is read from the control every frame, so turning
        the dial mid-run takes effect on the next one.
        """
        self._session.steps_per_chunk = self._panel.steps_per_frame()
        self._session.advance()
        self._draw_frame()

    def _draw_frame(self):
        if not self._viewer.is_open:
            return
        session = self._session
        name = self._panel.field()
        field = session.field.field(name)
        vmin, vmax = float(field.min()), float(field.max())
        # Scale the colors to the analytic range, not the frame's own, so a
        # field stuck short of the target looks stuck instead of stretching
        # to full color every frame.
        zones = session.analysis.zone_field(name)
        lo = min(vmin, float(zones.min()))
        hi = max(vmax, float(zones.max()))
        self._viewer.draw_field(self._painter.verts,
                                self._painter.colors(field, lo, hi),
                                self._painter.indices)
        self._panel.set_status(session, vmin, vmax)

    def _stop_timer(self):
        self._timer.stop()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
