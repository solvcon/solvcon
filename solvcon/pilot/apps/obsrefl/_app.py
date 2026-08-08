# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Pilot feature that runs the 2D Euler oblique-shock reflection and draws a
selected solution field as a flat color map.

The feature mirrors the mesh information panel: a toggle in the View
"Panels" submenu owns the control widget from :mod:`._panel`, and the
callbacks that widget fires build the run session from :mod:`._session`,
open the domain viewer sub-window, and pump the session into it on a timer.
"""

import collections

import numpy as np

from PySide6.QtCore import Qt, QTimer, QObject, QEvent
from PySide6.QtWidgets import QDockWidget

from .... import core
from ...base import _gui_common
from . import _field_render
from ._panel import SolutionPanel
from ._session import ReflectionSession

__all__ = [  # noqa: F822
    'ObliqueShockApp',
]


#: What the viewer needs to recolor one run: the packed triangle vertices, the
#: indices into them, and how many triangles each cell contributed.  It is cut
#: from the mesh, so it outlives every frame of the run.
FrameGeometry = collections.namedtuple('FrameGeometry',
                                       ['verts', 'indices', 'counts'])


class _SubWindowCloseFilter(QObject):
    """Report a watched sub-window's close synchronously.

    A ``QMdiSubWindow`` has no close signal, so this filter is the only way to
    stop the march before Qt frees the viewer.
    """

    def __init__(self, on_close, parent):
        super().__init__(parent)
        self._on_close = on_close

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Close:
            self._on_close()
        return False


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
    #: Lift the analytic shock overlay off the z = 0 field plane so it is
    #: not z-fought away by the colored triangles.
    PATH_LIFT = 0.01

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Fired after a run sets the viewer mesh, so the inspector can refresh.
        self.viewer_updated = None
        self._action = None
        self._dock = None
        self._panel = None
        self._session = None
        self._frame = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._advance)
        self._viewer = None
        self._subwin = None
        self._close_filter = None
        # Held for the panel's lifetime: a throwaway mdiArea wrapper is
        # garbage-collected right after use, invalidating any sub-window handle
        # taken through it.
        self._mdi = None

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
            self._close_viewer()

    def _open_viewer(self):
        """Open the domain viewer sub-window if it is not already open."""
        if self._viewer is not None:
            return
        self._viewer = self._mgr.add3DWidget()
        self._viewer.showAxis(True)
        # Delete the sub-window on close and watch for that close, so a close
        # from any source stops the march before Qt frees the viewer.
        if self._mdi is None:
            self._mdi = self._mgr.mdiArea
        self._subwin = self._mdi.activeSubWindow()
        if self._subwin is not None:
            self._subwin.setAttribute(Qt.WA_DeleteOnClose, True)
            self._close_filter = _SubWindowCloseFilter(
                self._on_viewer_closed, self._subwin)
            self._subwin.installEventFilter(self._close_filter)
        self._panel.set_viewer_open(True)

    def _close_viewer(self):
        """Close the viewer sub-window; its close event stops the run."""
        if self._subwin is not None:
            self._subwin.close()
        else:
            self._on_viewer_closed()

    def _on_viewer_closed(self):
        # Reached from the sub-window's close event; stop the run and drop the
        # viewer before Qt frees it.
        self._stop_timer()
        self._viewer = None
        self._subwin = None
        self._close_filter = None
        self._panel.set_viewer_open(False)

    def _viewer_alive(self):
        """True while the viewer is open; the close filter clears it."""
        return self._viewer is not None

    def _on_start(self):
        """(Re)build the run session from the controls and march it into the
        viewer, opening the viewer sub-window first if it was closed."""
        self._stop_timer()
        self._open_viewer()
        self._session = ReflectionSession(**self._panel.params())
        shock = self._session.shock
        # Set the viewer mesh so the inspector can report it; reusing an open
        # viewer raises no activation, so nudge the inspector directly.
        if self._viewer is not None:
            self._viewer.updateMesh(shock.mesh)
            self._draw_shock_path(shock)
            if self.viewer_updated is not None:
                self.viewer_updated()
        self._frame = self._build_frame(shock.mesh)
        self._panel.set_paused(False)
        self._draw_frame()
        self._timer.start(self.INTERVAL_MS)

    def _on_pause(self, paused):
        if self._session is None:
            return
        if paused:
            self._timer.stop()
        elif self._viewer_alive():
            self._timer.start(self.INTERVAL_MS)

    def _on_step(self):
        if self._session is not None and self._viewer_alive():
            self._march_frame()

    def _on_field(self, _name):
        if self._session is not None:
            self._draw_frame()

    def _advance(self):
        if not self._viewer_alive():
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

    @staticmethod
    def _build_frame(mesh):
        """Cut the :class:`FrameGeometry` of a run from its mesh.

        ``updateColorField`` wants an indexed vertex soup; the cell fan
        already is one, so its vertices are packed once, the geometry being
        fixed for the run, and indexed sequentially.
        """
        fan, counts = _field_render.cell_triangulation(mesh)
        verts = fan.pack_array().ndarray.reshape(-1, 3)
        indices = np.arange(verts.shape[0], dtype='uint32').reshape(-1, 3)
        return FrameGeometry(core.SimpleArrayFloat32(array=verts),
                             core.SimpleArrayUint32(array=indices), counts)

    def _draw_shock_path(self, shock):
        """Overlay the analytic shock polyline on the viewer.

        The two-arm measurement draws the incident and the reflected shock
        as one path with the reflection angle annotated, so the computed
        field can be judged against where the shocks have to stand.
        """
        path = [(x, y, self.PATH_LIFT) for x, y in shock.shock_path()]
        if 3 == len(path):
            self._viewer.measureAngle(path[0], path[1], path[2])
        else:
            self._viewer.measureDistance(path[0], path[1])

    def _draw_frame(self):
        if not self._viewer_alive():
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
        colors = _field_render.field_colors(field, self._frame.counts, lo, hi)
        self._viewer.updateColorField(self._frame.verts,
                                      core.SimpleArrayFloat32(array=colors),
                                      self._frame.indices)
        self._panel.set_status(session, vmin, vmax)

    def _stop_timer(self):
        self._timer.stop()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
