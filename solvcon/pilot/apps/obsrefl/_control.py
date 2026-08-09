# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Drive one reflection run between the control panel and the viewer.

:class:`RunController` is the presenter of the app: the panel and the
viewer stay passive widgets, the session stays a Qt-free model, and every
wire between them lands here.  Panel gestures arrive through the callbacks
the controller plants on the panel, a viewer close arrives through the
viewer's ``closed``, and everything the widgets show is pushed back into
them from this side.  No widget ever reaches into a sibling.
"""

from PySide6.QtCore import QTimer

from ._field_render import FieldPainter
from ._session import ReflectionSession

__all__ = [  # noqa: F822
    'RunController',
]


class RunController(object):
    """March one run into the viewer on the frame timer.

    The run itself belongs to :class:`~._session.ReflectionSession`, which
    decides how far a chunk marches and when the run is over.  The
    controller only pumps it: one chunk per timer frame, one frame drawn
    after each, so however fast the solver is, the widgets update at the
    frame rate and no march outlives its viewer.

    :ivar session: The running :class:`~._session.ReflectionSession`, or
        None before the first start.
    """

    #: Qt timer interval in milliseconds.
    INTERVAL_MS = 50

    def __init__(self, panel, viewer):
        self._panel = panel
        self._viewer = viewer
        self.session = None
        self._painter = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._advance)
        panel.viewer_toggled = self._on_viewer
        panel.start_requested = self.start
        panel.pause_toggled = self._on_pause
        panel.step_requested = self._on_step
        panel.field_changed = self._on_field
        viewer.closed = self._on_viewer_closed

    def start(self):
        """(Re)build the run session from the controls and march it into the
        viewer, opening the viewer sub-window first if it was closed."""
        self._timer.stop()
        self._open_viewer()
        self.session = ReflectionSession(**self._panel.params())
        shock = self.session.shock
        self._viewer.update_mesh(shock.mesh)
        self._viewer.draw_shock_path(shock)
        self._painter = FieldPainter(shock.mesh)
        self._panel.set_paused(False)
        self._draw_frame()
        self._timer.start(self.INTERVAL_MS)

    def _open_viewer(self):
        self._viewer.open()
        self._panel.set_viewer_open(True)

    def _on_viewer(self, open_):
        if open_:
            self._open_viewer()
        else:
            self._viewer.close()

    def _on_viewer_closed(self):
        # Reached from the sub-window's close event; stop the run before Qt
        # frees the viewer.
        self._timer.stop()
        self._panel.set_viewer_open(False)

    def _on_pause(self, paused):
        if self.session is None:
            return
        if paused:
            self._timer.stop()
        elif self._viewer.is_open:
            self._timer.start(self.INTERVAL_MS)

    def _on_step(self):
        if self.session is not None and self._viewer.is_open:
            self._march_frame()

    def _on_field(self, _name):
        if self.session is not None:
            self._draw_frame()

    def _advance(self):
        if not self._viewer.is_open:
            self._timer.stop()
            return
        if self.session.finished:
            self._timer.stop()
            self._panel.set_paused(True)
            return
        self._march_frame()

    def _march_frame(self):
        """March one chunk and draw what it left behind.

        The chunk length is read from the control every frame, so turning
        the dial mid-run takes effect on the next one.
        """
        self.session.steps_per_chunk = self._panel.steps_per_frame()
        self.session.advance()
        self._draw_frame()

    def _draw_frame(self):
        if not self._viewer.is_open:
            return
        session = self.session
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

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
