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
        None until the first :meth:`preview` or :meth:`start` builds one.
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
        panel.stop_requested = self.stop
        panel.reset_requested = self.reset
        panel.remesh_requested = self.remesh
        panel.field_changed = self._on_field
        viewer.closed = self._on_viewer_closed

    def preview(self):
        """Open the viewer on the initial state of the configured run.

        The session is built and its step-zero field drawn, and the march
        waits, paused, so the first thing on screen is the state a run
        would start from; Start, Resume, and Step all proceed from it.
        """
        self._build()
        self._panel.set_paused(True)

    def start(self):
        """(Re)build the run session from the controls and march it into the
        viewer, opening the viewer sub-window first if it was closed."""
        self._build()
        self._panel.set_paused(False)
        self._timer.start(self.INTERVAL_MS)

    def remesh(self):
        """Rebuild the run at the resolution the controls now hold.

        A session owns its mesh, so a new resolution means a new session and
        a march that starts over, waiting on its initial state.
        """
        self.preview()

    def stop(self):
        """End the run where it stands, leaving its field on screen."""
        self._timer.stop()
        if self.session is not None:
            self.session.stop()
            self._draw_frame()

    def reset(self):
        """Drop the run and its viewer, back to where the panel started."""
        self._timer.stop()
        self.session = None
        self._painter = None
        self._viewer.close()
        self._panel.set_paused(False)
        self._draw_frame()

    def _build(self):
        """Halt any march, then build the configured run and draw it."""
        self._timer.stop()
        self.session = ReflectionSession(**self._panel.params())
        self._painter = FieldPainter(self.session.shock.mesh)
        self._open_viewer()

    def _open_viewer(self):
        """Open the sub-window and draw the standing run into it.

        Closing deletes the sub-window, so what opens is always an empty
        viewer and every layer of the run has to go in again.
        """
        self._viewer.open()
        self._panel.set_viewer_open(True)
        if self.session is not None:
            shock = self.session.shock
            self._viewer.update_mesh(shock.mesh)
            self._viewer.draw_shock_path(shock)
        self._draw_frame()

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
        """Read the run out into the panel, and into the viewer if it is
        open.

        The readout belongs to the run, so a closed viewer leaves the panel
        current rather than stale.
        """
        session = self.session
        if None is session:
            self._panel.set_status(None, None, None)
            return
        name = self._panel.field()
        field = session.field.field(name)
        vmin, vmax = float(field.min()), float(field.max())
        if self._viewer.is_open:
            # Scale the colors to the analytic range, not the frame's own, so
            # a field stuck short of the target looks stuck instead of
            # stretching to full color every frame.
            zones = session.analysis.zone_field(name)
            lo = min(vmin, float(zones.min()))
            hi = max(vmax, float(zones.max()))
            self._viewer.draw_field(self._painter.verts,
                                    self._painter.colors(field, lo, hi),
                                    self._painter.indices)
        self._panel.set_status(session, vmin, vmax)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
