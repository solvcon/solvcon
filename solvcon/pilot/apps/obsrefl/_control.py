# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Drive one reflection run between the control panel and the viewer.

:class:`RunController` is the presenter of the app: the panel and the
viewer stay passive widgets, the session stays a Qt-free model, and every
wire between them lands here.  Panel gestures arrive through the callbacks
the controller plants on the panel, a viewer close arrives through the
viewer's ``closed``, and everything the widgets show is pushed back into
them from this side.  No widget ever reaches into a sibling.

The movie a run records is driven from here as well.  It rides on the
frames the viewer draws, so it belongs beside the march rather than in the
run session, which stays free of the GUI it is watched from.
"""

from PySide6.QtCore import QTimer

from ...visual import _movie
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
    :ivar movie: The open :class:`~solvcon.pilot.visual.MovieRecorder`, or
        None while the run is not being recorded.
    :ivar reported: Owner-supplied callback that says what became of a
        movie, or None to keep it to the panel.
    """

    #: Qt timer interval in milliseconds.
    INTERVAL_MS = 50

    def __init__(self, panel, viewer):
        self._panel = panel
        self._viewer = viewer
        self.session = None
        self.movie = None
        self.reported = None
        self._painter = None
        # The mesh flavor the standing run was built with, which names its
        # movie; the control may have moved on since.
        self._cell_type = None
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
        panel.placement_changed = self._on_bar_placement
        panel.record_toggled = self._on_record
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
        viewer, opening the viewer sub-window first if it was closed.

        A run started with the record box ticked records from its first
        frame, so the movie opens on the state the march starts from.
        """
        self._build(record=self._panel.recording())
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
        self._finish_movie()

    def reset(self):
        """Drop the run and its viewer, back to where the panel started."""
        self._timer.stop()
        self._finish_movie()
        self.session = None
        self._painter = None
        self._viewer.close()
        self._panel.set_paused(False)
        self._draw_frame()

    def _build(self, record=False):
        """Halt any march, then build the configured run and draw it.

        Whatever the previous run recorded is written out first: the movie
        belongs to the run that drew its frames, not to the one replacing
        it.
        """
        self._timer.stop()
        self._finish_movie()
        params = self._panel.params()
        self._cell_type = params['cell_type']
        self.session = ReflectionSession(**params)
        self._apply_limits()
        self._painter = FieldPainter(self.session.shock.mesh)
        if record:
            self._start_movie()
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
        # A reopened sub-window is bare, so the legend goes back on with
        # the rest of the layers.
        self._viewer.show_bar(self._panel.bar_placement())
        self._draw_frame()

    def _on_viewer(self, open_):
        if open_:
            self._open_viewer()
        else:
            self._viewer.close()

    def _on_viewer_closed(self):
        # Reached from the sub-window's close event; stop the run before Qt
        # frees the viewer, and write out what it recorded, as no frame can
        # follow a viewer that is gone.
        self._timer.stop()
        self._finish_movie()
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

    def _on_bar_placement(self, placement):
        """Move the legend to the edge the field box now names."""
        self._viewer.show_bar(placement)

    def _on_record(self, on):
        """Open or close the recorder of a run already under way; a run
        started later reads the box in :meth:`start`."""
        if self.session is None:
            return
        if on:
            self._start_movie()
        else:
            self._finish_movie()

    def _advance(self):
        if not self._viewer.is_open:
            self._timer.stop()
            return
        if self.session.finished:
            self._timer.stop()
            self._panel.set_paused(True)
            self._finish_movie()
            return
        self._march_frame()

    def _march_frame(self):
        """March one chunk and draw what it left behind."""
        self._apply_limits()
        self.session.advance()
        self._draw_frame()

    def _apply_limits(self):
        """Put the panel's step cap and chunk length onto the session.

        Read every frame, so a change mid-run lands on the next chunk.
        """
        self.session.max_steps = self._panel.max_steps()
        self.session.steps_per_chunk = self._panel.steps_per_frame()

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
            lo, hi = session.analysis.color_range(name, vmin, vmax)
            # The legend reads the same pinned range the colors are mapped
            # over, so the two cannot say different things about a frame.
            self._viewer.show_scale(lo, hi)
            self._viewer.draw_field(self._painter.verts,
                                    self._painter.colors(field, lo, hi),
                                    self._painter.indices)
            if self.movie is not None:
                self._capture_frame()
        self._panel.set_status(session, vmin, vmax)

    def _start_movie(self):
        """Open a recorder over the running session, if none is open."""
        if self.movie is None:
            self.movie = _movie.MovieRecorder()
            self._report(f"recording to {self._movie_path()}")

    def _capture_frame(self):
        """Hold the frame the viewer just drew.

        A viewer with nothing to grab (no graphics surface behind it)
        drops the recording instead of breaking the march.
        """
        try:
            self._viewer.capture(self.movie)
        except RuntimeError as exc:
            self.movie.close()
            self.movie = None
            self._panel.set_recording(False)
            self._report(f"stopped recording: {exc}")

    def _finish_movie(self):
        """Write out what the recorder holds and report where it landed."""
        if self.movie is None:
            return
        movie = self.movie
        self.movie = None
        path = self._movie_path()
        try:
            nframe = movie.write(path)
        except (OSError, ValueError) as exc:
            self._report(f"wrote no movie: {exc}")
        else:
            self._report(f"wrote {nframe} frames to {path}")
        finally:
            movie.close()

    def _movie_path(self):
        """Where the standing run's movie is to land."""
        return self._panel.movie_path(self._cell_type)

    def _report(self, message):
        """Say what became of the movie, in the panel and to the owner.

        The panel keeps the line in view, since the console scrolls away
        under the run's own output and the movie is easy to lose track of.
        """
        self._panel.set_movie_status(message)
        if self.reported is not None:
            self.reported(message)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
