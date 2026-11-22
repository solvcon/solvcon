# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Drive one reflection run between the control panel and the viewer, or
between the window and the control column."""

import sys

import numpy as np

from PySide6.QtCore import QTimer

from ...pilot import visual

from . import _presenter
from . import _setup
from ._driver import check_reflection_solvable
from ._session import ReflectionSession

__all__ = [
    'RunController',
    'ReflectionController',
]

#: What can go wrong building the solver of a valid setup.
BUILD_ERRORS = (ValueError, RuntimeError, ArithmeticError)


class RunController(object):
    """March one run into the viewer on the frame timer.

    The run itself belongs to :class:`~._session.ReflectionSession`, which
    decides how far a chunk marches and when the run is over.  The
    controller only pumps it: one chunk per timer frame, one frame drawn
    after each, so however fast the solver is, the widgets update at the
    frame rate and no march outlives its viewer.

    The plot window is optional, so a run can be driven with only a viewer;
    every call into it is guarded.

    :ivar session: The running :class:`~._session.ReflectionSession`, or
        None until the first :meth:`preview` or :meth:`start` builds one.
    :ivar movie: The open :class:`~solvcon.pilot.visual.MovieRecorder`, or
        None while the run is not being recorded.
    :ivar reported: Owner-supplied callback that says what became of a
        movie, or None to keep it to the panel.
    """

    #: Qt timer interval in milliseconds.
    INTERVAL_MS = 50

    def __init__(self, panel, viewer, lineplots=None):
        self._panel = panel
        self._viewer = viewer
        self._lineplots = lineplots
        self.session = None
        self.movie = None
        self.reported = None
        self._colorizer = None
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
        panel.gauge_changed = self._on_gauge_changed
        panel.record_toggled = self._on_record
        panel.lineplots_toggled = self._on_lineplots
        viewer.closed = self._on_viewer_closed
        if lineplots is not None:
            lineplots.closed = self._on_lineplots_closed

    def preview(self):
        """Open the viewer on the initial state of the configured run.

        The session is built and its step-zero field drawn, and the march
        waits, paused, so the first thing on screen is the state a run
        would start from; Start, Resume, and Step all proceed from it.
        """
        if self._build():
            self._panel.set_paused(True)

    def start(self):
        """(Re)build the run session from the controls and march it into the
        viewer, opening the viewer sub-window first if it was closed.

        A run started with the record box ticked records from its first
        frame, so the movie opens on the state the march starts from.
        """
        if self._build(record=self._panel.recording()):
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
        self._colorizer = None
        self._viewer.close()
        self._panel.set_paused(False)
        self._panel.set_configuration_error("")
        self._draw_frame()

    def _build(self, record=False):
        """Halt any march, then build the configured run and draw it.

        Whatever the previous run recorded is written out first: the movie
        belongs to the run that drew its frames, not to the one replacing
        it.
        """
        params = self._panel.params()
        if not check_reflection_solvable(params['gamma'], params['mach'],
                                         params['angle']):
            self._panel.set_configuration_error(
                "Adjust Mach, gamma, or the deflection angle.")
            return False
        session = ReflectionSession(**params)
        if session.shock.reflected_shock_hits_top():
            self._panel.set_analytic_warning(
                "The reflected shock reaches the top inlet.")
        else:
            self._panel.set_configuration_error("")
        self._timer.stop()
        self._finish_movie()
        self._cell_type = params['cell_type']
        self.session = session

        self._apply_limits()
        self._colorizer = visual.FieldColorizer(self.session.shock.mesh)
        # A new colorizer carries no gauge; give it the standing one.
        self._set_gauge()
        if record:
            self._start_movie()
        self._open_viewer()
        return True

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
            self._viewer.draw_path(shock.shock_path())
        # A reopened sub-window is bare, so the legend goes back on with
        # the rest of the layers.
        self._viewer.show_bar(self._panel.bar_placement())
        self._draw_frame()

    def _on_viewer(self, open_):
        if open_:
            self._open_viewer()
        else:
            self._viewer.close()

    def _on_lineplots(self, open_):
        if self._lineplots is None:
            return
        if open_:
            self._lineplots.open()
            self._panel.set_lineplots_open(True)
            self._draw_lineplots()
        else:
            self._lineplots.close()

    def _on_lineplots_closed(self):
        self._panel.set_lineplots_open(False)

    def _on_gauge_changed(self):
        self._set_gauge()
        self._draw_frame()

    def _draw_lineplots(self):
        """Read the cut out into the plot window, if it is open.

        The plot follows the cut, not the gauge marks over it.
        """
        if self._lineplots is None:
            return
        height = None if self.session is None else self._cut_height()
        self._lineplots.show_run(self.session, self._panel.field(), height)

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
            self._draw_lineplots()
            return
        name = self._panel.field()
        field = session.field.field(name)
        vmin, vmax = float(field.min()), float(field.max())
        if self._viewer.is_open:
            lo, hi = session.analysis.color_range(name, vmin, vmax)
            # The legend reads the same pinned range the colors are mapped
            # over, so the two cannot say different things about a frame.
            self._viewer.show_scale(lo, hi)
            self._viewer.draw_field(self._colorizer.verts,
                                    self._colorizer.colors(field, lo, hi),
                                    self._colorizer.indices)
            if self.movie is not None:
                self._capture_frame()
        self._panel.set_status(session, vmin, vmax)
        self._draw_lineplots()

    def _set_gauge(self):
        """Put the marks of the profile cut into the colorizer, or take them
        out.

        The outline says which cells the reading was interpolated
        across; the bar is the line it was read along.
        """
        if self._colorizer is None:
            return
        if self.session is None:
            self._colorizer.set_gauge(None, None)
            return
        height = self._cut_height()
        cells = (self.session.analysis.profile_cells(height)
                 if self._panel.marking_cells() else None)
        self._colorizer.set_gauge(
            cells, height if self._panel.marking_line() else None)

    def _cut_height(self):
        """Where the standing run's cut runs, as a y coordinate."""
        return self.session.analysis.cut_height(self._panel.cut_fraction())

    def _start_movie(self):
        """Open a recorder over the running session, if none is open."""
        if self.movie is None:
            self.movie = visual.MovieRecorder()
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


class ReflectionController(object):
    """March one reflection run into the window and its control column on
    the frame timer.

    The controller owns the applied setup, the session built of it, the
    run state, and the timer.  Each tick marches the session a frame and
    hands what is to be shown to the presenter, which reads it out into
    the widgets; none of the widgets sees the session.

    The state the strip shows is one of 'ready' (built, at step 0),
    'running', 'paused', 'finished' (the step cap is reached), and 'failed'
    (the march raised or the density lost its finite values).  Edits to the
    free stream and the numerics wait for Reset, which rebuilds the run from
    them at step 0; an edit the setup rejects is outlined in its group and
    keeps Reset from the standing run.  The step cap, the steps per frame,
    and the display apply at once.

    :param window: The :class:`~._window.ReflectionWindow` to drive.
    :param column: The :class:`~._column.ControlColumn` beside it.
    :param setup: The :class:`~._setup.ReflectionSetup` to start from, or
        None for the defaults.
    """

    INTERVAL_MS = 40

    def __init__(self, window, column, setup=None):
        self.window = window
        self.column = column
        #: The :class:`~._presenter.ReflectionPresenter`.
        self.presenter = _presenter.ReflectionPresenter(window, column)
        #: The applied :class:`~._setup.ReflectionSetup`.
        self.setup = _setup.ReflectionSetup() if setup is None else setup
        #: The :class:`~._session.ReflectionSession` built of it.
        self.session = None
        #: The run state.
        self.state = 'ready'
        #: Why the run failed, or an empty string.
        self.error = ''
        #: Owner-supplied callback fired with the window title after a build.
        self.title_changed = None
        self._build_error = ''
        self._dirty = False
        self._can_reset = True
        self._timer = QTimer()
        self._timer.setInterval(self.INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

        strip = window.strip
        strip.run_clicked = self.run_or_pause
        strip.step_clicked = self.step
        strip.reset_clicked = self.reset
        column.freestream.edited = self._on_edited
        column.numerics.edited = self._on_edited
        column.numerics.run_changed = self._on_run_changed
        column.display.display_changed = self._on_display
        if self._build(self.setup):
            self._announce()

    @property
    def display(self):
        """The :class:`~._viewmodel.Display` the column holds."""
        return self.column.display.display()

    def run_or_pause(self):
        """Run, pause, resume, or run again, as the accent action reads."""
        if 'running' == self.state:
            self.pause()
        elif self.state in ('finished', 'failed'):
            if self.reset():
                self.run()
        else:
            self.run()

    def run(self):
        """March on the timer from where the run stands."""
        if self.session is None or self.session.finished or self.error:
            return
        self.state = 'running'
        self._timer.start()
        self._show_strip()

    def pause(self):
        """Stop the timer, holding the run where it is."""
        self._timer.stop()
        if self.session is not None and 'running' == self.state:
            self.state = 'paused'
        self._show_strip()

    def step(self):
        """March one frame from a ready or a paused run."""
        if self.session is None or self.state not in ('ready', 'paused'):
            return
        self.state = 'paused'
        self._tick()

    def reset(self):
        """Apply the edits and build the run again at step 0.

        :return: Whether a run was built; edits that cannot build one leave
            the standing run, and the state it is in, as they are.
        """
        setup, errors = self._edited_setup()
        if errors:
            self._on_edited()
            return False
        if not self._build(setup):
            self._show_strip()
            return False
        self._announce()
        return True

    def shutdown(self):
        """Stop the march, as the window goes away."""
        self._timer.stop()

    def _build(self, setup):
        """Build the session of a setup and take both up.

        The session is built before the standing one is let go, so a setup
        the solver cannot build says why in the strip and leaves the
        standing run marching or paused as it was.

        :param setup: The :class:`~._setup.ReflectionSetup` to build.
        :return: Whether the session was built.
        """
        cap, per_frame = self.column.numerics.run_settings()
        try:
            session = setup.session(max_steps=cap, steps_per_chunk=per_frame)
        except BUILD_ERRORS as exc:
            self._build_error = (f"No run can be built of "
                                 f"{setup.summary()}: {exc}")
            return False
        self._timer.stop()
        self._build_error = ''
        self.setup = setup
        self.session = session
        self.state = 'ready'
        self.error = ''
        self.column.freestream.show_setup(setup)
        self.column.numerics.show_setup(setup)
        shock = session.shock
        self.window.figure.set_mesh(shock.mesh, shock.shock_path())
        return True

    def _announce(self):
        """Read the new run out everywhere after a build."""
        self.window.strip.set_problem(self.setup.summary())
        self._on_edited()
        self._on_display(self.display)
        if self.title_changed is not None:
            self.title_changed(self.setup.title())

    def _tick(self):
        """March one frame and read it out, on the timer or for Step.

        What raises, in the march or in the read-out, stops the march
        instead of raising again on every tick: the timer stops, the state
        reads 'failed', the strip names the error, and the traceback is
        reported once.  A density that is no longer finite fails the run
        alike, as nothing drawn after it would mean anything.
        """
        try:
            if not self.session.finished:
                self.session.advance()
                density = self.session.field.field('density')
                if not np.isfinite(density).all():
                    raise FloatingPointError(
                        "the density is no longer finite")
            self._settle()
            self._show()
        except Exception as exc:
            self._fail(exc)

    def _fail(self, exc):
        """Stop the run on an error and say why in the strip.

        :param exc: The exception the march raised.
        """
        self._timer.stop()
        self.state = 'failed'
        self.error = (f"The march raised {type(exc).__name__} at step "
                      f"{self.session.step}: {exc}")
        sys.excepthook(type(exc), exc, exc.__traceback__)
        try:
            self._show()
        except Exception:
            # The read-out raised, and was reported; the strip still says
            # why the march stopped.
            self._show_strip()

    def _settle(self):
        """Move to 'finished' once the session is over."""
        if self.session is None or not self.session.finished:
            return
        self._timer.stop()
        self.state = 'finished'

    def _edited_setup(self):
        """Build the setup the column now holds.

        Every field in the wrong is named at once: one that holds no
        number, and one whose number the setup rejects, checked against the
        numbers the other fields hold.

        :return: ``(setup, errors)``, errors keyed by setup field in the
            order of :attr:`~._setup.ReflectionSetup.FIELDS`; the setup is
            None when anything is wrong.
        """
        column = self.column
        values, errors = column.freestream.read()
        more_values, more_errors = column.numerics.read()
        values.update(more_values)
        errors.update(more_errors)
        setup = self.setup.replace(**values)
        for key, message in setup.check().items():
            errors.setdefault(key, message)
        if errors:
            return None, {key: errors[key] for key in setup.FIELDS
                          if key in errors}
        return setup, {}

    def _on_edited(self):
        """Mark what waits for Reset, and what keeps Reset from it."""
        column = self.column
        _, errors = self._edited_setup()
        self._dirty = False
        for group, summary in (
                (column.freestream, _presenter.freestream_summary),
                (column.numerics, _presenter.numerics_summary)):
            group.show_errors(errors)
            dirty = group.is_dirty()
            group.set_dirty_hint(dirty and not group.invalid_fields())
            group.set_summary(summary(self.setup, dirty), dirty)
            self._dirty = self._dirty or dirty
        self._can_reset = not errors
        self._show_strip()

    def _on_run_changed(self, max_steps, steps_per_frame):
        """Put the step cap and the steps per frame on the standing run.

        A cap at or under the step the run stands at ends the run there,
        since the session marches no chunk past its cap.

        :param max_steps: The step cap of the run.
        :param steps_per_frame: The steps marched between two frames.
        """
        session = self.session
        if session is None:
            return
        session.max_steps = max_steps
        session.steps_per_chunk = steps_per_frame
        if session.step >= max_steps and not session.finished:
            session.stop()
            self._settle()
        self._show_strip()

    def _on_display(self, display):
        self.column.display.set_summary(_presenter.display_summary(display))
        self._show()

    def _show_strip(self):
        failure = self.error or self._build_error
        self.window.strip.show_state(_presenter.strip_state(
            self.session, self.state, failure, self._dirty, self._can_reset))

    def _show(self):
        """Read the standing run out into every widget."""
        if self.session is None:
            self.presenter.blank()
        else:
            self.presenter.show(self.session, self.display)
        self._show_strip()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
