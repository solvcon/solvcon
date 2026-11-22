# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Drive one reflection run between the window and the control
column."""

import sys

import numpy as np

from PySide6.QtCore import QTimer

from . import _presenter
from . import _setup

__all__ = [
    'ReflectionController',
]

#: What can go wrong building the solver of a valid setup.
BUILD_ERRORS = (ValueError, RuntimeError, ArithmeticError)


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

    #: The interval of the frame timer, in milliseconds.
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
        self._edit_error = ''
        self._dirty = False
        self._can_reset = True
        # Owned by the window, so no tick outlives the widgets it draws.
        self._timer = QTimer(window)
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
            # The read-out raised again, and is left unreported; the strip
            # still says why the march stopped.
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
        # The message of a build that failed no longer applies to the edit.
        self._build_error = ''
        wrong = []
        for group, summary in (
                (column.freestream, _presenter.freestream_summary),
                (column.numerics, _presenter.numerics_summary)):
            group.show_errors(errors)
            dirty = group.is_dirty()
            invalid = bool(group.invalid_fields())
            group.set_dirty_hint(dirty and not invalid)
            group.set_summary(summary(self.setup, dirty, invalid), dirty)
            self._dirty = self._dirty or dirty
            if invalid:
                wrong.append(group.title)
        self._can_reset = not errors
        self._edit_error = ''
        # The strip names the groups that hold a wrong value, as they may be
        # folded and hide the outline.
        if wrong:
            self._edit_error = (f"Fix the outlined value in "
                                f"{' and '.join(wrong)} to apply the edits.")
        self._show_strip()

    def _on_run_changed(self, max_steps, steps_per_frame):
        """Put the step cap and the steps per frame on the standing run.

        A cap at or under the step the run stands at ends the run there,
        and a higher cap lets a run the cap ended go on, paused.  A failed
        run stays failed.

        :param max_steps: The step cap of the run.
        :param steps_per_frame: The steps marched between two frames.
        """
        session = self.session
        if session is None or 'failed' == self.state:
            return
        session.steps_per_chunk = steps_per_frame
        session.set_max_steps(max_steps)
        if session.finished:
            self._settle()
        elif 'finished' == self.state:
            self.state = 'paused'
        self._show_strip()

    def _on_display(self, display):
        self.column.display.set_summary(_presenter.display_summary(display))
        self._show()

    def _show_strip(self):
        failure = self.error or self._build_error or self._edit_error
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
