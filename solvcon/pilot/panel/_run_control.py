# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Drive the run of an app between its window and its control column."""

import sys

from PySide6.QtCore import QTimer

__all__ = [
    'RunController',
]


class RunController(object):
    """March one run of an app into its :class:`~._run_strip.RunWindow` and
    its control column on the frame timer.

    The controller owns the applied setup, the session built of it, the
    run state, and the timer.  Each tick marches the session a chunk and
    hands it to :meth:`show_session`.

    The state is one of ``'ready'`` (built, at step 0), ``'running'``,
    ``'paused'``, ``'finished'`` (the step cap is reached), and
    ``'failed'`` (the march raised, or :meth:`check_march` found the run
    gone wrong).  Edits to the setup groups wait for Reset, which rebuilds
    the run from them at step 0; an edit the setup rejects keeps Reset off.
    The step cap, the steps per frame, and the display apply at once.

    An app subclass names its groups and its read-out in the hooks:
    :meth:`setup_groups`, :meth:`take_session`, :meth:`check_march`,
    :meth:`show_session`, :meth:`blank`, :meth:`strip_state`, and
    :meth:`display_summary`.  The column holds a ``display`` group with
    ``display()``, ``set_summary()``, and a ``display_changed`` callback,
    and a ``numerics`` group with ``run_settings()`` and a ``run_changed``
    callback.

    :param window: The :class:`~._run_strip.RunWindow` to drive.
    :param column: The control column beside it.
    :param setup: The setup to start from, a frozen dataclass with
        ``replace()``, ``check()``, ``session()``, ``summary()``,
        ``title()``, and ``FIELDS``.
    """

    #: The interval of the frame timer, in milliseconds.
    INTERVAL_MS = 40
    #: What can go wrong building the solver of a valid setup.
    BUILD_ERRORS = (ValueError, RuntimeError, ArithmeticError)

    def __init__(self, window, column, setup):
        self.window = window
        self.column = column
        #: The applied setup.
        self.setup = setup
        #: The session built of it.
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
        for group, _ in self.setup_groups():
            group.edited = self._on_edited
        column.numerics.run_changed = self._on_run_changed
        column.display.display_changed = self._on_display
        if self._build(self.setup):
            self._announce()

    def setup_groups(self):
        """Return ``(group, summary)`` of each group of the column that
        edits the setup, in the order of ``FIELDS``; ``summary(setup,
        dirty, invalid)`` returns the text of the group head."""
        return ()

    def take_session(self, session):
        """Show what a new session holds before its first frame, such as
        its mesh.

        :param session: The session just built.
        """

    def check_march(self, session):
        """Raise for a run that went wrong in a way the march does not
        raise for, such as a field no longer finite.

        :param session: The session just marched.
        """

    def show_session(self, session, display):
        """Read a session out into every widget but the strip.

        :param session: The session to show.
        :param display: The display the column holds.
        """
        raise NotImplementedError

    def blank(self):
        """Clear every widget but the strip, for no run to show."""
        raise NotImplementedError

    def strip_state(self, session, state, failure, dirty, can_reset):
        """Return the :class:`~._readout.StripState` of a session.

        :param session: The session, or None.
        :param state: The run state.
        :param failure: What made the run fail, or empty.
        :param dirty: Whether the groups hold edits the run has not taken.
        :param can_reset: Whether the edits can build a run.
        """
        raise NotImplementedError

    def display_summary(self, display):
        """Return the text of the head of the display group.

        :param display: The display the column holds.
        """
        return ""

    @property
    def display(self):
        """The display the column holds."""
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

        :param setup: The setup to build.
        :return: Whether the session was built.
        """
        # Built before the standing session is let go, so a setup the
        # solver cannot build leaves the standing run as it was.
        cap, per_frame = self.column.numerics.run_settings()
        try:
            session = setup.session(max_steps=cap, steps_per_chunk=per_frame)
        except self.BUILD_ERRORS as exc:
            self._build_error = (f"No run can be built of "
                                 f"{setup.summary()}: {exc}.  Change the "
                                 f"inputs and press Reset.")
            return False
        self._timer.stop()
        self._build_error = ''
        self.setup = setup
        self.session = session
        self.state = 'ready'
        self.error = ''
        for group, _ in self.setup_groups():
            group.show_setup(setup)
        self.take_session(session)
        return True

    def _announce(self):
        """Read the new run out everywhere after a build."""
        self.window.strip.set_problem(self.setup.summary())
        self._on_edited()
        self._on_display(self.display)
        if self.title_changed is not None:
            self.title_changed(self.setup.title())

    def _tick(self):
        """March one frame and read it out, on the timer or for Step."""
        # What raises, in the march or in the read-out, fails the run once
        # rather than raising again on every tick.
        try:
            if not self.session.finished:
                self.session.advance()
                self.check_march(self.session)
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
        self.error = (f"The run failed at step {self.session.step}: {exc} "
                      f"({type(exc).__name__}).  Run again or Reset "
                      f"rebuilds it from the column.")
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

        :return: ``(setup, errors)``, errors keyed by setup field in the
            order of ``FIELDS``; the setup is None when anything is wrong.
        """
        values, errors = {}, {}
        for group, _ in self.setup_groups():
            more_values, more_errors = group.read()
            values.update(more_values)
            errors.update(more_errors)
        # Every wrong field is named at once: one that holds no number, and
        # one whose number the setup rejects against the others.
        setup = self.setup.replace(**values)
        for key, message in setup.check().items():
            errors.setdefault(key, message)
        if errors:
            return None, {key: errors[key] for key in setup.FIELDS
                          if key in errors}
        return setup, {}

    def _on_edited(self):
        """Mark what waits for Reset, and what keeps Reset from it."""
        _, errors = self._edited_setup()
        self._dirty = False
        # The message of a build that failed no longer applies to the edit.
        self._build_error = ''
        wrong = []
        for group, summary in self.setup_groups():
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
        # The whole run, since a figure may scale by the time of the cap.
        self._show()

    def _on_display(self, display):
        self.column.display.set_summary(self.display_summary(display))
        self._show()

    def _show_strip(self):
        failure = self.error or self._build_error or self._edit_error
        self.window.strip.show_state(self.strip_state(
            self.session, self.state, failure, self._dirty, self._can_reset))

    def _show(self):
        """Read the standing run out into every widget."""
        if self.session is None:
            self.blank()
        else:
            self.show_session(self.session, self.display)
        self._show_strip()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
