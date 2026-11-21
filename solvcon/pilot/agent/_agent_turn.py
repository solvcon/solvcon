# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Agent turns in the pilot.

:class:`AgentTurnRunner` pumps a :class:`~solvcon.agent.Turn` one step at a
time on the active canvas; the functions describe a finished turn's commands,
count what undoing it takes, and write it back as Python.
"""

import math

from PySide6.QtCore import QObject, QThread, Signal

from ...agent import (AgentBackend, AgentSession, BackendRegistry,
                      BackendResponse, StopReason, TranscriptTurn,
                      TransportOutcome, Turn, default_token, op_of)
from ...agent import draw
from ...config import Config
from . import _agent_control

__all__ = [  # noqa: F822
    'AgentTurnRunner',
    'commands_as_python',
    'format_command',
    'undo_steps',
    'undo_turn_steps',
]

#: Ops after which undoing the turn cannot restore the canvas: the window
#: ops change which canvas later draw commands land on, and ``clear`` empties
#: the world's undo stack along with its shapes.
_UNDO_BREAKING_OPS = frozenset(("new_canvas", "activate_window",
                                "close_window", "clear"))


def format_command(command, limit=72):
    """``op key=value ...``, cut to ``limit`` characters."""
    args = " ".join("%s=%s" % (key, _short(value))
                    for key, value in command.items() if key != "op")
    text = ("%s %s" % (op_of(command), args)).strip()
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _short(value):
    if isinstance(value, float):
        return "%g" % value
    return repr(value) if isinstance(value, str) else str(value)


def undo_steps(command, result):
    """How many steps ``command`` left on the world's undo stack.

    ``World`` records shape edits only: a shape's creation (the result names
    the new ``shape_id``), a move, and a removal.  Bare points, segments, and
    beziers record nothing, and a zero move is dropped.
    """
    if not getattr(result, "ok", False):
        return 0
    op = op_of(command)
    if op == "translate_shape":
        return int(bool(command.get("dx") or command.get("dy")))
    if op == "remove_shape":
        return 1
    value = getattr(result, "value", None)
    return int(isinstance(value, dict) and "shape_id" in value
               and op in draw.commands_by_category()["create"])


def undo_turn_steps(pairs):
    """How many undo steps revert a turn's ``(command, result)`` pairs, or
    0 when undoing cannot restore the canvas.

    ``World.begin_operation()`` only merges edits of one shape (a drag), so
    a turn is undone one recorded step at a time.  A turn that moved to
    another canvas drew on a world this one cannot reach, and one that
    cleared the canvas left no history to undo it with.
    """
    if any(op_of(c) in _UNDO_BREAKING_OPS and getattr(r, "ok", False)
           for c, r in pairs):
        return 0
    return sum(undo_steps(c, r) for c, r in pairs)


def _literal(value):
    """``repr`` that reads back, including infinite and NaN floats."""
    if isinstance(value, float) and not math.isfinite(value):
        return "float(%r)" % str(value)
    if isinstance(value, dict):
        return "{%s}" % ", ".join("%r: %s" % (k, _literal(v))
                                  for k, v in value.items())
    if isinstance(value, (list, tuple)):
        items = ", ".join(_literal(v) for v in value)
        if isinstance(value, tuple):
            return "(%s,)" % items if len(value) == 1 else "(%s)" % items
        return "[%s]" % items
    return repr(value)


def commands_as_python(commands, skipped=()):
    """An ``agent_run([...])`` call that replays ``commands``.

    ``skipped`` holds ``(command, reason)`` pairs the turn did not apply;
    they are listed as comments, so the replay never runs what the turn
    refused.
    """
    lines = ["    %s," % _literal(dict(c)) for c in commands]
    lines += ["    # %s: %s" % (reason, _literal(dict(c)))
              for c, reason in skipped]
    return "agent_run([\n%s\n])" % "\n".join(lines) if lines \
        else "agent_run([])"


class AgentTurnRunner(QObject):
    """Run agent turns on the pilot's active canvas without blocking it.

    Composing a request and applying its reply stay on the Qt thread, and
    only the backend call runs on a worker.  The turn applies nothing if the
    user switched, edited, or moved the canvas while the backend was
    thinking.  One turn runs at a time.
    """

    #: The first step's scene and the conversation it replays.
    payload_composed = Signal(str)
    #: A step's record, once its reply was applied or it failed.
    step_applied = Signal(object)
    #: The turn ended: its stop reason, its step records, and the world it
    #: started on.
    turn_finished = Signal(object, object, object)

    #: Backend calls one prompt may spend: a plan, a repair after an error,
    #: and the empty batch that says the work is done.
    TURN_BUDGET = 4

    def __init__(self, mgr, parent=None):
        super().__init__(parent)
        self._mgr = mgr
        #: The :class:`~solvcon.agent.AgentSession` every turn runs in.
        self.session = AgentSession(
            runner=_agent_control.build_control_dispatcher(mgr))
        BackendRegistry.load_settings(Config.instance())
        self._turn = None
        self._worker = None
        self._records = []
        self._world = None
        self._logged = False

    @property
    def busy(self):
        """Whether a turn holds the backend, a stopping one included."""
        return self._turn is not None

    @property
    def worker(self):
        """The thread running the backend call in flight, or None."""
        return self._worker

    def start(self, prompt, backend):
        """Start a turn of ``prompt`` with ``backend``; with no backend the
        turn ends at once."""
        if self.busy:
            raise RuntimeError("an agent turn is already running")
        session = self.session
        session.backend = backend
        widget = self._mgr.currentR2DWidget()
        world = None if widget is None else widget.world
        self._records, self._world, self._logged = [], world, False
        try:
            # Bind before Turn records the prompt, so a canvas-switch marker
            # precedes it.
            session.bind_world(world)
            self._turn = Turn(session, prompt, budget=self.TURN_BUDGET,
                              scene=self._scene, token=self._token)
        except Exception as exc:
            # Shown only: the session never saw the prompt, so a failure
            # recorded there would reach the next request without it.
            record = TranscriptTurn(role="agent", text=_error_text(exc),
                                    failed=True)
            self._records.append(record)
            self.step_applied.emit(record)
            self._turn = None
            self._finish(StopReason.TRANSPORT)
            return
        if backend is None:
            self._turn.stop(StopReason.NO_BACKEND)
            self._finish()
            return
        self._pump()

    def _scene(self, session):
        """Bind the active canvas, which a step may have changed, and
        describe it with its windows and view."""
        widget = self._mgr.currentR2DWidget()
        session.bind_world(None if widget is None else widget.world)
        return _agent_control.pilot_scene_context(
            session.runner, session.scene_context())

    @staticmethod
    def _token(session, scene):
        """What must hold between composing a step and applying its reply.

        The core token covers the world's identity and the scene, which
        names the windows and the view; the world's revision counter adds
        the edits too small for the scene's rounding to show.
        """
        world = session.world
        stamp = None if world is None else world.state_stamp
        return default_token(session, scene) + (stamp,)

    def _pump(self):
        """Send the turn's next step, or finish when it has none."""
        try:
            request = self._turn.next_request()
            if request is not None and not self._logged:
                # Later steps replay the same conversation with the steps
                # between appended, so only the first is shown.
                self._logged = True
                compose = getattr(self.session.backend, "history_section",
                                  AgentBackend.history_section)
                payload = compose(request.prompt, request.scene_context,
                                  request.tool_surface, request.history)
                self.payload_composed.emit("\n".join(
                    part for part in (request.scene_context, payload)
                    if part))
        except Exception as exc:
            self._fail_turn(exc)
            return
        if request is None:
            self._finish()
            return
        self._worker = _StepWorker(self.session.backend, request,
                                   parent=self)
        self._worker.finished.connect(self._step_finished)
        self._worker.start()

    def _step_finished(self):
        """Apply one step's reply and pump the next; the turn drops a reply
        that lands after Stop or a canvas change."""
        worker, self._worker = self._worker, None
        worker.deleteLater()
        try:
            record = self._turn.feed(worker.response)
        except Exception as exc:
            self._fail_turn(exc)
            return
        if record is not None:
            self._records.append(record)
            self.step_applied.emit(record)
        self._pump()

    def _fail_turn(self, exc):
        """End the turn on ``exc``; escaping to the event loop would leave it
        running."""
        record = self.session.fail_turn(_error_text(exc))
        self._records.append(record)
        self.step_applied.emit(record)
        self._turn.stop(StopReason.TRANSPORT)
        self._finish(StopReason.TRANSPORT)

    def _finish(self, reason=None):
        """End the turn as ``reason``, or as its own stop reason."""
        turn, self._turn = self._turn, None
        if reason is None:
            reason = turn.stop_reason
        self.turn_finished.emit(reason, self._records, self._world)

    def stop(self):
        """End the running turn, so nothing it returns is applied, and cancel
        the backend call in flight.  Returns True while that call is still
        out; the turn finishes only when it returns."""
        if self._turn is None or self._turn.done:
            return False
        self._turn.stop()
        cancel = getattr(self.session.backend, "cancel", None)
        if cancel is not None:
            cancel()
        if self._worker is None:
            self._finish()
            return False
        return True

    def wait(self):
        """Block until the backend call in flight returns."""
        if self._worker is not None:
            self._worker.wait()


def _error_text(exc):
    return "%s: %s" % (type(exc).__name__, exc)


class _StepWorker(QThread):
    """Run one backend call off the Qt thread.

    The request was frozen on the Qt thread before the thread started.  The
    reply lands on :attr:`response` and the thread's ``finished`` announces
    it; a backend that raises arrives as a transport outcome.
    """

    def __init__(self, backend, request, parent=None):
        super().__init__(parent)
        self._backend = backend
        self._request = request
        self.response = None

    def run(self):
        try:
            self.response = self._request.send_to(self._backend)
        except Exception as exc:
            self.response = BackendResponse(
                error=_error_text(exc), outcome=TransportOutcome.TRANSPORT)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
