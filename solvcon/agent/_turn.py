# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The bounded turn loop of the Agent.

:class:`Turn` drives one user request to a stop under a step budget.
:meth:`Turn.next_request` freezes scene and token on the caller's thread;
:meth:`Turn.feed` applies the reply and decides whether to continue.  Splitting
the two lets a GUI keep the slow backend call on a worker while state reads and
commands stay on the owning thread.  :func:`run_turn` loops the pair on one
thread; a budget of one is a single shot.  No Qt is imported.
"""

import enum
import hashlib

from . import _backend


class StopReason(enum.Enum):
    """Why a turn ended.

    ``COMPLETED`` is an empty batch, ``PROSE`` is words instead of commands,
    ``BUDGET`` is steps exhausted with work still proposed, ``STATE`` is a
    token mismatch, and ``STOPPED`` is an external halt between steps.  The
    four transport reasons share names with
    :class:`~solvcon.agent.TransportOutcome`, so a non-``ok`` outcome maps
    straight onto one.
    """

    COMPLETED = "completed"
    PROSE = "prose"
    BUDGET = "budget"
    STATE = "state"
    STOPPED = "stopped"
    NO_BACKEND = "no backend"
    TRANSPORT = _backend.TransportOutcome.TRANSPORT.value
    TIMEOUT = _backend.TransportOutcome.TIMEOUT.value
    CANCELLED = _backend.TransportOutcome.CANCELLED.value


_NO_REQUEST = object()  # a token seam may legitimately return None


def default_scene(session):
    return session.scene_context()


def default_token(session, scene):
    """The headless state token: the bound world's identity and a digest of
    the scene composed against it.

    Identity comes first because content alone cannot tell two blank worlds
    apart.  The world itself rides along so that identity stays valid: an id
    outlives the object it named, and CPython hands the address straight back
    to the next allocation.  The digest sees only what the scene summary says,
    so a change the summary does not show does not trip it.
    """
    digest = hashlib.sha256(scene.encode("utf-8")).hexdigest()
    return (id(session.world), session.world, digest)


class Turn:
    """One user request driven to a stop under a step budget.

    Construction records the prompt and freezes the tool surface, which the
    steps of one turn all share.  ``scene`` and ``token`` are the seams a
    GUI replaces: ``scene(session)`` returns the scene text, and
    ``token(session, scene)`` returns whatever must not change between
    composing a request and applying its commands.
    """

    def __init__(self, session, prompt, budget=2, scene=None, token=None):
        budget = int(budget)
        if budget < 1:
            raise ValueError("budget must be at least 1 step")
        self._session = session
        self._prompt = prompt
        self._budget = budget
        self._scene = scene if scene is not None else default_scene
        self._token = token if token is not None else default_token
        self._steps = 0
        self._stop = None
        self._pending = _NO_REQUEST
        self._tool_surface = session.tool_surface()
        self._index = session.record_prompt(prompt)

    @property
    def done(self):
        return self._stop is not None

    @property
    def stop_reason(self):
        return self._stop

    def next_request(self):
        """The next :class:`~solvcon.agent.TurnRequest`, or ``None`` when done.

        Exhausting a multi-step budget records a marker so the transcript does
        not look like the model went silent; a one-shot budget of one records
        nothing, because a single step is the whole turn.
        """
        if self.done:
            return None
        if self._steps >= self._budget:
            note = ("step budget of %d reached; turn ended with work still "
                    "proposed" % self._budget) if self._budget > 1 else None
            self._finish(StopReason.BUDGET, note)
            return None
        scene = self._scene(self._session)
        self._pending = self._token(self._session, scene)
        self._steps += 1
        return _backend.TurnRequest(
            prompt=self._prompt, scene_context=scene,
            tool_surface=self._tool_surface,
            history=self._session.history(skip=self._index))

    def feed(self, response):
        """Apply one backend reply and decide whether the turn goes on.

        Returns the transcript turn it recorded, or ``None`` when it recorded
        nothing: a state mismatch, or a reply that lands after :meth:`stop`.
        Feeding with no outstanding request raises.
        """
        if self.done:
            return None
        if self._pending is _NO_REQUEST:
            raise RuntimeError("feed() without an outstanding request")
        token, self._pending = self._pending, _NO_REQUEST
        outcome = response.outcome
        if outcome is not _backend.TransportOutcome.OK:
            # The model never answered, so there is nothing to fix and a
            # retry would only spend the budget on the same failure.
            self._stop = StopReason(outcome.value)
            return self._session.fail_turn(
                response.error or "backend %s" % outcome.value)
        if token != self._token(self._session, self._scene(self._session)):
            self._finish(
                StopReason.STATE,
                "canvas state changed mid-turn; %d commands dropped"
                % len(response.commands))
            return None
        turn = self._session.complete_turn(response)
        if response.status is _backend.ParseStatus.EMPTY:
            self._stop = StopReason.COMPLETED
        elif response.status is _backend.ParseStatus.PROSE:
            self._stop = StopReason.PROSE
        # MALFORMED and COMMANDS both go on: the model is shown the parse
        # error or the command results and gets another step to act on them.
        return turn

    def stop(self, reason=StopReason.STOPPED):
        """End the turn from outside, between steps.

        An outstanding request is forgotten, so a reply that lands after it is
        dropped rather than applied.
        """
        if self.done:
            return
        self._pending = _NO_REQUEST
        self._finish(reason, None)

    def _finish(self, reason, note):
        self._stop = reason
        if note:
            self._session.mark(note)


def run_turn(session, prompt, budget=2, scene=None, token=None):
    """Drive one request on ``session`` to a stop and return the last turn it
    recorded.

    The backend runs on this thread.  A backend that raises is folded into a
    transport outcome, so the loop ends the way a backend reporting one does
    rather than propagating to a headless caller.  With no backend the prompt
    is recorded and ``None`` comes back.
    """
    turn = Turn(session, prompt, budget=budget, scene=scene, token=token)
    if session.backend is None:
        turn.stop(StopReason.NO_BACKEND)
        return None
    recorded = None
    while True:
        request = turn.next_request()
        if request is None:
            return recorded
        try:
            response = request.send_to(session.backend)
        except Exception as exc:
            response = _backend.BackendResponse(
                error="%s: %s" % (type(exc).__name__, exc),
                outcome=_backend.TransportOutcome.TRANSPORT)
        recorded = turn.feed(response) or recorded

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
