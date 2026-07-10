# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Headless core of the Agent.

:class:`AgentSession` binds the active ``World``, an optional backend, and a
command *runner* (the Agent Draw ``Executor`` by default), and records every
applied command into a transcript a caller can render.  No Qt is imported.
"""

import json
import dataclasses


@dataclasses.dataclass
class TranscriptTurn:
    """One transcript entry: a ``role`` with its text, commands, and results.

    ``results`` holds Agent Draw ``CommandResult`` objects (or any object with
    an ``ok`` attribute).
    """

    role: str
    text: str = ""
    commands: list = dataclasses.field(default_factory=list)
    results: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _OutcomeStub:
    """Failed-result stand-in for when a runner raises instead of returning."""

    op: str
    ok: bool = False
    error: str = None


_draw_module = None


def _resolve_draw():
    """The module supplying the command surface and runner: Agent Draw when it
    is importable, else the built-in :mod:`_draw` bridge (#965).  Both expose
    the same handles (``Executor``, ``tool_definitions``), so callers need not
    know which won.  Resolved once per process."""
    # TODO: when the agentdraw package ships in-tree, import it at module level
    # and drop the _draw fallback and this selector.
    global _draw_module
    if _draw_module is None:
        try:
            from solvcon.pilot import agentdraw
            _draw_module = agentdraw
        except ImportError:
            from . import _draw
            _draw_module = _draw
    return _draw_module


def _make_executor(world, renderer=None):
    """Build the command runner for ``world`` from the resolved draw module."""
    return _resolve_draw().Executor(world, renderer)


def tool_surface():
    """The tool definitions a backend proposes against, from the resolved draw
    module."""
    return _resolve_draw().tool_definitions()


class AgentSession:
    """Bind a ``World``, a backend, and a runner; record a transcript.

    ``runner`` is any object exposing ``run(command) -> result``; it defaults
    to a lazily built Agent Draw ``Executor(world, renderer)``.  ``backend`` is
    an :class:`~solvcon.agent.AgentBackend` or ``None``.
    """

    def __init__(self, world=None, backend=None, runner=None, renderer=None):
        self.world = world
        self.backend = backend
        self._renderer = renderer
        self._runner = runner
        self._runner_injected = runner is not None
        self._transcript = []

    @property
    def transcript(self):
        """The recorded turns, oldest first (a copy)."""
        return list(self._transcript)

    @property
    def runner(self):
        """The command runner, built from ``agentdraw`` on first use."""
        if self._runner is None:
            self._runner = _make_executor(self.world, self._renderer)
        return self._runner

    def bind_world(self, world):
        """Point the session at ``world`` for later turns, dropping a lazily
        built runner so the next command batch targets the new world.  A runner
        passed to the constructor is kept."""
        self.world = world
        if not self._runner_injected:
            self._runner = None

    def tool_surface(self):
        """The Agent Draw tool definitions to hand the backend."""
        return tool_surface()

    def scene_context(self, level="basic"):
        """A short text summary of the world for the model: the shape count
        and distinct types from ``world.describe_state(...)`` (JSON), or a
        plain count when it cannot be described."""
        world = self.world
        if world is None:
            return "no active world"
        try:
            state = json.loads(world.describe_state(level=level))
        except Exception:
            return "world with %s shapes" % getattr(world, "nshape", "?")
        shapes = state.get("shapes", [])
        types = sorted({s["type"] for s in shapes if "type" in s})
        kinds = ", ".join(types) if types else "none"
        return "world with %d shapes (types: %s)" % (len(shapes), kinds)

    @staticmethod
    def _op_of(command):
        """The command's declared ``op``, or ``"?"`` when it names none."""
        return command.get("op", "?") if isinstance(command, dict) else "?"

    def _execute(self, commands):
        """Run each command in order and return one result per command.

        An empty batch builds no runner.  A runner that fails to build, or that
        raises on a command, becomes a failed :class:`_OutcomeStub` (one per
        command), so a bad runner or command never aborts the batch and the
        results always line up with the commands.  This does not touch the
        transcript.
        """
        if not commands:
            return []
        try:
            runner = self.runner
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            return [_OutcomeStub(self._op_of(c), error=error)
                    for c in commands]
        results = []
        for command in commands:
            try:
                results.append(runner.run(command))
            except Exception as exc:
                results.append(_OutcomeStub(
                    self._op_of(command),
                    error="%s: %s" % (type(exc).__name__, exc)))
        return results

    def _record_agent(self, text, commands=(), results=()):
        """Append and return one agent turn."""
        turn = TranscriptTurn(
            role="agent", text=text,
            commands=list(commands), results=list(results))
        self._transcript.append(turn)
        return turn

    def apply_commands(self, commands):
        """Run each command, recording one agent turn.  An empty batch is a
        no-op that builds no runner and records nothing."""
        if not commands:
            return []
        results = self._execute(commands)
        self._record_agent("", commands, results)
        return results

    def run_turn(self, prompt):
        """Drive one request end to end for a single-turn chat.

        Record the user's ``prompt``, ask the backend for commands against the
        current scene and tool surface, run them, and record one agent turn
        carrying the reply text, the commands, and their results.  With no
        backend, record only the user turn and return ``None``.  A backend that
        raises is recorded as a failed agent turn rather than propagated, so a
        headless caller always gets a turn back.  No prior turns are replayed:
        multi-turn chat history is a later addition.
        """
        self._transcript.append(TranscriptTurn(role="user", text=prompt))
        if self.backend is None:
            return None
        try:
            response = self.backend.send(
                prompt, self.scene_context(), self.tool_surface())
        except Exception as exc:
            return self._record_agent(
                "[error] %s: %s" % (type(exc).__name__, exc))
        parts = [response.text] if response.text else []
        if response.error:
            parts.append("[error] %s" % response.error)
        return self._record_agent(
            "\n".join(parts), response.commands,
            self._execute(response.commands))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
