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


def _make_executor(world, renderer=None):
    """Build an Agent Draw ``Executor`` for ``world``."""
    # TODO: when the agentdraw package ships in-tree, import it at module
    # level (here and in tool_surface) and drop the empty-list fallback.
    from solvcon.agent import draw as agentdraw
    return agentdraw.Executor(world, renderer)


def tool_surface():
    """The Agent Draw tool definitions for a backend, or ``[]`` if the
    ``agentdraw`` package is unavailable."""
    try:
        from solvcon.agent import draw as agentdraw
    except ImportError:
        return []
    return agentdraw.tool_definitions()


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

    def apply_commands(self, commands):
        """Run each command, recording one transcript turn.  An empty batch is
        a no-op that builds no runner; a runner that raises is captured into a
        failed :class:`_OutcomeStub`, so one bad command never aborts a batch.
        """
        if not commands:
            return []
        runner = self.runner
        results = []
        for command in commands:
            try:
                results.append(runner.run(command))
            except Exception as exc:
                op = command.get("op", "?") if isinstance(command, dict) \
                    else "?"
                results.append(_OutcomeStub(
                    op, error="%s: %s" % (type(exc).__name__, exc)))
        self._transcript.append(TranscriptTurn(
            role="agent", commands=list(commands), results=results))
        return results

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
