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

from . import _backend
from . import _command


@dataclasses.dataclass
class TranscriptTurn:
    """One transcript entry: a ``role`` with its text, commands, and results.

    ``results`` holds ``CommandResult`` objects (or any object with an ``ok``
    attribute).  ``failed`` marks a turn that went wrong before any command
    ran, which no result can record.
    """

    role: str
    text: str = ""
    commands: list = dataclasses.field(default_factory=list)
    results: list = dataclasses.field(default_factory=list)
    failed: bool = False


@dataclasses.dataclass
class _OutcomeStub:
    """Failed-result stand-in for when a runner raises instead of returning."""

    op: str
    ok: bool = False
    error: str = None


def _make_executor(world, renderer=None):
    """Build the current ``World`` command executor."""
    from . import draw
    return draw.Executor(world, renderer)


class AgentSession:
    """Bind a ``World``, a backend, and a runner; record a transcript.

    ``runner`` is any object exposing ``run(command) -> result``; it defaults
    to a lazily built command executor for ``world``.  ``backend`` is an
    :class:`~solvcon.agent.AgentBackend` or ``None``.  Delete commands are
    hidden from the backend and rejected unless ``allow_destructive`` is true.
    ``hidden_ops`` names the ops to keep off the tool surface and out of the
    runner; it defaults to :attr:`HIDDEN_OPS`, or to nothing once a
    ``renderer`` is injected, since what that renderer is for is the op
    :attr:`HIDDEN_OPS` names.
    """

    INVENTORY_LIMIT = 40
    HIDDEN_OPS = frozenset({"render_png"})

    def __init__(self, world=None, backend=None, runner=None, renderer=None,
                 allow_destructive=False, hidden_ops=None):
        self.world = world
        self.backend = backend
        self._renderer = renderer
        self._runner = runner
        self._runner_injected = runner is not None
        self.allow_destructive = allow_destructive
        if hidden_ops is None:
            hidden_ops = () if renderer is not None else self.HIDDEN_OPS
        self.hidden_ops = frozenset(hidden_ops)
        self._transcript = []

    @property
    def transcript(self):
        """The recorded turns, oldest first (a copy)."""
        return list(self._transcript)

    def history(self, skip=None):
        """The turns to replay to the backend, oldest first.

        Trailing user turns are left out: the composed request already carries
        the current prompt.  ``skip`` drops the turn at that index, which is
        how a multi-step turn leaves out a prompt no longer trailing.
        """
        end = len(self._transcript)
        while end and self._transcript[end - 1].role == "user":
            end -= 1
        turns = self._transcript[:end]
        if skip is not None and 0 <= skip < end:
            del turns[skip]
        return turns

    @property
    def runner(self):
        """The command runner, built on first use."""
        if self._runner is None:
            self._runner = _make_executor(self.world, self._renderer)
        return self._runner

    def bind_world(self, world):
        """Point the session at ``world`` for later turns, dropping a lazily
        built runner so the next command batch targets the new world.  A runner
        passed to the constructor is kept.

        A switch to a different world marks the transcript.  The shape ids in
        the turns before it name shapes in the world that was open then, so
        replaying them unmarked beside the new world's inventory would invite
        a command aimed at an id that now belongs to something else.
        """
        if world is not self.world and self._transcript:
            self.mark("canvas switched")
        self.world = world
        if not self._runner_injected:
            self._runner = None

    def mark(self, text):
        """Record ``text`` as a marker turn.

        A marker that would follow another is dropped; consecutive markers say
        nothing the first did not.
        """
        role = _backend.HistoryFormatter.MARKER_ROLE
        if self._transcript and self._transcript[-1].role == role:
            return
        self._transcript.append(TranscriptTurn(role=role, text=text))

    def _command_provider(self):
        """What answers ``tool_definitions`` and ``commands_by_category``: the
        bound runner if it carries that surface, else Agent Draw.  Reads
        ``self._runner`` directly to avoid forcing a lazy build."""
        runner = self._runner
        if runner is not None and hasattr(runner, "tool_definitions"):
            return runner
        from . import draw
        return draw

    def tool_surface(self):
        """The command tool definitions to hand the backend, with delete ops
        dropped unless this session allows them and :attr:`hidden_ops` dropped
        always.

        ``render_png`` is hidden by default: a session with no renderer cannot
        run it, and its inline base64 result fits no prompt budget.
        """
        tools = [tool for tool in self._command_provider().tool_definitions()
                 if tool["name"] not in self.hidden_ops]
        if self.allow_destructive:
            return tools
        return [tool for tool in tools if tool["category"] != "delete"]

    def _blocked_ops(self):
        """Op names refused at execution: :attr:`hidden_ops` plus the delete
        category while destructive commands are disabled.

        An op kept off :meth:`tool_surface` is refused here rather than run.
        """
        blocked = set(self.hidden_ops)
        if not self.allow_destructive:
            by_category = self._command_provider().commands_by_category()
            blocked |= set(by_category.get("delete", ()))
        return blocked

    @staticmethod
    def _bbox_text(bbox):
        """One shape's bounding box, or ``[?]`` when it has none to report."""
        try:
            x_min, y_min, x_max, y_max = bbox
            return "[%g, %g, %g, %g]" % (x_min, y_min, x_max, y_max)
        except (TypeError, ValueError):
            return "[?]"

    @classmethod
    def _inventory(cls, shapes):
        """Per-shape id, type, and bounding box; geometry from
        ``describe_state`` is omitted for prompt size.  A crowded world keeps
        the newest :attr:`INVENTORY_LIMIT` shapes and counts the rest."""
        if not shapes:
            return []
        lines = ["shapes (#id type [x_min, y_min, x_max, y_max]):"]
        hidden = len(shapes) - cls.INVENTORY_LIMIT
        if hidden > 0:
            lines.append("  ... %d earlier shapes" % hidden)
        for shape in shapes[-cls.INVENTORY_LIMIT:]:
            lines.append("  #%s %s %s"
                         % (shape.get("id", "?"), shape.get("type", "?"),
                            cls._bbox_text(shape.get("bbox"))))
        return lines

    def scene_context(self, level="basic"):
        """A text summary of the world for the model: the shape count and
        distinct types from ``world.describe_state(...)`` (JSON) followed by
        the per-shape inventory, or a plain count when it cannot be
        described."""
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
        lines = ["world with %d shapes (types: %s)" % (len(shapes), kinds)]
        lines.extend(self._inventory(shapes))
        return "\n".join(lines)

    def _execute(self, commands):
        """Run each command in order and return one result per command.

        An empty batch builds no runner.  A runner that fails to build, or that
        raises on a command, becomes a failed :class:`_OutcomeStub` (one per
        command), so a bad runner or command never aborts the batch and the
        results always line up with the commands.  Commands naming an op this
        session keeps off the tool surface are rejected before reaching the
        runner (see :meth:`_blocked_ops`).  This does not touch the transcript.
        """
        if not commands:
            return []
        blocked = self._blocked_ops()
        gated = [_command.op_of(command) in blocked for command in commands]
        if all(gated):
            return [self._blocked_result(command) for command in commands]
        try:
            runner = self.runner
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            return [_OutcomeStub(_command.op_of(c), error=error)
                    for c in commands]
        results = []
        for command, is_gated in zip(commands, gated):
            if is_gated:
                results.append(self._blocked_result(command))
                continue
            try:
                results.append(runner.run(command))
            except Exception as exc:
                results.append(_OutcomeStub(
                    _command.op_of(command),
                    error="%s: %s" % (type(exc).__name__, exc)))
        return results

    @staticmethod
    def _blocked_result(command):
        op = _command.op_of(command)
        return _OutcomeStub(
            op, error="op %r is disabled for this session" % op)

    def _record_agent(self, text, commands=(), results=(), failed=False):
        """Append and return one agent turn."""
        turn = TranscriptTurn(
            role="agent", text=text, commands=list(commands),
            results=list(results), failed=failed)
        self._transcript.append(turn)
        return turn

    def apply_commands(self, commands):
        """Run each command, recording one agent turn.  An empty batch is a
        no-op that builds no runner and records nothing."""
        commands = list(commands)
        if not commands:
            return []
        results = self._execute(commands)
        self._record_agent("", commands, results)
        return results

    def record_prompt(self, prompt):
        """Record the user's ``prompt`` as a turn.

        Split out so a caller can record the prompt, run the slow backend call
        elsewhere, and finish with :meth:`complete_turn` or :meth:`fail_turn`.
        Returns where the turn landed, which is what :meth:`history` skips.
        """
        self._transcript.append(TranscriptTurn(role="user", text=prompt))
        return len(self._transcript) - 1

    def complete_turn(self, response):
        """Finish a turn from a :class:`~solvcon.agent.BackendResponse`: run
        its commands and record one agent turn carrying the reply, commands,
        and results.  Any backend ``error`` is folded into the reply text."""
        parts = [response.text] if response.text else []
        if response.error:
            parts.append("[error] %s" % response.error)
        commands = list(response.commands)
        return self._record_agent(
            "\n".join(parts), commands, self._execute(commands),
            failed=bool(response.error))

    def fail_turn(self, error):
        """Record a failed agent turn for a transport or error outcome, so the
        turn still lands in the transcript instead of propagating."""
        return self._record_agent("[error] %s" % error, failed=True)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
