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

from . import _observe


@dataclasses.dataclass
class TranscriptTurn:
    """One transcript entry: a ``role`` with its text, commands, and results.

    ``results`` holds ``CommandResult`` objects (or any object with an ``ok``
    attribute).
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
    """Build the current ``World`` command executor."""
    from . import draw
    return draw.Executor(world, renderer)


class AgentSession:
    """Bind a ``World``, a backend, and a runner; record a transcript.

    ``runner`` is any object exposing ``run(command) -> result``; it defaults
    to a lazily built command executor for ``world``.  ``backend`` is an
    :class:`~solvcon.agent.AgentBackend` or ``None``.  Delete commands are
    hidden from the backend and rejected unless ``allow_destructive`` is true.
    """

    def __init__(self, world=None, backend=None, runner=None, renderer=None,
                 allow_destructive=False):
        self.world = world
        self.backend = backend
        self._renderer = renderer
        self._runner = runner
        self._runner_injected = runner is not None
        self.allow_destructive = allow_destructive
        self._transcript = []
        self._artifacts = None

    @property
    def transcript(self):
        """The recorded turns, oldest first (a copy)."""
        return list(self._transcript)

    @property
    def artifacts(self):
        """The session artifact store, built on first use so a session that
        never renders never creates a temp directory."""
        if self._artifacts is None:
            from ._artifact import ArtifactStore
            self._artifacts = ArtifactStore()
        return self._artifacts

    def close(self):
        """Release session resources: remove the artifact store directory.
        Safe to call more than once."""
        if self._artifacts is not None:
            self._artifacts.close()
            self._artifacts = None

    @property
    def runner(self):
        """The command runner, built on first use."""
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
        dropped unless this session allows them."""
        tools = self._command_provider().tool_definitions()
        if self.allow_destructive:
            return tools
        return [tool for tool in tools if tool["category"] != "delete"]

    def _gated_ops(self):
        """Op names blocked while destructive commands are disabled: the
        delete category across the provider's families."""
        by_category = self._command_provider().commands_by_category()
        return set(by_category.get("delete", ()))

    def scene_context(self, level="basic"):
        """A bounded scene snapshot for the model from
        ``world.describe_state(...)`` (JSON), formatted per the composition
        rules, or a plain count when the world cannot be described."""
        world = self.world
        if world is None:
            return "no active world"
        try:
            state = json.loads(world.describe_state(level=level))
        except Exception:
            return "world with %s shapes" % getattr(world, "nshape", "?")
        return _observe.format_scene(state)

    @staticmethod
    def _op_of(command):
        """The command's declared ``op``, or ``"?"`` when it names none."""
        if not isinstance(command, dict):
            return "?"
        op = command.get("op")
        return op if isinstance(op, str) else "?"

    def _execute(self, commands):
        """Run each command in order and return one result per command.

        An empty batch builds no runner.  A runner that fails to build, or that
        raises on a command, becomes a failed :class:`_OutcomeStub` (one per
        command), so a bad runner or command never aborts the batch and the
        results always line up with the commands.  Delete commands are rejected
        before reaching the runner unless this session allows them.  This does
        not touch the transcript.
        """
        if not commands:
            return []
        blocked = set() if self.allow_destructive else self._gated_ops()
        allowed = [command for command in commands
                   if self._op_of(command) not in blocked]
        if not allowed:
            return [_OutcomeStub(
                self._op_of(command),
                error="destructive command %r is disabled for this session"
                % self._op_of(command)) for command in commands]
        try:
            runner = self.runner
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            return [_OutcomeStub(self._op_of(c), error=error)
                    for c in commands]
        results = []
        for command in commands:
            op = self._op_of(command)
            if op in blocked:
                results.append(_OutcomeStub(
                    op, error="destructive command %r is disabled for this "
                    "session" % op))
                continue
            try:
                results.append(runner.run(command))
            except Exception as exc:
                results.append(_OutcomeStub(
                    self._op_of(command),
                    error="%s: %s" % (type(exc).__name__, exc)))
        return self._offload_results(results)

    def _offload_results(self, results):
        """Move any base64 blob in a result value into the artifact store,
        leaving a path reference so the transcript and prompt never carry the
        bytes.  A blob that cannot be stored (quota, bad base64, a filesystem
        error) leaves an ``error`` in its reference rather than a path; the
        command outcome is unchanged, since storing the artifact is the
        harness's job, not the command's.  Builds no store until a blob
        actually appears."""
        for result in results:
            value = getattr(result, "value", None)
            if value is None or not _observe.has_blob(value):
                continue
            result.value = _observe.offload_blobs(value, self.artifacts)
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
        commands = list(commands)
        if not commands:
            return []
        results = self._execute(commands)
        self._record_agent("", commands, results)
        return results

    def record_prompt(self, prompt):
        """Record the user's ``prompt`` as a turn.

        Split out of :meth:`run_turn` so a GUI can record the prompt, run the
        slow backend call off the main thread, and finish the turn later with
        :meth:`complete_turn` or :meth:`fail_turn`.
        """
        self._transcript.append(TranscriptTurn(role="user", text=prompt))

    def complete_turn(self, response):
        """Finish a turn from a :class:`~solvcon.agent.BackendResponse`: run
        its commands and record one agent turn carrying the reply, commands,
        and results.  Any backend ``error`` is folded into the reply text."""
        parts = [response.text] if response.text else []
        if response.error:
            parts.append("[error] %s" % response.error)
        commands = list(response.commands)
        return self._record_agent(
            "\n".join(parts), commands, self._execute(commands))

    def fail_turn(self, error):
        """Record a failed agent turn for a backend that raised, so the turn
        still lands in the transcript instead of propagating."""
        return self._record_agent("[error] %s" % error)

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
        self.record_prompt(prompt)
        if self.backend is None:
            return None
        try:
            response = self.backend.send(
                prompt, self.scene_context(), self.tool_surface())
        except Exception as exc:
            return self.fail_turn("%s: %s" % (type(exc).__name__, exc))
        return self.complete_turn(response)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
