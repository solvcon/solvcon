# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Concrete AI backends over external command-line tools.

This module holds the backends that shell out to an installed AI CLI, plus the
shared plumbing they need: :class:`SubprocessBackend` (PATH discovery and a
cancellable child process) and :func:`parse_tool_calls` (turn a model reply
into Agent Draw command dicts).  Only the Claude CLI backend lives here for
now; the Codex CLI and an HTTP backend are follow-ups that reuse the same base.

The module imports no Qt and makes no network call at import time.  A backend
registers itself only as a class instance in the shared registry, so a caller
lists it and probes :meth:`~solvcon.agent.AgentBackend.available` before use.
"""

import abc
import json
import shutil
import subprocess

from . import _backend


# TODO: solvcon is an application platform for geometry-based computation:
# editing graphics and geometry, visualizing, meshing, and solving
# conservation laws.  These instructions frame the agent around the 2D
# drawing canvas alone; reframe that as one capability among the platform's
# rather than the agent's whole scope.  Related to #966.
_INSTRUCTIONS = (
    "You drive a 2D drawing canvas. Translate the user's request into a JSON "
    "array of drawing commands. Each command is an object with an \"op\" key "
    "naming the operation and the operation's arguments as sibling keys. "
    "Reply with only the JSON array, no prose and no code fences. Use an "
    "empty array when no drawing is needed."
)


def _tool_op_names(tool_surface):
    """The set of op names a tool surface advertises via each tool's ``name``.

    Empty when the surface is empty or names nothing (for example when the
    Agent Draw package is absent), which tells :func:`parse_tool_calls` to skip
    op validation rather than reject everything.
    """
    names = set()
    for tool in tool_surface or []:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            names.add(tool["name"])
    return names


def _strip_code_fences(text):
    """Drop a surrounding triple-backtick fence (bare or tagged) if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _load_json_payload(text):
    """Parse the first JSON array or object out of a model reply, tolerating a
    code fence or surrounding prose.  Return the parsed value, or ``None`` when
    nothing parses."""
    text = _strip_code_fences(text).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def parse_tool_calls(text, tool_surface=None):
    """Turn a model reply into a list of Agent Draw command dicts.

    Accept a JSON array, or a lone object treated as a one-command array.  Each
    command must be an object with an ``op``; when ``tool_surface`` names ops,
    an unknown op is rejected.  Raise :class:`ValueError` on a malformed reply
    so the backend records it as an error rather than running a bad command.
    """
    data = _load_json_payload(text)
    if data is None:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("model reply is not a JSON array of commands")
    valid = _tool_op_names(tool_surface)
    commands = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("command is not an object: %r" % (entry,))
        op = entry.get("op")
        if not isinstance(op, str):
            raise ValueError("command needs a string \"op\": %r" % (entry,))
        if valid and op not in valid:
            raise ValueError("unknown op %r" % (op,))
        commands.append(entry)
    return commands


class SubprocessBackend(_backend.AgentBackend):
    """Base for backends that shell out to an AI CLI found on ``PATH``.

    A subclass sets :attr:`command` to the executable name and implements
    :meth:`_build_argv` (and, for a non-plain-text CLI, :meth:`_parse_output`).
    This base owns everything else: PATH discovery, the :meth:`available`
    check, a cancellable child process, and the whole :meth:`send` flow that
    turns a run into a :class:`BackendResponse`.  A new CLI backend is thus the
    two hooks, never a copied error-handling skeleton.  The running process is
    kept on the instance so a driver thread can :meth:`cancel` a long-running
    call.
    """

    #: The executable name a subclass discovers on ``PATH``.
    command = None

    def __init__(self, timeout=120):
        self._timeout = timeout
        self._proc = None

    @property
    def name(self):
        """Selector label derived from the CLI, e.g. ``claude (cli)``."""
        return "%s (cli)" % self.command

    def executable(self):
        """The resolved path to :attr:`command`, or ``None`` if not on PATH."""
        return shutil.which(self.command) if self.command else None

    def available(self):
        return self.executable() is not None

    @abc.abstractmethod
    def _build_argv(self, exe, prompt):
        """The argv that runs ``exe`` on the composed ``prompt``."""

    def _parse_output(self, stdout):
        """Extract the assistant text from CLI ``stdout``.  The default treats
        stdout as the reply; override for a CLI that wraps it (JSON, etc.)."""
        return (stdout or "").strip()

    @staticmethod
    def _compose_prompt(prompt, scene_context, tool_surface):
        """Fold the instructions, tool surface, scene, and user request into
        one prompt string for a CLI that takes a single prompt argument."""
        tools = json.dumps(tool_surface or [], indent=2)
        return (
            "%s\n\nAvailable operations (tool definitions):\n%s\n\n"
            "Current scene:\n%s\n\nUser request:\n%s"
            % (_INSTRUCTIONS, tools, scene_context, prompt))

    def send(self, prompt, scene_context, tool_surface):
        exe = self.executable()
        if exe is None:
            return _backend.BackendResponse(
                error="%s not found on PATH" % self.command)
        composed = self._compose_prompt(prompt, scene_context, tool_surface)
        try:
            code, out, err = self._communicate(self._build_argv(exe, composed))
        except subprocess.TimeoutExpired:
            return _backend.BackendResponse(
                error="%s timed out" % self.command)
        except OSError as exc:
            return _backend.BackendResponse(
                error="%s failed: %s" % (self.command, exc))
        if code != 0:
            return _backend.BackendResponse(
                error="%s exit %d: %s"
                % (self.command, code, (err or "").strip()))
        text = self._parse_output(out)
        try:
            commands = parse_tool_calls(text, tool_surface)
        except ValueError as exc:
            return _backend.BackendResponse(text=text, error=str(exc))
        return _backend.BackendResponse(text=text, commands=commands)

    def cancel(self):
        """Terminate the in-flight child, if any.  Safe to call from another
        thread while :meth:`send` blocks in :meth:`_communicate`."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def _communicate(self, argv):
        """Run ``argv``, returning ``(returncode, stdout, stderr)``.

        The child is held on ``self._proc`` so :meth:`cancel` can reach it, and
        killed if it outruns the timeout (then the timeout propagates)."""
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self._proc = proc
        try:
            out, err = proc.communicate(timeout=self._timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        finally:
            self._proc = None
        return proc.returncode, out, err


class ClaudeCliBackend(SubprocessBackend):
    """Backend over Anthropic's ``claude`` command-line tool.

    It runs the CLI in print mode with JSON output, folds the tool surface and
    scene context into the prompt, and parses the model's JSON reply into
    Agent Draw commands.  No API key lives here: the CLI owns authentication.
    """

    command = "claude"

    def _build_argv(self, exe, prompt):
        return [exe, "-p", prompt, "--output-format", "json"]

    def _parse_output(self, stdout):
        """Pull the assistant text out of ``claude --output-format json``
        output, falling back to the raw text when it is not that envelope."""
        stdout = (stdout or "").strip()
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout
        if isinstance(payload, dict):
            result = payload.get("result")
            return result if isinstance(result, str) else stdout
        return stdout


_backend.register(ClaudeCliBackend())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
