# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Concrete AI backends over external CLIs and HTTP APIs.

This module holds the backends that talk to an installed AI CLI or an
OpenAI-compatible HTTP server, plus the shared plumbing they need:
:class:`SubprocessBackend` (PATH discovery and a cancellable child process),
:class:`OpenAIHttpBackend` (stdlib ``http.client``, no SDK), and
:func:`parse_tool_calls` (turn a model reply into Agent Draw command dicts).
The Codex CLI backend is a follow-up that reuses :class:`SubprocessBackend`.

The module imports no Qt and makes no network call at import time.  A backend
registers itself only as a class instance in the shared registry, so a caller
lists it and probes :meth:`~solvcon.agent.AgentBackend.available` before use.
"""

import abc
import http.client
import json
import os
import shutil
import subprocess
import tempfile
import urllib.parse

from . import _backend


def _tool_op_names(tool_surface):
    """The set of op names a tool surface advertises via each tool's ``name``.

    Empty when the surface is empty or names nothing, which tells
    :func:`parse_tool_calls` to skip op validation rather than reject
    everything.
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
    code fence or surrounding prose.

    Return the parsed value, or ``None`` when the reply has no JSON-looking
    span (plain prose).  Raise :class:`ValueError` when a ``[``/``{`` span is
    present but does not parse, so a truncated or invalid command batch is not
    mistaken for an empty one.
    """
    text = _strip_code_fences(text).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    saw_span = False
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start == -1 or end <= start:
            continue
        saw_span = True
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            continue
    if saw_span or text[0] in "[{":
        raise ValueError("model reply has malformed JSON")
    return None


def parse_tool_calls(text, tool_surface=None):
    """Turn a model reply into a list of command dicts.

    Accept a JSON array, or a lone object treated as a one-command array.  Each
    command must be an object with an ``op``; when ``tool_surface`` names ops,
    an unknown op is rejected.  Raise :class:`ValueError` on a malformed reply
    (including invalid JSON that looks like a command batch) so the backend
    records it as an error rather than running a bad command.  Plain prose with
    no JSON yields an empty list.
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

    #: The environment variables to pass through to the agent CLI.
    env_passthrough = (
        "HOME", "USER", "LOGNAME", "PATH", "TMPDIR",
        "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR")

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

        env = {name: os.environ[name]
               for name in self.env_passthrough if name in os.environ}
        workdir = tempfile.mkdtemp(prefix="solvcon-agent-")
        try:
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=workdir, env=env)
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
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class ClaudeCliBackend(SubprocessBackend):
    """Backend over Anthropic's ``claude`` command-line tool.

    It runs the CLI in print mode with JSON output, folds the tool surface and
    scene context into the prompt, and parses the model's JSON reply into
    commands.  No API key lives here: the CLI owns authentication.
    """

    command = "claude"

    def _build_argv(self, exe, prompt):
        # TODO: provide more permission and config to the CLI sandbox later.
        return [
            exe, "-p", prompt, "--output-format", "json",
            "--tools", "",
            "--permission-mode", "dontAsk",  # no interactive prompts
            "--setting-sources", "",  # no config files
            "--strict-mcp-config",  # no mcp config files
            "--disable-slash-commands",  # no interactive slash commands
            "--no-session-persistence",  # no session files
        ]

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


class OpenAIHttpBackend(_backend.AgentBackend):
    """Backend over an OpenAI-compatible Chat Completions HTTP API.

    Uses only the stdlib (``http.client`` and ``urllib.parse``); no vendor
    SDK.  Point ``base_url`` at OpenAI, Ollama's ``/v1`` endpoint, or any
    compatible server.  Defaults and the optional API key come from the
    constructor or the ``SOLVCON_OPENAI_BASE_URL``, ``SOLVCON_OPENAI_MODEL``,
    and ``SOLVCON_OPENAI_API_KEY`` environment variables.  The in-flight
    connection is kept on the instance so a driver thread can :meth:`cancel`.
    """

    # Local Ollama's OpenAI-compatible root; override for a remote provider.
    _DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
    _DEFAULT_MODEL = "qwen2.5vl:7b"

    def __init__(self, base_url=None, model=None, api_key=None, timeout=120):
        self._base_url = base_url if base_url is not None else self._env_or(
            "SOLVCON_OPENAI_BASE_URL", self._DEFAULT_BASE_URL)
        self._model = model if model is not None else self._env_or(
            "SOLVCON_OPENAI_MODEL", self._DEFAULT_MODEL)
        self._api_key = api_key if api_key is not None else self._env_or(
            "SOLVCON_OPENAI_API_KEY", "")
        self._timeout = timeout
        self._conn = None

    @staticmethod
    def _env_or(name, default):
        """``os.environ[name]`` when set and non-empty, else ``default``."""
        value = os.environ.get(name)
        return value if value else default

    @property
    def name(self):
        return "openai (http)"

    @property
    def base_url(self):
        """API root including the ``/v1`` suffix, with no trailing slash."""
        return (self._base_url or "").rstrip("/")

    @property
    def model(self):
        return self._model

    def available(self):
        """True when both a base URL and a model name are configured."""
        return bool(self.base_url) and bool(self._model)

    def send(self, prompt, scene_context, tool_surface):
        if not self.available():
            return _backend.BackendResponse(
                error="openai http backend needs base_url and model")
        composed = self._compose_prompt(prompt, scene_context, tool_surface)
        body = {
            "model": self._model,
            "stream": False,
            "messages": [{"role": "user", "content": composed}],
        }
        try:
            status, raw = self._post_chat(body)
        except TimeoutError:
            return _backend.BackendResponse(
                error="openai http timed out")
        except (OSError, http.client.HTTPException) as exc:
            return _backend.BackendResponse(
                error="openai http failed: %s" % exc)
        if status != 200:
            detail = (raw or b"").decode("utf-8", errors="replace").strip()
            return _backend.BackendResponse(
                error="openai http status %d: %s" % (status, detail))
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return _backend.BackendResponse(
                error="openai http bad JSON: %s" % exc)
        text = self._parse_chat_payload(payload)
        if text is None:
            return _backend.BackendResponse(
                error="openai http response missing assistant text")
        try:
            commands = parse_tool_calls(text, tool_surface)
        except ValueError as exc:
            return _backend.BackendResponse(text=text, error=str(exc))
        return _backend.BackendResponse(text=text, commands=commands)

    def cancel(self):
        """Close the in-flight HTTP connection, if any.  Safe to call from
        another thread while :meth:`send` blocks in :meth:`_post_chat`."""
        conn = self._conn
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    @classmethod
    def _parse_chat_payload(cls, payload):
        """Assistant text from a Chat Completions JSON body, or ``None``."""
        if not isinstance(payload, dict):
            return None
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        return cls._message_text(first.get("message") or {})

    @staticmethod
    def _message_text(message):
        """Assistant text from an OpenAI-style ``message`` object.

        Accept a plain string ``content``, or a list of content parts (the
        multimodal shape) by joining the text pieces.
        """
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    def _post_chat(self, body):
        """POST ``body`` to ``/chat/completions``; return ``(status, raw)``.

        Builds an ``http.client`` connection from :attr:`base_url`, holds it
        on ``self._conn`` for :meth:`cancel`, and always clears that slot.
        """
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise OSError("invalid base_url: %s" % self.base_url)
        path = parsed.path.rstrip("/") + "/chat/completions"
        if parsed.query:
            path = "%s?%s" % (path, parsed.query)
        host = parsed.hostname
        if not host:
            raise OSError("invalid base_url host: %s" % self.base_url)
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = "Bearer %s" % self._api_key
        payload = json.dumps(body).encode("utf-8")
        if parsed.scheme == "https":
            conn = http.client.HTTPSConnection(
                host, port, timeout=self._timeout)
        else:
            conn = http.client.HTTPConnection(
                host, port, timeout=self._timeout)
        self._conn = conn
        try:
            conn.request("POST", path, body=payload, headers=headers)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            try:
                conn.close()
            except OSError:
                pass
            self._conn = None


_backend.register(OpenAIHttpBackend())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
