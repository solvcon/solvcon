# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Concrete AI backends over external CLIs and HTTP APIs.

This module holds the backends that talk to an installed AI CLI or an
OpenAI-compatible HTTP server, plus the shared plumbing they need:
:class:`SubprocessBackend` (PATH discovery and a cancellable child process),
:class:`OpenAIHttpBackend` (stdlib ``http.client``, no SDK), and
:class:`ToolCallParser` (turn a model reply into Agent Draw command dicts).
The Claude Code and Codex CLI backends reuse :class:`SubprocessBackend`.

The module imports no Qt and makes no network call at import time.  A backend
registers itself only as a class instance in the shared registry, so a caller
lists it and probes :meth:`~solvcon.agent.AgentBackend.available` before use.
"""

import abc
import dataclasses
import http.client
import json
import os
import shutil
import subprocess
import tempfile
import urllib.parse

from . import _backend


@dataclasses.dataclass
class ParsedReply:
    """Commands, parse ``status``, and ``error`` from one model reply."""

    commands: list = dataclasses.field(default_factory=list)
    status: _backend.ParseStatus = _backend.ParseStatus.EMPTY
    error: str = None

    def response(self, text):
        return _backend.BackendResponse(
            text=text, commands=self.commands, error=self.error,
            status=self.status)


class ToolCallParser:
    """Turns a model reply into the command dicts a session runs.

    :attr:`NO_JSON` is what says a reply carried no JSON at all, which a
    reply of the literal ``null`` must not be mistaken for: the first is the
    model talking, the second is the model answering with the wrong shape.

    Op names are not checked here.  An op the tool surface does not advertise
    is a command the runner rejects with its own error, which the model can
    see and fix; rejecting it while parsing would throw away the whole batch
    over one bad entry.
    """

    NO_JSON = object()

    #: The tag that closes a reasoning model's chain of thought.  A server
    #: such as vLLM leaves the thinking in the message content.
    REASONING_END = "</think>"

    #: Every character a JSON value can start with.
    VALUE_START = "{[\"-0123456789tfn"

    @classmethod
    def strip_reasoning(cls, text):
        """Return the answer after a reasoning model's chain of thought.

        The thinking argues with itself, so an array it tries out along the
        way is not the reply and must not reach the runner.  Take what
        follows the last :attr:`REASONING_END` outside a JSON string.  A
        model cut off mid-thought leaves nothing after that tag, so the reply
        yields no JSON and reads as prose.  Prose is the honest reading: an
        empty batch would say the request is already done.

        A command carries arbitrary text, a ``log`` message above all, so the
        tag also turns up inside the payload.  An odd number of quotes before
        it puts it in a string, where it is message text and cutting there
        would throw the whole batch away.
        """
        marker = text.rfind(cls.REASONING_END)
        while marker != -1:
            if text.count('"', 0, marker) % 2 == 0:
                return text[marker + len(cls.REASONING_END):]
            marker = text.rfind(cls.REASONING_END, 0, marker)
        return text

    @classmethod
    def strip_code_fences(cls, text):
        """Drop a surrounding triple-backtick fence (bare or tagged) if
        present."""
        stripped = text.strip()
        if not stripped.startswith("```"):
            return text
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)

    @classmethod
    def opens_value(cls, text, index):
        """Whether a JSON value can start at ``index``, skipping space.

        The check is what keeps the cost of a bracket-heavy reply in hand.
        Prose brackets such as ``[ref]`` fail it outright, and each decode
        they would otherwise cost builds an exception that rescans the whole
        reply to find its line and column.
        """
        while index < len(text) and text[index].isspace():
            index += 1
        return index < len(text) and text[index] in cls.VALUE_START

    @classmethod
    def load_json_payload(cls, text):
        """Parse the JSON array or object a model reply ends with, tolerating
        a chain of thought, a code fence, or surrounding prose.

        Return the parsed value, or :attr:`NO_JSON` when the reply has no
        JSON-looking span (plain prose).  Raise :class:`ValueError` when a
        ``[``/``{`` span is present but does not parse, so a truncated or
        invalid command batch is not mistaken for an empty one.

        The payload is the value that closes on the reply's last bracket, and
        the outermost one that does.  Each half of that rule answers its own
        failure.  A model reasoning in prose tries batches out before it
        answers.  An earlier value is therefore one the model rejected, and
        it must not run in place of a malformed final batch.  A command can
        nest an array, such as a polygon's vertices.  That array closes an
        inner bracket last, so the innermost value is an argument, not the
        batch.

        A candidate must also pass :meth:`opens_value`, which bounds what a
        bracket-heavy reply costs.
        """
        text = cls.strip_code_fences(cls.strip_reasoning(text)).strip()
        if not text:
            return cls.NO_JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        end = max(text.rfind("]"), text.rfind("}"))
        start, saw_span = -1, False
        if end != -1:
            opener = "[" if text[end] == "]" else "{"
            start = text.find(opener)
        while -1 < start < end:
            if cls.opens_value(text, start + 1):
                saw_span = True
                try:
                    value, offset = decoder.raw_decode(text, start)
                    if offset == end + 1:
                        return value
                except json.JSONDecodeError:
                    pass
            start = text.find(opener, start + 1)
        if saw_span or text[0] in "[{":
            raise ValueError("model reply has malformed JSON")
        return cls.NO_JSON

    @classmethod
    def commands_of(cls, data):
        """The command dicts in an already-parsed JSON payload.

        Accept an array, or a lone object treated as a one-command array.
        Each command must be an object with a string ``op``; anything else
        raises :class:`ValueError`.
        """
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError("model reply is not a JSON array of commands")
        commands = []
        for entry in data:
            if not isinstance(entry, dict):
                raise ValueError("command is not an object: %r" % (entry,))
            op = entry.get("op")
            if not isinstance(op, str):
                raise ValueError(
                    "command needs a string \"op\": %r" % (entry,))
            commands.append(entry)
        return commands

    @classmethod
    def parse(cls, text):
        """Turn a model reply into a list of command dicts.

        Raise :class:`ValueError` on a malformed reply.  Plain prose yields an
        empty list; :meth:`parse_reply` is what tells the two apart.
        """
        data = cls.load_json_payload(text)
        return [] if data is cls.NO_JSON else cls.commands_of(data)

    @classmethod
    def parse_reply(cls, text):
        """:meth:`parse` as a :class:`ParsedReply`.

        A blank reply is :attr:`~ParseStatus.EMPTY` rather than prose: it
        carries no text worth recording, and a loop should end on it like an
        explicit ``[]``.
        """
        status = _backend.ParseStatus
        if not (text or "").strip():
            return ParsedReply(status=status.EMPTY)
        try:
            data = cls.load_json_payload(text)
        except ValueError as exc:
            return ParsedReply(status=status.MALFORMED, error=str(exc))
        if data is cls.NO_JSON:
            return ParsedReply(status=status.PROSE)
        try:
            commands = cls.commands_of(data)
        except ValueError as exc:
            return ParsedReply(status=status.MALFORMED, error=str(exc))
        return ParsedReply(
            commands, status.COMMANDS if commands else status.EMPTY)


class CancellableBackend:
    """The cancellation bookkeeping a backend with an in-flight call shares.

    A cancelled call surfaces as an ordinary failure (a killed child, a closed
    socket), so the flag is what lets :meth:`failure` report it as the
    deliberate stop it was instead of a transport fault a caller might retry.
    """

    _cancelled = False

    def begin(self):
        self._cancelled = False

    def failure(self, error, outcome=_backend.TransportOutcome.TRANSPORT):
        """A failed response, or ``CANCELLED`` when this call was stopped."""
        if self._cancelled:
            outcome = _backend.TransportOutcome.CANCELLED
        return _backend.BackendResponse(error=error, outcome=outcome)

    def cancelled_reply(self):
        """``CANCELLED`` response if this call was stopped, else ``None``.

        A cancel that lands before the child or the connection is reachable
        tears down nothing, so the call can still succeed.  That answer is
        unwanted: returning it would let commands land after the user asked
        for none.
        """
        if not self._cancelled:
            return None
        return self.failure("cancelled")


class SubprocessBackend(CancellableBackend, _backend.AgentBackend):
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

    #: The selector label a subclass names itself with.
    name = None

    #: The process basics every agent CLI receives.  A subclass extends this
    #: with only its own authentication variables.
    env_passthrough = (
        "HOME", "USER", "LOGNAME", "PATH", "TMPDIR")

    def __init__(self, timeout=120):
        super().__init__()
        # Naming itself is the one thing the base cannot do for a subclass, and
        # a nameless backend would reach the selector and the configuration
        # file as a null entry.
        if not self.name:
            raise TypeError("%s must set name" % type(self).__name__)
        self._timeout = timeout
        self._proc = None

    def executable(self):
        """The resolved path to :attr:`command`, or ``None`` if not on PATH."""
        return shutil.which(self.command) if self.command else None

    def available(self):
        return self.executable() is not None

    @abc.abstractmethod
    def _build_argv(self, exe, user_prompt, system_prompt):
        """The argv that runs ``exe`` on the ``user_prompt``, passing
        ``system_prompt`` through whatever system-prompt channel the CLI
        offers."""

    def _parse_output(self, stdout):
        """Extract the assistant text from CLI ``stdout``.  The default treats
        stdout as the reply; override for a CLI that wraps it (JSON, etc.)."""
        return (stdout or "").strip()

    def send(self, prompt, scene_context, tool_surface, history=()):
        self.begin()
        exe = self.executable()
        if exe is None:
            return self.failure("%s not found on PATH" % self.command)
        user_prompt = self._compose_user(
            prompt, scene_context, tool_surface, history)
        argv = self._build_argv(exe, user_prompt, self._INSTRUCTIONS)
        try:
            code, out, err = self._communicate(argv)
        except subprocess.TimeoutExpired:
            return self.failure("%s timed out" % self.command,
                                _backend.TransportOutcome.TIMEOUT)
        except OSError as exc:
            return self.failure("%s failed: %s" % (self.command, exc))
        if code != 0:
            return self.failure(
                "%s exit %d: %s" % (self.command, code, (err or "").strip()))
        stopped = self.cancelled_reply()
        if stopped is not None:
            return stopped
        text = self._parse_output(out)
        return ToolCallParser.parse_reply(text).response(text)

    def cancel(self):
        """Terminate the in-flight child, if any.  Safe to call from another
        thread while :meth:`send` blocks in :meth:`_communicate`."""
        self._cancelled = True
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
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, cwd=workdir, env=env)
            self._proc = proc
            if self._cancelled:
                # A cancel between spawning the child and publishing it here
                # found nothing to terminate; act on it now.
                proc.terminate()
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

    The model and the reasoning effort are settings the user picks; each maps
    to a CLI flag.  Naming the model matters because ``--setting-sources ""``
    cuts the CLI off from the config files it would otherwise pick one from,
    so :attr:`DEFAULT_CHOICE` leaves the flag out and the request runs on
    whatever the CLI itself defaults to, which moves over time.
    """

    command = "claude"
    name = "Claude Code"
    env_passthrough = SubprocessBackend.env_passthrough + (
        "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR")

    DEFAULT_CHOICE = "default"

    SETTINGS = (
        _backend.BackendSetting(
            name="model", label="Model",
            choices=(DEFAULT_CHOICE, "fable", "opus", "sonnet", "haiku"),
            default=DEFAULT_CHOICE,
            tooltip="Model alias passed to the CLI as --model."),
        _backend.BackendSetting(
            name="effort", label="Effort",
            choices=(DEFAULT_CHOICE, "low", "medium", "high", "xhigh", "max"),
            default=DEFAULT_CHOICE,
            tooltip="Reasoning effort passed to the CLI as --effort."),
    )

    def settings_spec(self):
        return self.SETTINGS

    def _build_argv(self, exe, user_prompt, system_prompt):
        # TODO: provide more permission and config to the CLI sandbox later.
        argv = [
            exe, "-p", user_prompt, "--output-format", "json",
            "--append-system-prompt", system_prompt,
            "--tools", "",
            "--permission-mode", "dontAsk",  # no interactive prompts
            "--setting-sources", "",  # no config files
            "--strict-mcp-config",  # no mcp config files
            "--disable-slash-commands",  # no interactive slash commands
            "--no-session-persistence",  # no session files
        ]
        for setting, flag in (("model", "--model"), ("effort", "--effort")):
            value = self.get_setting(setting)
            if value and value != self.DEFAULT_CHOICE:
                # Joined rather than two elements, so a value that happens to
                # start with a dash reaches the CLI as this flag's value and
                # never as another option.
                argv.append("%s=%s" % (flag, value))
        return argv

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


_backend.BackendRegistry.register(ClaudeCliBackend())


class CodexCliBackend(SubprocessBackend):
    """Backend over OpenAI's ``codex`` command-line tool.

    ``codex exec`` runs non-interactively in a fresh read-only workspace.  It
    receives no user configuration, rules, shell, apps, or web-search tool,
    and saves no session.  Authentication remains the CLI's concern: a stored
    login comes through ``HOME`` or ``CODEX_HOME``, and automation can provide
    ``CODEX_API_KEY`` for this child alone.

    The user may leave the model and reasoning effort on the moving CLI
    default or pin either setting.  The system prompt reaches Codex as
    developer instructions, separate from the composed user prompt.
    """

    command = "codex"
    name = "Codex"
    env_passthrough = SubprocessBackend.env_passthrough + (
        "CODEX_HOME", "CODEX_API_KEY")

    DEFAULT_CHOICE = "default"

    SETTINGS = (
        _backend.BackendSetting(
            name="model", label="Model",
            choices=(DEFAULT_CHOICE, "gpt-5.6-sol", "gpt-5.6-terra",
                     "gpt-5.6-luna"),
            default=DEFAULT_CHOICE,
            tooltip="Model passed to the CLI as --model."),
        _backend.BackendSetting(
            name="effort", label="Effort",
            choices=(DEFAULT_CHOICE, "low", "medium", "high", "xhigh", "max"),
            default=DEFAULT_CHOICE,
            tooltip="Reasoning effort passed as model_reasoning_effort."),
    )

    def settings_spec(self):
        return self.SETTINGS

    @staticmethod
    def _config(name, value):
        """One joined ``--config`` override with a TOML string value."""
        return "--config=%s=%s" % (name, json.dumps(value))

    def _build_argv(self, exe, user_prompt, system_prompt):
        argv = [
            exe, "exec", "--sandbox=read-only", "--skip-git-repo-check",
            "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--strict-config", "--color=never", "--disable=shell_tool",
            "--disable=apps",
            self._config("web_search", "disabled"),
            self._config("developer_instructions", system_prompt),
        ]
        model = self.get_setting("model")
        if model and model != self.DEFAULT_CHOICE:
            argv.append("--model=%s" % model)
        effort = self.get_setting("effort")
        if effort and effort != self.DEFAULT_CHOICE:
            argv.append(self._config("model_reasoning_effort", effort))
        argv.append(user_prompt)
        return argv


_backend.BackendRegistry.register(CodexCliBackend())


class OpenAIHttpBackend(CancellableBackend, _backend.AgentBackend):
    """Backend over an OpenAI-compatible Chat Completions HTTP API.

    Uses only the stdlib (``http.client`` and ``urllib.parse``); no vendor
    SDK.  Point ``base_url`` at OpenAI, Ollama's ``/v1`` endpoint, or any
    compatible server.

    The base URL and the model name are user-tunable knobs; see
    :meth:`settings_spec`.  The settings dialog edits them, and the
    configuration file keeps them across runs.  The knobs start on what the
    constructor or the ``SOLVCON_OPENAI_BASE_URL`` and
    ``SOLVCON_OPENAI_MODEL`` environment variables give.  A stored value then
    wins, because the user chose it in the dialog.

    The API key stays out of the knobs.  It comes only from the constructor
    or ``SOLVCON_OPENAI_API_KEY``.  The configuration file is plain text, and
    a saved knob would write the secret into it.  The in-flight connection is
    kept on the instance so a driver thread can :meth:`cancel`.
    """

    # Local Ollama's OpenAI-compatible root; override for a remote provider.
    _DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
    _DEFAULT_MODEL = "qwen2.5vl:7b"

    def __init__(self, base_url=None, model=None, api_key=None, timeout=120):
        super().__init__()
        self._url_default = base_url if base_url is not None else self._env_or(
            "SOLVCON_OPENAI_BASE_URL", self._DEFAULT_BASE_URL)
        self._model_default = model if model is not None else self._env_or(
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

    def settings_spec(self):
        """Return the free-text knobs for the server address and model name.

        The defaults are per instance, not class constants.  That is what
        carries the constructor arguments and the environment variables into
        the value the dialog opens on.  Emptying either knob restores that
        default, which
        :meth:`~solvcon.agent.AgentBackend.set_setting` does for every
        free-text knob.
        """
        return (
            _backend.BackendSetting(
                name="base_url", label="Base URL",
                default=self._url_default,
                tooltip="API root including the /v1 suffix, "
                        "such as https://api.openai.com/v1."),
            _backend.BackendSetting(
                name="model", label="Model",
                default=self._model_default,
                tooltip="Model name sent in the request body."),
        )

    @property
    def base_url(self):
        """API root including the ``/v1`` suffix, with no trailing slash."""
        return self.get_setting("base_url").rstrip("/")

    @property
    def model(self):
        return self.get_setting("model")

    def available(self):
        """True when both a base URL and a model name are configured."""
        return bool(self.base_url) and bool(self.model)

    def send(self, prompt, scene_context, tool_surface, history=()):
        self.begin()
        if not self.available():
            return self.failure(
                "openai http backend needs base_url and model")
        user_prompt = self._compose_user(
            prompt, scene_context, tool_surface, history)
        body = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self._INSTRUCTIONS},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            status, raw = self._post_chat(body)
        except TimeoutError:
            return self.failure("openai http timed out",
                                _backend.TransportOutcome.TIMEOUT)
        except (OSError, http.client.HTTPException) as exc:
            return self.failure("openai http failed: %s" % exc)
        if status != 200:
            detail = (raw or b"").decode("utf-8", errors="replace").strip()
            return self.failure(
                "openai http status %d: %s" % (status, detail))
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self.failure("openai http bad JSON: %s" % exc)
        text = self._parse_chat_payload(payload)
        if text is None:
            return self.failure(
                "openai http response missing assistant text")
        stopped = self.cancelled_reply()
        if stopped is not None:
            return stopped
        return ToolCallParser.parse_reply(text).response(text)

    def cancel(self):
        """Close the in-flight HTTP connection, if any.  Safe to call from
        another thread while :meth:`send` blocks in :meth:`_post_chat`."""
        self._cancelled = True
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
            if self._cancelled:
                # Cancel before publish: closing an unconnected conn does not
                # stop request() from opening the socket.
                raise OSError("cancelled before the request was sent")
            conn.request("POST", path, body=payload, headers=headers)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            try:
                conn.close()
            except OSError:
                pass
            self._conn = None


_backend.BackendRegistry.register(OpenAIHttpBackend())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
