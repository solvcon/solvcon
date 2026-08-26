# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests for the concrete CLI/HTTP backends and tool-call parsing.

GUI-free by default: PATH discovery is patched, the child process is replaced,
and HTTP posts are stubbed, so no real AI CLI or network call runs.
These exercise the parsing contract, availability checks, and the ``send``
pipeline.  Opt-in classes hit a live CLI or OpenAI-compatible server.
"""

import json
import os
import subprocess
import unittest
from unittest import mock

from solvcon import agent


_WHICH = "solvcon.agent._backends_impl.shutil.which"

_TOOLS = [
    {"name": "add_circle", "description": "add a circle"},
    {"name": "add_line", "description": "add a line"},
]


def _envelope(result_text):
    """A ``claude --output-format json`` reply carrying ``result_text``."""
    return json.dumps({"type": "result", "result": result_text})


class ParseToolCallsTC(unittest.TestCase):
    def test_plain_json_array(self):
        text = '[{"op": "add_circle", "r": 1.0}]'
        self.assertEqual(
            agent.ToolCallParser.parse(text),
            [{"op": "add_circle", "r": 1.0}])

    def test_lone_object_becomes_one_command(self):
        commands = agent.ToolCallParser.parse('{"op": "add_line"}')
        self.assertEqual(commands, [{"op": "add_line"}])

    def test_strips_code_fence(self):
        text = '```json\n[{"op": "add_circle"}]\n```'
        self.assertEqual(agent.ToolCallParser.parse(text),
                         [{"op": "add_circle"}])

    def test_extracts_array_from_surrounding_prose(self):
        text = 'Sure! Here you go:\n[{"op": "add_circle"}]\nThanks.'
        self.assertEqual(agent.ToolCallParser.parse(text),
                         [{"op": "add_circle"}])

    def test_empty_array_is_empty(self):
        self.assertEqual(agent.ToolCallParser.parse("[]"), [])

    def test_no_json_yields_empty(self):
        self.assertEqual(agent.ToolCallParser.parse("I cannot help."), [])

    def test_malformed_json_rejected(self):
        # A JSON-looking but invalid payload must not become a successful
        # empty batch; send() should record a parser error instead.
        with self.assertRaises(ValueError):
            agent.ToolCallParser.parse('[{"op": "add_circle",}]')
        with self.assertRaises(ValueError):
            agent.ToolCallParser.parse('[{"op": "add_circle"')

    def test_missing_op_rejected(self):
        with self.assertRaises(ValueError):
            agent.ToolCallParser.parse('[{"r": 1.0}]')

    def test_non_string_op_raises_valueerror_not_typeerror(self):
        with self.assertRaises(ValueError):
            agent.ToolCallParser.parse('[{"op": {"nested": 1}}]')
        with self.assertRaises(ValueError):
            agent.ToolCallParser.parse('[{"op": ["a", "b"]}]')

    def test_unknown_op_survives_parsing(self):
        commands = agent.ToolCallParser.parse(
            '[{"op": "add_circle"}, {"op": "delete_universe"}]')
        self.assertEqual([command["op"] for command in commands],
                         ["add_circle", "delete_universe"])


class ParseReplyStatusTC(unittest.TestCase):
    def _status(self, text):
        return agent.ToolCallParser.parse_reply(text).status

    def test_commands(self):
        reply = agent.ToolCallParser.parse_reply('[{"op": "add_line"}]')
        self.assertEqual(reply.status, agent.ParseStatus.COMMANDS)
        self.assertEqual(reply.commands, [{"op": "add_line"}])
        self.assertIsNone(reply.error)

    def test_explicit_empty_batch_is_not_prose(self):
        self.assertEqual(self._status("[]"), agent.ParseStatus.EMPTY)
        self.assertEqual(self._status("```json\n[]\n```"),
                         agent.ParseStatus.EMPTY)
        self.assertEqual(self._status("I cannot do that."),
                         agent.ParseStatus.PROSE)

    def test_blank_reply_is_empty(self):
        self.assertEqual(self._status("   \n"), agent.ParseStatus.EMPTY)

    def test_malformed_carries_the_parse_error(self):
        reply = agent.ToolCallParser.parse_reply('[{"op": "add_circle",}]')
        self.assertEqual(reply.status, agent.ParseStatus.MALFORMED)
        self.assertEqual(reply.commands, [])
        self.assertTrue(reply.error)

    def test_bad_command_entry_is_malformed(self):
        self.assertEqual(self._status('[{"r": 1.0}]'),
                         agent.ParseStatus.MALFORMED)

    def test_a_json_scalar_is_malformed_not_prose(self):
        # `null` is the trap: it parses, but to the wrong shape.
        for text in ("null", "42", '"done"', "true"):
            self.assertEqual(self._status(text),
                             agent.ParseStatus.MALFORMED, text)


class SubprocessBackendDiscoveryTC(unittest.TestCase):
    def test_available_true_when_on_path(self):
        backend = agent.ClaudeCliBackend()
        with mock.patch(_WHICH, lambda name: "/usr/bin/" + name):
            self.assertTrue(backend.available())
            self.assertEqual(backend.executable(), "/usr/bin/claude")

    def test_available_false_when_absent(self):
        backend = agent.ClaudeCliBackend()
        with mock.patch(_WHICH, lambda name: None):
            self.assertFalse(backend.available())

    def test_codex_resolves_its_own_executable(self):
        backend = agent.CodexCliBackend()
        with mock.patch(_WHICH, lambda name: "/usr/bin/" + name):
            self.assertTrue(backend.available())
            self.assertEqual(backend.executable(), "/usr/bin/codex")

    def test_command_none_never_resolves(self):
        # A subclass that names no executable is never available even though
        # which() would answer for a real name.
        class Nameless(agent.SubprocessBackend):
            name = "nameless (test)"

            def _build_argv(self, exe, user_prompt, system_prompt):
                return [exe]

        with mock.patch(_WHICH, lambda name: "/usr/bin/" + str(name)):
            self.assertIsNone(Nameless().executable())
            self.assertFalse(Nameless().available())


class ClaudeCliSendTC(unittest.TestCase):
    def setUp(self):
        self.backend = agent.ClaudeCliBackend()
        patcher = mock.patch(_WHICH, lambda name: "/usr/bin/" + name)
        self.which = patcher.start()
        self.addCleanup(patcher.stop)

    def test_send_parses_commands(self):
        reply = _envelope('[{"op": "add_circle", "r": 2.0}]')
        self.backend._communicate = lambda argv: (0, reply, "")
        response = self.backend.send("draw a circle", "empty world", _TOOLS)
        self.assertIsInstance(response, agent.BackendResponse)
        self.assertIsNone(response.error)
        self.assertEqual(response.commands, [{"op": "add_circle", "r": 2.0}])

    def test_send_not_on_path_is_error(self):
        with mock.patch(_WHICH, lambda name: None):
            response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIsNotNone(response.error)
        self.assertEqual(response.commands, [])
        self.assertEqual(response.outcome, agent.TransportOutcome.TRANSPORT)

    def test_send_nonzero_exit_is_error(self):
        self.backend._communicate = lambda argv: (1, "", "boom")
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIn("boom", response.error)
        self.assertEqual(response.commands, [])
        self.assertEqual(response.outcome, agent.TransportOutcome.TRANSPORT)

    def test_send_timeout_is_error(self):
        def _timeout(argv):
            raise subprocess.TimeoutExpired(argv, 120)
        self.backend._communicate = _timeout
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIn("timed out", response.error)
        self.assertEqual(response.outcome, agent.TransportOutcome.TIMEOUT)

    def test_send_after_cancel_reports_cancelled_not_transport(self):
        def _killed(argv):
            self.backend.cancel()
            return -15, "", ""

        self.backend._communicate = _killed
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertEqual(response.outcome, agent.TransportOutcome.CANCELLED)

    def test_a_cancelled_call_that_still_succeeded_applies_nothing(self):
        def _survived(argv):
            self.backend.cancel()
            return 0, _envelope('[{"op": "add_circle"}]'), ""

        self.backend._communicate = _survived
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertEqual(response.outcome, agent.TransportOutcome.CANCELLED)
        self.assertEqual(response.commands, [])

    def test_a_cancel_during_spawn_still_terminates_the_child(self):
        killed = []

        class _Proc:
            returncode = 0

            def poll(self):
                return None

            def terminate(self):
                killed.append(True)

            def communicate(self, timeout=None):
                return _envelope("[]"), ""

        def _popen(argv, **kwargs):
            self.backend.cancel()
            return _Proc()

        with mock.patch("subprocess.Popen", _popen):
            self.backend.send("draw", "scene", _TOOLS)
        self.assertTrue(killed)

    def test_send_forgets_an_earlier_cancel(self):
        self.backend.cancel()
        self.backend._communicate = lambda argv: (1, "", "boom")
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertEqual(response.outcome, agent.TransportOutcome.TRANSPORT)

    def test_send_unknown_op_reaches_the_runner(self):
        reply = _envelope('[{"op": "delete_universe"}]')
        self.backend._communicate = lambda argv: (0, reply, "")
        response = self.backend.send("wreck it", "scene", _TOOLS)
        self.assertIsNone(response.error)
        self.assertEqual(response.commands, [{"op": "delete_universe"}])
        self.assertEqual(response.status, agent.ParseStatus.COMMANDS)

    def test_send_malformed_json_is_error_not_empty_success(self):
        # Invalid JSON must surface as an error, not a silent empty batch.
        reply = _envelope('[{"op": "add_circle",}]')
        self.backend._communicate = lambda argv: (0, reply, "")
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIsNotNone(response.error)
        self.assertEqual(response.commands, [])
        self.assertEqual(response.outcome, agent.TransportOutcome.OK)
        self.assertEqual(response.status, agent.ParseStatus.MALFORMED)

    def test_send_unhashable_op_is_error_not_crash(self):
        # A malformed reply must come back as an error result, never an
        # unhandled exception out of send().
        reply = _envelope('[{"op": {"nested": 1}}]')
        self.backend._communicate = lambda argv: (0, reply, "")
        response = self.backend.send("break it", "scene", _TOOLS)
        self.assertIsNotNone(response.error)
        self.assertEqual(response.commands, [])

    def test_send_passes_prompt_and_json_flags(self):
        seen = {}

        def _capture(argv):
            seen["argv"] = argv
            return 0, _envelope("[]"), ""

        self.backend._communicate = _capture
        self.backend.send("hello", "one shape", _TOOLS)
        argv = seen["argv"]
        self.assertEqual(argv[0], "/usr/bin/claude")
        self.assertIn("-p", argv)
        self.assertIn("--output-format", argv)
        self.assertIn("json", argv)
        prompt = argv[argv.index("-p") + 1]
        self.assertIn("hello", prompt)
        self.assertIn("one shape", prompt)
        self.assertIn("add_circle", prompt)  # tool surface folded in
        # The instruction rides as a real system prompt, not the user turn.
        self.assertIn("--append-system-prompt", argv)
        system = argv[argv.index("--append-system-prompt") + 1]
        self.assertIn("drive a 2D drawing canvas", system)
        self.assertNotIn("drive a 2D drawing canvas", prompt)


class CodexCliSendTC(unittest.TestCase):
    def setUp(self):
        self.backend = agent.CodexCliBackend()
        patcher = mock.patch(_WHICH, lambda name: "/usr/bin/" + name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_send_parses_commands(self):
        reply = '[{"op": "add_circle", "r": 2.0}]'
        self.backend._communicate = lambda argv: (0, reply, "")
        response = self.backend.send(
            "draw a circle", "empty world", _TOOLS)
        self.assertIsNone(response.error)
        self.assertEqual(response.commands, [{"op": "add_circle", "r": 2.0}])

    def test_send_passes_prompt_and_pins_the_cli(self):
        seen = {}

        def _capture(argv):
            seen["argv"] = argv
            return 0, "[]", ""

        self.backend._communicate = _capture
        self.backend.send("hello", "one shape", _TOOLS)
        argv = seen["argv"]
        self.assertEqual(argv[:2], ["/usr/bin/codex", "exec"])
        for flag in ("--sandbox=read-only", "--skip-git-repo-check",
                     "--ephemeral", "--ignore-user-config", "--ignore-rules",
                     "--strict-config", "--color=never",
                     "--disable=shell_tool", "--disable=apps"):
            self.assertIn(flag, argv)
        self.assertIn('--config=web_search="disabled"', argv)
        prompt = argv[-1]
        self.assertIn("hello", prompt)
        self.assertIn("one shape", prompt)
        self.assertIn("add_circle", prompt)
        system_arg = next(
            arg for arg in argv
            if arg.startswith("--config=developer_instructions="))
        system = json.loads(system_arg.split("=", 2)[2])
        self.assertIn("drive a 2D drawing canvas", system)
        self.assertNotIn("drive a 2D drawing canvas", prompt)

    def test_settings_reach_the_cli(self):
        self.backend.set_setting("model", "gpt-5.6-sol")
        self.backend.set_setting("effort", "high")
        argv = self.backend._build_argv(
            "/usr/bin/codex", "draw", "system")
        self.assertIn("--model=gpt-5.6-sol", argv)
        self.assertIn('--config=model_reasoning_effort="high"', argv)

    def test_default_settings_leave_the_cli_defaults_alone(self):
        argv = self.backend._build_argv(
            "/usr/bin/codex", "draw", "system")
        self.assertFalse(any(arg.startswith("--model=") for arg in argv))
        self.assertFalse(any("model_reasoning_effort" in arg for arg in argv))


class _FakeProc:
    """Stand-in for the CLI child: records nothing, answers an empty batch."""

    def __init__(self, stdout):
        self._stdout = stdout
        self.returncode = 0

    def communicate(self, timeout=None):
        return self._stdout, ""


class SubprocessBackendPinningTC(unittest.TestCase):
    """The child process runs pinned: a scratch cwd and an allowlisted env.

    These assert the boundary the harness builds, by capturing the arguments
    it hands ``Popen``.  They prove no unlisted variable is passed, not that
    the CLI itself reads nothing else; proving that needs a live run against
    the real binary.
    """

    _SECRETS = {"AWS_SECRET_ACCESS_KEY": "aws", "GITHUB_TOKEN": "gh",
                "SSH_AUTH_SOCK": "/tmp/sock", "OPENAI_API_KEY": "oai"}

    def setUp(self):
        self.backend = agent.ClaudeCliBackend()
        patcher = mock.patch(_WHICH, lambda name: "/usr/bin/" + name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, environ):
        """Send one prompt under ``environ``; return the captured Popen
        keyword arguments."""
        seen = {}

        def _popen(argv, **kwargs):
            seen.update(kwargs)
            seen["argv"] = argv
            # Sampled here because send() removes the directory before
            # returning, which is itself what the caller then asserts.
            seen["cwd_exists"] = os.path.isdir(kwargs["cwd"])
            return _FakeProc(_envelope("[]"))

        with mock.patch.dict(os.environ, environ, clear=True):
            with mock.patch("subprocess.Popen", _popen):
                response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIsNone(response.error)
        return seen

    def test_child_env_is_exactly_the_allowlist(self):
        environ = {"HOME": "/home/u", "USER": "u", "LOGNAME": "u",
                   "PATH": "/bin", "TMPDIR": "/tmp",
                   "ANTHROPIC_API_KEY": "key", **self._SECRETS}
        seen = self._run(environ)
        self.assertEqual(seen["env"], {
            "HOME": "/home/u", "USER": "u", "LOGNAME": "u",
            "PATH": "/bin", "TMPDIR": "/tmp", "ANTHROPIC_API_KEY": "key"})

    def test_unset_allowlist_entries_are_not_invented(self):
        # Only what the parent actually holds is forwarded, so an absent
        # variable stays absent instead of arriving empty.
        seen = self._run({"PATH": "/bin"})
        self.assertEqual(seen["env"], {"PATH": "/bin"})

    def test_windows_process_basics_reach_the_child(self):
        basics = {
            "PATH": r"C:\Windows\System32",
            "USERPROFILE": r"C:\Users\user",
            "SystemRoot": r"C:\Windows",
            "ComSpec": r"C:\Windows\System32\cmd.exe",
            "PATHEXT": ".COM;.EXE;.BAT;.CMD",
            "TEMP": r"C:\Users\user\Temp",
            "TMP": r"C:\Users\user\Temp",
            "APPDATA": r"C:\Users\user\AppData\Roaming",
            "LOCALAPPDATA": r"C:\Users\user\AppData\Local",
        }
        seen = self._run({**basics, **self._SECRETS})
        self.assertEqual(seen["env"], basics)

    def test_each_supported_auth_mode_reaches_the_child(self):
        base = {"HOME": "/home/u", "PATH": "/bin"}
        modes = (
            {"ANTHROPIC_API_KEY": "key"},
            {"CLAUDE_CODE_OAUTH_TOKEN": "token"},
            {"CLAUDE_CONFIG_DIR": "/home/u/.claude"},  # stored login
        )
        for mode in modes:
            seen = self._run({**base, **mode, **self._SECRETS})
            self.assertEqual(seen["env"], {**base, **mode})

    def test_child_runs_in_a_scratch_directory_removed_afterwards(self):
        seen = self._run({"PATH": "/bin"})
        workdir = seen["cwd"]
        self.assertTrue(seen["cwd_exists"])
        self.assertNotEqual(workdir, os.getcwd())
        self.assertIn("solvcon-agent-", os.path.basename(workdir))
        self.assertFalse(os.path.exists(workdir))

    def test_child_cannot_read_the_parent_stdin(self):
        seen = self._run({"PATH": "/bin"})
        self.assertEqual(seen["stdin"], subprocess.DEVNULL)

    def test_child_output_is_decoded_as_utf8(self):
        seen = self._run({"PATH": "/bin"})
        self.assertTrue(seen["text"])
        self.assertEqual(seen["encoding"], "utf-8")
        self.assertEqual(seen["errors"], "replace")

    def test_argv_pins_the_cli_sandbox(self):
        argv = self._run({"PATH": "/bin"})["argv"]
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "")
        for flag in ("--strict-mcp-config", "--disable-slash-commands",
                     "--no-session-persistence"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--permission-mode") + 1], "dontAsk")


class CodexCliPinningTC(unittest.TestCase):
    def test_child_receives_only_codex_auth(self):
        backend = agent.CodexCliBackend()
        seen = {}

        def _popen(argv, **kwargs):
            seen.update(kwargs)
            return _FakeProc("[]")

        environ = {
            "HOME": "/home/u", "PATH": "/bin",
            "CODEX_HOME": "/home/u/.codex", "CODEX_API_KEY": "codex",
            "ANTHROPIC_API_KEY": "anthropic", "GITHUB_TOKEN": "github",
        }
        with mock.patch(_WHICH, lambda name: "/usr/bin/" + name):
            with mock.patch.dict(os.environ, environ, clear=True):
                with mock.patch("subprocess.Popen", _popen):
                    response = backend.send("draw", "scene", _TOOLS)
        self.assertIsNone(response.error)
        self.assertEqual(seen["env"], {
            "HOME": "/home/u", "PATH": "/bin",
            "CODEX_HOME": "/home/u/.codex", "CODEX_API_KEY": "codex",
        })


class RegistrationTC(unittest.TestCase):
    def test_claude_registers_on_import(self):
        backend = agent.BackendRegistry.get("Claude Code")
        self.assertIsNotNone(backend)
        self.assertIsInstance(backend, agent.ClaudeCliBackend)

    def test_openai_http_registers_on_import(self):
        backend = agent.BackendRegistry.get("openai (http)")
        self.assertIsNotNone(backend)
        self.assertIsInstance(backend, agent.OpenAIHttpBackend)

    def test_codex_registers_on_import(self):
        backend = agent.BackendRegistry.get("Codex")
        self.assertIsNotNone(backend)
        self.assertIsInstance(backend, agent.CodexCliBackend)


class OpenAIHttpBackendTC(unittest.TestCase):
    def setUp(self):
        self.backend = agent.OpenAIHttpBackend(
            base_url="http://127.0.0.1:11434/v1",
            model="qwen2.5vl:7b",
            api_key="")

    def _chat_body(self, content):
        message = {"role": "assistant", "content": content}
        return json.dumps({
            "choices": [{"message": message}],
        }).encode("utf-8")

    def test_available_needs_url_and_model(self):
        self.assertTrue(self.backend.available())
        self.assertFalse(agent.OpenAIHttpBackend(
            base_url="", model="m").available())
        self.assertFalse(agent.OpenAIHttpBackend(
            base_url="http://127.0.0.1:11434/v1", model="").available())

    def test_send_parses_commands(self):
        raw = self._chat_body('[{"op": "add_circle", "r": 2.0}]')
        self.backend._post_chat = lambda body: (200, raw)
        response = self.backend.send("draw a circle", "empty world", _TOOLS)
        self.assertIsInstance(response, agent.BackendResponse)
        self.assertIsNone(response.error)
        self.assertEqual(response.commands, [{"op": "add_circle", "r": 2.0}])

    def test_send_posts_openai_chat_shape(self):
        seen = {}

        def _capture(body):
            seen["body"] = body
            return 200, self._chat_body("[]")

        self.backend._post_chat = _capture
        self.backend.send("hello", "one shape", _TOOLS)
        body = seen["body"]
        self.assertEqual(body["model"], "qwen2.5vl:7b")
        self.assertIs(body["stream"], False)
        messages = body["messages"]
        self.assertEqual(len(messages), 2)
        # The instruction rides as a real system prompt, first; the user turn
        # carries only the tools, scene, and request.
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("drive a 2D drawing canvas", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("hello", messages[1]["content"])
        self.assertIn("one shape", messages[1]["content"])
        self.assertIn("add_circle", messages[1]["content"])
        self.assertNotIn("drive a 2D drawing canvas", messages[1]["content"])

    def test_send_http_error_status_is_error(self):
        self.backend._post_chat = lambda body: (500, b"boom")
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIn("status 500", response.error)
        self.assertEqual(response.commands, [])
        self.assertEqual(response.outcome, agent.TransportOutcome.TRANSPORT)

    def test_send_transport_failure_is_error(self):
        def _fail(body):
            raise OSError("connection refused")
        self.backend._post_chat = _fail
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIn("failed", response.error)
        self.assertEqual(response.commands, [])
        self.assertEqual(response.outcome, agent.TransportOutcome.TRANSPORT)

    def test_send_timeout_is_error(self):
        def _timeout(body):
            raise TimeoutError("timed out")
        self.backend._post_chat = _timeout
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIn("timed out", response.error)
        self.assertEqual(response.outcome, agent.TransportOutcome.TIMEOUT)

    def test_send_after_cancel_reports_cancelled(self):
        def _closed(body):
            self.backend.cancel()
            raise OSError("closed")

        self.backend._post_chat = _closed
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertEqual(response.outcome, agent.TransportOutcome.CANCELLED)

    def test_a_cancel_before_the_socket_opens_sends_nothing(self):
        sent = []

        class FakeConn:
            def __init__(self, host, port, timeout=None):
                pass

            def request(self, method, path, body=None, headers=None):
                sent.append(path)

            def getresponse(self):
                raise AssertionError("the cancelled request was sent")

            def close(self):
                pass

        def _cancel_then_build(*args, **kwargs):
            self.backend.cancel()
            return FakeConn(*args, **kwargs)

        with mock.patch(
                "solvcon.agent._backends_impl.http.client.HTTPConnection",
                _cancel_then_build):
            response = self.backend.send("draw", "scene", _TOOLS)
        self.assertEqual(sent, [])
        self.assertEqual(response.outcome, agent.TransportOutcome.CANCELLED)

    def test_a_cancelled_call_that_still_answered_applies_nothing(self):
        raw = self._chat_body('[{"op": "add_circle"}]')

        def _raced(body):
            self.backend.cancel()
            return 200, raw

        self.backend._post_chat = _raced
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertEqual(response.outcome, agent.TransportOutcome.CANCELLED)
        self.assertEqual(response.commands, [])

    def test_send_unknown_op_reaches_the_runner(self):
        raw = self._chat_body('[{"op": "delete_universe"}]')
        self.backend._post_chat = lambda body: (200, raw)
        response = self.backend.send("wreck it", "scene", _TOOLS)
        self.assertIsNone(response.error)
        self.assertEqual(response.commands, [{"op": "delete_universe"}])

    def test_send_malformed_json_is_error_not_empty_success(self):
        raw = self._chat_body('[{"op": "add_circle",}]')
        self.backend._post_chat = lambda body: (200, raw)
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIsNotNone(response.error)
        self.assertEqual(response.commands, [])
        self.assertEqual(response.status, agent.ParseStatus.MALFORMED)

    def test_parse_chat_payload_joins_content_parts(self):
        text = agent.OpenAIHttpBackend._parse_chat_payload({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": '[{"op": '},
                        {"type": "text", "text": '"add_line"}]'},
                    ],
                },
            }],
        })
        self.assertEqual(text, '[{"op": "add_line"}]')

    def test_env_defaults_when_ctor_omits(self):
        env = {
            "SOLVCON_OPENAI_BASE_URL": "http://example.test/v1",
            "SOLVCON_OPENAI_MODEL": "demo-model",
            "SOLVCON_OPENAI_API_KEY": "secret",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            backend = agent.OpenAIHttpBackend()
        self.assertEqual(backend.base_url, "http://example.test/v1")
        self.assertEqual(backend.model, "demo-model")
        self.assertEqual(backend._api_key, "secret")

    def test_post_chat_uses_http_client(self):
        # Stub http.client so send() still exercises URL, headers, and path
        # assembly without a live server.
        seen = {}
        raw = self._chat_body('[{"op": "add_circle"}]')

        class FakeResponse:
            status = 200

            def read(self):
                return raw

        class FakeConn:
            def __init__(self, host, port, timeout=None):
                seen["host"] = host
                seen["port"] = port
                seen["timeout"] = timeout

            def request(self, method, path, body=None, headers=None):
                seen["method"] = method
                seen["path"] = path
                seen["body"] = body
                seen["headers"] = headers

            def getresponse(self):
                return FakeResponse()

            def close(self):
                seen["closed"] = True

        self.backend._api_key = "tok"
        with mock.patch(
                "solvcon.agent._backends_impl.http.client.HTTPConnection",
                FakeConn):
            response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIsNone(response.error)
        self.assertEqual(response.commands, [{"op": "add_circle"}])
        self.assertEqual(seen["host"], "127.0.0.1")
        self.assertEqual(seen["port"], 11434)
        self.assertEqual(seen["method"], "POST")
        self.assertEqual(seen["path"], "/v1/chat/completions")
        self.assertEqual(seen["headers"]["Authorization"], "Bearer tok")
        self.assertTrue(seen.get("closed"))


_REAL = "SOLVCON_TEST_REAL_CLAUDE"


@unittest.skipUnless(os.environ.get(_REAL) == "1",
                     "set %s=1 to hit the installed claude CLI" % _REAL)
class ClaudeCliRealTC(unittest.TestCase):
    """Opt-in end-to-end test against the installed claude CLI.

    Skipped by default so CI stays hermetic and free; a local run with
    ``SOLVCON_TEST_REAL_CLAUDE=1`` makes a real, billed CLI call to confirm
    the flags, the JSON envelope, and parsing hold against the live tool.  It
    hands a hand-written tool surface in place of Agent Draw's, so the package
    need not be present.
    """

    def setUp(self):
        self.backend = agent.ClaudeCliBackend()
        if not self.backend.available():
            self.skipTest("claude CLI not found on PATH")

    def test_draws_a_circle_end_to_end(self):
        response = self.backend.send(
            "Add exactly one circle of radius 1 at the origin.",
            "empty world with 0 shapes", _TOOLS)
        # A real reply must parse cleanly into circle-drawing commands; a
        # broken flag, envelope, or parser would surface as an error or an
        # empty batch here.
        self.assertIsNone(response.error)
        self.assertTrue(response.commands)
        ops = {tool["name"] for tool in _TOOLS}
        for command in response.commands:
            self.assertIn(command.get("op"), ops)
        self.assertIn("add_circle",
                      [command["op"] for command in response.commands])


_REAL_CODEX = "SOLVCON_TEST_REAL_CODEX"


@unittest.skipUnless(os.environ.get(_REAL_CODEX) == "1",
                     "set %s=1 to hit the installed codex CLI" % _REAL_CODEX)
class CodexCliRealTC(unittest.TestCase):
    """Opt-in end-to-end test against the installed Codex CLI."""

    def setUp(self):
        self.backend = agent.CodexCliBackend()
        if not self.backend.available():
            self.skipTest("codex CLI not found on PATH")

    def test_draws_a_circle_end_to_end(self):
        response = self.backend.send(
            "Add exactly one circle of radius 1 at the origin.",
            "empty world with 0 shapes", _TOOLS)
        self.assertIsNone(response.error)
        self.assertTrue(response.commands)
        ops = {tool["name"] for tool in _TOOLS}
        for command in response.commands:
            self.assertIn(command.get("op"), ops)
        self.assertIn("add_circle",
                      [command["op"] for command in response.commands])


_REAL_OPENAI = "SOLVCON_TEST_REAL_OPENAI_HTTP"


@unittest.skipUnless(
    os.environ.get(_REAL_OPENAI) == "1",
    "set %s=1 to hit a live OpenAI-compatible server" % _REAL_OPENAI)
class OpenAIHttpRealTC(unittest.TestCase):
    """Opt-in end-to-end test against a live OpenAI-compatible server.

    Skipped by default so CI stays hermetic.  A local run with
    ``SOLVCON_TEST_REAL_OPENAI_HTTP=1`` posts to the configured base URL
    (default: Ollama at ``http://127.0.0.1:11434/v1``) to confirm the request
    shape, response parsing, and command extraction against a real model.
    """

    def setUp(self):
        self.backend = agent.OpenAIHttpBackend()
        if not self.backend.available():
            self.skipTest("openai http backend not configured")

    def test_draws_a_circle_end_to_end(self):
        response = self.backend.send(
            "Add exactly one circle of radius 1 at the origin.",
            "empty world with 0 shapes", _TOOLS)
        self.assertIsNone(response.error, response.error)
        self.assertTrue(response.commands)
        ops = {tool["name"] for tool in _TOOLS}
        for command in response.commands:
            self.assertIn(command.get("op"), ops)
        self.assertIn("add_circle",
                      [command["op"] for command in response.commands])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
