# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests for the concrete CLI/HTTP backends and tool-call parsing.

GUI-free by default: PATH discovery is patched, the child process is replaced,
and HTTP posts are stubbed, so no real ``claude`` CLI or network call runs.
These exercise the parsing contract, availability checks, and the ``send``
pipeline.  Opt-in classes hit a live CLI or OpenAI-compatible server.
"""

import json
import os
import subprocess
import unittest
from unittest import mock

from solvcon.agent import (
    BackendResponse,
    ClaudeCliBackend,
    OpenAIHttpBackend,
    SubprocessBackend,
    get_backend,
    parse_tool_calls,
)


_WHICH = "solvcon.agent._backends_impl.shutil.which"

_TOOLS = [
    {"name": "add_circle", "description": "add a circle"},
    {"name": "add_line", "description": "add a line"},
]


class ParseToolCallsTC(unittest.TestCase):
    def test_plain_json_array(self):
        text = '[{"op": "add_circle", "r": 1.0}]'
        self.assertEqual(
            parse_tool_calls(text, _TOOLS),
            [{"op": "add_circle", "r": 1.0}])

    def test_lone_object_becomes_one_command(self):
        commands = parse_tool_calls('{"op": "add_line"}', _TOOLS)
        self.assertEqual(commands, [{"op": "add_line"}])

    def test_strips_code_fence(self):
        text = '```json\n[{"op": "add_circle"}]\n```'
        self.assertEqual(parse_tool_calls(text, _TOOLS),
                         [{"op": "add_circle"}])

    def test_extracts_array_from_surrounding_prose(self):
        text = 'Sure! Here you go:\n[{"op": "add_circle"}]\nThanks.'
        self.assertEqual(parse_tool_calls(text, _TOOLS),
                         [{"op": "add_circle"}])

    def test_empty_array_is_empty(self):
        self.assertEqual(parse_tool_calls("[]", _TOOLS), [])

    def test_no_json_yields_empty(self):
        self.assertEqual(parse_tool_calls("I cannot help.", _TOOLS), [])

    def test_malformed_json_rejected(self):
        # A JSON-looking but invalid payload must not become a successful
        # empty batch; send() should record a parser error instead.
        with self.assertRaises(ValueError):
            parse_tool_calls('[{"op": "add_circle",}]', _TOOLS)
        with self.assertRaises(ValueError):
            parse_tool_calls('[{"op": "add_circle"', _TOOLS)

    def test_unknown_op_rejected(self):
        with self.assertRaises(ValueError):
            parse_tool_calls('[{"op": "delete_universe"}]', _TOOLS)

    def test_missing_op_rejected(self):
        with self.assertRaises(ValueError):
            parse_tool_calls('[{"r": 1.0}]', _TOOLS)

    def test_non_string_op_raises_valueerror_not_typeerror(self):
        # An unhashable op (list/object) must not blow past the ValueError
        # contract into a set-membership TypeError, or send() would crash
        # instead of recording an error.
        with self.assertRaises(ValueError):
            parse_tool_calls('[{"op": {"nested": 1}}]', _TOOLS)
        with self.assertRaises(ValueError):
            parse_tool_calls('[{"op": ["a", "b"]}]', _TOOLS)

    def test_no_tool_surface_skips_op_validation(self):
        # With no advertised ops (e.g. Agent Draw absent) any op is accepted,
        # so the pipeline still runs rather than rejecting everything.
        commands = parse_tool_calls('[{"op": "anything"}]', [])
        self.assertEqual(commands, [{"op": "anything"}])


class SubprocessBackendDiscoveryTC(unittest.TestCase):
    def test_available_true_when_on_path(self):
        backend = ClaudeCliBackend()
        with mock.patch(_WHICH, lambda name: "/usr/bin/" + name):
            self.assertTrue(backend.available())
            self.assertEqual(backend.executable(), "/usr/bin/claude")

    def test_available_false_when_absent(self):
        backend = ClaudeCliBackend()
        with mock.patch(_WHICH, lambda name: None):
            self.assertFalse(backend.available())

    def test_command_none_never_resolves(self):
        # A subclass that names no executable is never available even though
        # which() would answer for a real name.
        class Nameless(SubprocessBackend):
            def _build_argv(self, exe, prompt):
                return [exe]

        with mock.patch(_WHICH, lambda name: "/usr/bin/" + str(name)):
            self.assertIsNone(Nameless().executable())
            self.assertFalse(Nameless().available())


class ClaudeCliSendTC(unittest.TestCase):
    def setUp(self):
        self.backend = ClaudeCliBackend()
        patcher = mock.patch(_WHICH, lambda name: "/usr/bin/" + name)
        self.which = patcher.start()
        self.addCleanup(patcher.stop)

    def _envelope(self, result_text):
        return json.dumps({"type": "result", "result": result_text})

    def test_send_parses_commands(self):
        reply = self._envelope('[{"op": "add_circle", "r": 2.0}]')
        self.backend._communicate = lambda argv: (0, reply, "")
        response = self.backend.send("draw a circle", "empty world", _TOOLS)
        self.assertIsInstance(response, BackendResponse)
        self.assertIsNone(response.error)
        self.assertEqual(response.commands, [{"op": "add_circle", "r": 2.0}])

    def test_send_not_on_path_is_error(self):
        with mock.patch(_WHICH, lambda name: None):
            response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIsNotNone(response.error)
        self.assertEqual(response.commands, [])

    def test_send_nonzero_exit_is_error(self):
        self.backend._communicate = lambda argv: (1, "", "boom")
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIn("boom", response.error)
        self.assertEqual(response.commands, [])

    def test_send_timeout_is_error(self):
        def _timeout(argv):
            raise subprocess.TimeoutExpired(argv, 120)
        self.backend._communicate = _timeout
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIn("timed out", response.error)

    def test_send_unknown_op_reports_error_without_commands(self):
        reply = self._envelope('[{"op": "delete_universe"}]')
        self.backend._communicate = lambda argv: (0, reply, "")
        response = self.backend.send("wreck it", "scene", _TOOLS)
        self.assertIsNotNone(response.error)
        self.assertEqual(response.commands, [])

    def test_send_malformed_json_is_error_not_empty_success(self):
        # Invalid JSON must surface as an error, not a silent empty batch.
        reply = self._envelope('[{"op": "add_circle",}]')
        self.backend._communicate = lambda argv: (0, reply, "")
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIsNotNone(response.error)
        self.assertEqual(response.commands, [])

    def test_send_unhashable_op_is_error_not_crash(self):
        # A malformed reply must come back as an error result, never an
        # unhandled exception out of send().
        reply = self._envelope('[{"op": {"nested": 1}}]')
        self.backend._communicate = lambda argv: (0, reply, "")
        response = self.backend.send("break it", "scene", _TOOLS)
        self.assertIsNotNone(response.error)
        self.assertEqual(response.commands, [])

    def test_send_passes_prompt_and_json_flags(self):
        seen = {}

        def _capture(argv):
            seen["argv"] = argv
            return 0, self._envelope("[]"), ""

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


class RegistrationTC(unittest.TestCase):
    def test_claude_registers_on_import(self):
        backend = get_backend("claude (cli)")
        self.assertIsNotNone(backend)
        self.assertIsInstance(backend, ClaudeCliBackend)

    def test_openai_http_registers_on_import(self):
        backend = get_backend("openai (http)")
        self.assertIsNotNone(backend)
        self.assertIsInstance(backend, OpenAIHttpBackend)


class OpenAIHttpBackendTC(unittest.TestCase):
    def setUp(self):
        self.backend = OpenAIHttpBackend(
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
        self.assertFalse(OpenAIHttpBackend(
            base_url="", model="m").available())
        self.assertFalse(OpenAIHttpBackend(
            base_url="http://127.0.0.1:11434/v1", model="").available())

    def test_send_parses_commands(self):
        raw = self._chat_body('[{"op": "add_circle", "r": 2.0}]')
        self.backend._post_chat = lambda body: (200, raw)
        response = self.backend.send("draw a circle", "empty world", _TOOLS)
        self.assertIsInstance(response, BackendResponse)
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
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("hello", messages[0]["content"])
        self.assertIn("one shape", messages[0]["content"])
        self.assertIn("add_circle", messages[0]["content"])

    def test_send_http_error_status_is_error(self):
        self.backend._post_chat = lambda body: (500, b"boom")
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIn("status 500", response.error)
        self.assertEqual(response.commands, [])

    def test_send_transport_failure_is_error(self):
        def _fail(body):
            raise OSError("connection refused")
        self.backend._post_chat = _fail
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIn("failed", response.error)
        self.assertEqual(response.commands, [])

    def test_send_timeout_is_error(self):
        def _timeout(body):
            raise TimeoutError("timed out")
        self.backend._post_chat = _timeout
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIn("timed out", response.error)

    def test_send_unknown_op_reports_error_without_commands(self):
        raw = self._chat_body('[{"op": "delete_universe"}]')
        self.backend._post_chat = lambda body: (200, raw)
        response = self.backend.send("wreck it", "scene", _TOOLS)
        self.assertIsNotNone(response.error)
        self.assertEqual(response.commands, [])

    def test_send_malformed_json_is_error_not_empty_success(self):
        raw = self._chat_body('[{"op": "add_circle",}]')
        self.backend._post_chat = lambda body: (200, raw)
        response = self.backend.send("draw", "scene", _TOOLS)
        self.assertIsNotNone(response.error)
        self.assertEqual(response.commands, [])

    def test_parse_chat_payload_joins_content_parts(self):
        text = OpenAIHttpBackend._parse_chat_payload({
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
            backend = OpenAIHttpBackend()
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
        self.backend = ClaudeCliBackend()
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
        self.backend = OpenAIHttpBackend()
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
