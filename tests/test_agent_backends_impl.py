# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests for the concrete CLI backends and tool-call parsing.

GUI-free and process-free: PATH discovery is patched and the child process is
replaced, so no real ``claude`` CLI runs.  These exercise the parsing
contract, PATH-based availability, and the ``send`` pipeline.
"""

import json
import os
import subprocess
import unittest
from unittest import mock

from solvcon.agent import (
    BackendResponse,
    ClaudeCliBackend,
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


_REAL = "SOLVCON_TEST_REAL_CLAUDE"


@unittest.skipUnless(os.environ.get(_REAL),
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


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
