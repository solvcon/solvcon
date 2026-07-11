# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests for the fallback drawing bridge: a dummy tool surface and executor
that stand in until Agent Draw (#965) ships the real World command schema."""

import unittest

from solvcon.agent import _draw


class ToolDefinitionsTC(unittest.TestCase):
    def test_surface_is_only_the_dummy_ping_tool(self):
        tools = _draw.tool_definitions()
        self.assertEqual([t["name"] for t in tools], ["ping"])
        self.assertEqual(tools[0]["parameters"], {})


class SimpleExecutorTC(unittest.TestCase):
    def test_ping_is_acknowledged_without_a_world(self):
        # The dummy op runs the dispatch path but touches no World.
        result = _draw.Executor(None).run({"op": "ping"})
        self.assertTrue(result.ok)
        self.assertEqual(result.op, "ping")

    def test_drawing_op_is_deferred_to_agent_draw(self):
        # Real ops live in agentdraw (#965); the fallback reports them as
        # unavailable rather than pretending to draw.
        result = _draw.Executor(None).run(
            {"op": "add_circle", "cx": 0.0, "cy": 0.0, "r": 1.0})
        self.assertFalse(result.ok)
        self.assertIn("Agent Draw", result.error)

    def test_non_dict_command_is_reported_not_raised(self):
        result = _draw.Executor(None).run("not a command")
        self.assertFalse(result.ok)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
