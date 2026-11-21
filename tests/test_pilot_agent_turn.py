# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the helpers behind an agent turn's block in the Console dock: the
command rows, the replay as Python, and the steps Undo turn takes.
"""

import math
import unittest

from solvcon import agent, testing

try:
    from solvcon import pilot
    from solvcon.pilot.agent import _agent_turn
except ImportError:
    pilot = None


@testing.requires_pilot
class FormatTC(unittest.TestCase):

    def test_format_command_lists_arguments(self):
        self.assertEqual(
            _agent_turn.format_command(
                {"op": "add_circle", "cx": 0.0, "cy": 1.5, "r": 2.0}),
            "add_circle cx=0 cy=1.5 r=2")

    def test_commands_as_python_replays_through_agent_run(self):
        source = _agent_turn.commands_as_python(
            [{"op": "add_circle", "cx": 0.0, "cy": 0.0, "r": 1.0}])
        self.assertTrue(source.startswith("agent_run(["))
        captured = []
        exec(source, {"agent_run": captured.extend})
        self.assertEqual(captured, [
            {"op": "add_circle", "cx": 0.0, "cy": 0.0, "r": 1.0}])

    def test_commands_as_python_reads_back_infinities(self):
        command = {"op": "pan", "dx": float("inf"), "dy": float("-inf"),
                   "pts": [(0.0, float("nan"))]}
        captured = []
        exec(_agent_turn.commands_as_python([command]),
             {"agent_run": captured.extend})
        (back,) = captured
        self.assertEqual(back["dx"], float("inf"))
        self.assertEqual(back["dy"], float("-inf"))
        self.assertTrue(math.isnan(back["pts"][0][1]))


@testing.requires_pilot
class UndoStepsTC(unittest.TestCase):

    def steps(self, command, ok=True, value=None):
        result = agent.CommandResult(command["op"], ok, value=value)
        return _agent_turn.undo_steps(command, result)

    def test_a_new_shape_is_one_step(self):
        self.assertEqual(self.steps({"op": "add_circle"}, value={
            "shape_id": 3}), 1)

    def test_bare_geometry_records_nothing(self):
        self.assertEqual(self.steps({"op": "add_point"}, value={
            "npoint": 1}), 0)
        self.assertEqual(self.steps({"op": "add_segment"}, value={
            "nsegment": 1}), 0)

    def test_a_failed_command_records_nothing(self):
        self.assertEqual(self.steps({"op": "remove_shape"}, ok=False), 0)

    def test_clearing_the_canvas_leaves_nothing_to_undo(self):
        circle = {"op": "add_circle"}
        made = agent.CommandResult("add_circle", True, value={"shape_id": 1})
        cleared = agent.CommandResult("clear", True)
        self.assertEqual(_agent_turn.undo_turn_steps([(circle, made)]), 1)
        self.assertEqual(_agent_turn.undo_turn_steps(
            [(circle, made), ({"op": "clear"}, cleared)]), 0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
