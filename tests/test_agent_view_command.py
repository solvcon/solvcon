# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The Agent View suite: schema documents and the Executor run against a real
ViewTransform2dFp64. The lifecycle half is covered in
test_agent_window_command.py."""

import math
import unittest

import jsonschema

import solvcon
from solvcon.agent import Command, CommandError, CommandProcessor
from solvcon.agent.window import view
from solvcon.agent.window.view import (
    Executor,
    command_schemas,
    commands,
    commands_by_category,
    result_schemas,
    schema,
    tool_definitions,
    validate_command,
    validate_script,
)


# Above double-precision noise, below any value used here.
EPS = 1e-9


class SchemaDocumentTC(unittest.TestCase):
    """The schema documents themselves are well-formed and consistent."""

    def test_schema_is_derived_from_the_commands(self):
        self.assertEqual(len(schema["oneOf"]), len(commands))
        grouped = commands_by_category()
        self.assertIn("get_view", grouped["read"])
        self.assertIn("pan", grouped["update"])
        self.assertIn("reset_view", grouped["update"])

    def test_command_schemas_are_valid_and_closed(self):
        for op, cmd_schema in command_schemas.items():
            jsonschema.Draft202012Validator.check_schema(cmd_schema)
            self.assertEqual(cmd_schema["properties"]["op"]["const"], op)
            self.assertFalse(cmd_schema["additionalProperties"])

    def test_combined_schema_is_valid(self):
        jsonschema.Draft202012Validator.check_schema(schema)

    def test_every_command_is_registered_and_applies(self):
        self.assertEqual(set(commands), set(command_schemas))
        self.assertEqual(set(commands), set(result_schemas))
        for op, cmd in commands.items():
            self.assertEqual(cmd.op, op)
            self.assertIsNot(type(cmd).apply, Command.apply)

    def test_tool_definitions_drop_op_and_carry_output_schema(self):
        tools = {t["name"]: t for t in tool_definitions()}
        self.assertEqual(set(tools), set(command_schemas))
        for op, tool in tools.items():
            self.assertNotIn("op", tool["inputSchema"]["properties"])
            self.assertNotIn("op", tool["inputSchema"]["required"])
            self.assertEqual(tool["outputSchema"], result_schemas[op])

    def test_result_schemas_are_valid_and_closed(self):
        self.assertEqual(set(result_schemas), set(command_schemas))
        for res_schema in result_schemas.values():
            jsonschema.Draft202012Validator.check_schema(res_schema)
            self.assertFalse(res_schema["additionalProperties"])
            self.assertEqual(set(res_schema["required"]),
                             set(res_schema["properties"]))


class JsonConformanceTC(unittest.TestCase):
    """A few pure-JSON cases pin what the schema accepts and rejects."""

    def test_valid_command_passes(self):
        command = {"op": "zoom_at", "factor": 2.0,
                   "anchor_screen_x": 0.0, "anchor_screen_y": 0.0}
        self.assertIs(validate_command(command), command)
        jsonschema.validate(command, schema,
                            cls=jsonschema.Draft202012Validator)

    def test_non_positive_zoom_is_rejected(self):
        with self.assertRaises(CommandError):
            validate_command({"op": "set_view", "pan_x": 0, "pan_y": 0,
                              "zoom": 0})
        with self.assertRaises(CommandError):
            validate_command({"op": "zoom_at", "factor": -1.0,
                              "anchor_screen_x": 0, "anchor_screen_y": 0})

    def test_unknown_op_and_extra_key_are_rejected(self):
        with self.assertRaises(CommandError):
            validate_command({"op": "fly"})
        with self.assertRaises(CommandError):
            validate_command({"op": "reset_view", "extra": 1})

    def test_script_validation(self):
        script = [{"op": "reset_view"},
                  {"op": "pan", "dx_screen": 1.0, "dy_screen": 2.0}]
        self.assertIs(validate_script(script), script)
        with self.assertRaises(CommandError):
            validate_script([{"op": "reset_view"}, {"op": "fly"}])
        with self.assertRaises(CommandError):
            validate_script({"op": "reset_view"})


class ViewExecutorTC(unittest.TestCase):
    """Application runs validated commands against a real view transform."""

    def setUp(self):
        self.view = solvcon.ViewTransform2dFp64()
        self.ex = Executor(self.view)

    def test_get_view_reports_identity(self):
        got = self.ex.run({"op": "get_view"})
        self.assertTrue(got.ok)
        self.assertEqual(got.value["view"],
                         {"pan_x": 0.0, "pan_y": 0.0, "zoom": 1.0})

    def test_set_view_writes_through_to_the_transform(self):
        res = self.ex.run(
            {"op": "set_view", "pan_x": 10.0, "pan_y": 20.0, "zoom": 4.0})
        self.assertTrue(res.ok)
        self.assertEqual(self.view.pan_x, 10.0)
        self.assertEqual(self.view.pan_y, 20.0)
        self.assertEqual(self.view.zoom, 4.0)
        self.assertEqual(self.ex.run({"op": "get_view"}).value["view"],
                         {"pan_x": 10.0, "pan_y": 20.0, "zoom": 4.0})

    def test_pan_translates_the_view_in_screen_space(self):
        self.assertTrue(self.ex.run(
            {"op": "pan", "dx_screen": 100.0, "dy_screen": 50.0}).ok)
        self.assertEqual(self.view.pan_x, 100.0)
        self.assertEqual(self.view.pan_y, 50.0)

    def test_zoom_at_keeps_the_anchor_world_point_fixed(self):
        self.ex.run({"op": "pan", "dx_screen": 120.0, "dy_screen": 80.0})
        before = self.view.world_from_screen(200.0, 175.0)
        self.assertTrue(self.ex.run(
            {"op": "zoom_at", "factor": 1.5,
             "anchor_screen_x": 200.0, "anchor_screen_y": 175.0}).ok)
        after = self.view.world_from_screen(200.0, 175.0)
        self.assertAlmostEqual(before[0], after[0], delta=EPS)
        self.assertAlmostEqual(before[1], after[1], delta=EPS)

    def test_zoom_at_clamped_honors_bounds(self):
        self.ex.run({"op": "set_view", "pan_x": 0, "pan_y": 0, "zoom": 10.0})
        self.assertTrue(self.ex.run(
            {"op": "zoom_at_clamped", "factor": 5.0, "anchor_screen_x": 0.0,
             "anchor_screen_y": 0.0, "min_zoom": 0.1, "max_zoom": 20.0}).ok)
        self.assertEqual(self.view.zoom, 20.0)

    def test_zoom_at_clamped_rejects_inverted_bounds(self):
        # max_zoom < min_zoom would no-op in C++ but wrongly report success;
        # the command rejects it as a clean failure and leaves zoom alone.
        self.ex.run({"op": "set_view", "pan_x": 0, "pan_y": 0, "zoom": 5.0})
        res = self.ex.run(
            {"op": "zoom_at_clamped", "factor": 2.0, "anchor_screen_x": 0.0,
             "anchor_screen_y": 0.0, "min_zoom": 10.0, "max_zoom": 1.0})
        self.assertFalse(res.ok)
        self.assertIn("max_zoom", res.error)
        self.assertEqual(self.view.zoom, 5.0)

    def test_reset_view_returns_to_identity(self):
        self.ex.run({"op": "set_view", "pan_x": 5, "pan_y": 7, "zoom": 3})
        self.assertTrue(self.ex.run({"op": "reset_view"}).ok)
        self.assertEqual(self.view.pan_x, 0.0)
        self.assertEqual(self.view.pan_y, 0.0)
        self.assertEqual(self.view.zoom, 1.0)

    def test_coordinate_maps_round_trip(self):
        self.ex.run({"op": "set_view", "pan_x": 10, "pan_y": 20, "zoom": 4})
        screen = self.ex.run(
            {"op": "screen_from_world", "world_x": 2.0, "world_y": 3.0})
        # screen_x = zoom*wx + pan_x = 18; screen_y = pan_y - zoom*wy = 8.
        self.assertEqual((screen.value["screen_x"], screen.value["screen_y"]),
                         (18.0, 8.0))
        world = self.ex.run(
            {"op": "world_from_screen", "screen_x": 18.0, "screen_y": 8.0})
        self.assertAlmostEqual(world.value["world_x"], 2.0, delta=EPS)
        self.assertAlmostEqual(world.value["world_y"], 3.0, delta=EPS)

    def test_bad_command_fails_cleanly_without_touching_the_view(self):
        bad = self.ex.run(
            {"op": "set_view", "pan_x": 0, "pan_y": 0, "zoom": -1})
        self.assertFalse(bad.ok)
        self.assertEqual(self.view.zoom, 1.0)

    def test_results_match_declared_schemas(self):
        script = [
            {"op": "set_view", "pan_x": 1, "pan_y": 2, "zoom": 3},
            {"op": "get_view"},
            {"op": "screen_from_world", "world_x": 0, "world_y": 0},
            {"op": "world_from_screen", "screen_x": 0, "screen_y": 0},
            {"op": "pan", "dx_screen": 1, "dy_screen": 1},
            {"op": "zoom_at", "factor": 2, "anchor_screen_x": 0,
             "anchor_screen_y": 0},
            {"op": "zoom_at_clamped", "factor": 2, "anchor_screen_x": 0,
             "anchor_screen_y": 0, "min_zoom": 0.1, "max_zoom": 100},
            {"op": "reset_view"},
        ]
        for command in script:
            with self.subTest(op=command["op"]):
                res = self.ex.run(command)
                self.assertTrue(res.ok, res.error)
                jsonschema.validate(
                    res.value, result_schemas[command["op"]],
                    cls=jsonschema.Draft202012Validator)

    def test_run_script_walks_the_vocabulary(self):
        results = self.ex.run_script([
            {"op": "pan", "dx_screen": 30.0, "dy_screen": 10.0},
            {"op": "zoom_at", "factor": 2.0, "anchor_screen_x": 0.0,
             "anchor_screen_y": 0.0},
            {"op": "get_view"},
            {"op": "reset_view"},
            {"op": "get_view"},
        ])
        self.assertTrue(all(r.ok for r in results),
                        [r.error for r in results])
        self.assertEqual(results[2].value["view"]["zoom"], 2.0)
        self.assertEqual(results[-1].value["view"],
                         {"pan_x": 0.0, "pan_y": 0.0, "zoom": 1.0})

    def test_the_module_doubles_as_a_command_set(self):
        # The module delegates its command API to the set, so a bare
        # CommandProcessor drives it like the draw module.
        proc = CommandProcessor(self.view, view)
        self.assertTrue(proc.run(
            {"op": "zoom_at", "factor": math.pi, "anchor_screen_x": 0.0,
             "anchor_screen_y": 0.0}).ok)
        self.assertAlmostEqual(self.view.zoom, math.pi, delta=EPS)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
