# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import json
import base64
import unittest

import jsonschema

import solvcon
from solvcon.agent.draw import (
    COMMANDS,
    COMMAND_SCHEMAS,
    CRUD_CATEGORIES,
    RESULT_SCHEMAS,
    SCHEMA,
    CommandError,
    Executor,
    commands_by_category,
    tool_definitions,
    validate_command,
    validate_script,
)


def _load_cases():
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        "data", "agentdraw_command_cases.json")
    with open(path, "r") as case_file:
        return json.load(case_file)


_CASES = _load_cases()


class SchemaDocumentTC(unittest.TestCase):
    """The schema documents themselves are well-formed and consistent."""

    def test_command_schemas_are_valid_json_schema(self):
        for op, schema in COMMAND_SCHEMAS.items():
            jsonschema.Draft202012Validator.check_schema(schema)
            self.assertEqual(schema["properties"]["op"]["const"], op)
            self.assertFalse(schema["additionalProperties"])

    def test_combined_schema_is_valid(self):
        jsonschema.Draft202012Validator.check_schema(SCHEMA)

    def test_every_command_is_registered_and_applies(self):
        # The schema documents are derived from the same registry the
        # executor dispatches through, so the op sets match by construction
        # and every command overrides apply rather than the base stub.
        self.assertEqual(set(COMMANDS), set(COMMAND_SCHEMAS))
        self.assertEqual(set(COMMANDS), set(RESULT_SCHEMAS))
        from solvcon.agent.draw import Command
        for op, cmd in COMMANDS.items():
            self.assertEqual(cmd.op, op)
            self.assertIsNot(type(cmd).apply, Command.apply)

    def test_defaulted_args_are_optional(self):
        # A property with a default that is also required would never use its
        # default; this guards the schema/apply_defaults contract.
        for op, schema in COMMAND_SCHEMAS.items():
            for name, prop in schema["properties"].items():
                if isinstance(prop, dict) and "default" in prop:
                    self.assertNotIn(name, schema["required"],
                                     f"{op}.{name} is defaulted but required")

    def test_crud_coverage(self):
        # Every command carries a known category, and all four CRUD roles
        # are represented so the vocabulary covers the shape lifecycle.
        grouped = commands_by_category()
        for schema in COMMAND_SCHEMAS.values():
            self.assertIn(schema["category"], CRUD_CATEGORIES)
        for role in ("create", "read", "update", "delete"):
            self.assertTrue(grouped[role], f"no {role} command")
        self.assertIn("get_shape", grouped["read"])
        self.assertIn("translate_shape", grouped["update"])

    def test_tool_definitions_drop_op(self):
        tools = {t["name"]: t for t in tool_definitions()}
        self.assertEqual(set(tools), set(COMMAND_SCHEMAS))
        for tool in tools.values():
            self.assertNotIn("op", tool["inputSchema"]["properties"])
            self.assertNotIn("op", tool["inputSchema"]["required"])

    def test_result_schemas_are_valid_and_closed(self):
        self.assertEqual(set(RESULT_SCHEMAS), set(COMMAND_SCHEMAS))
        for op, schema in RESULT_SCHEMAS.items():
            jsonschema.Draft202012Validator.check_schema(schema)
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]),
                             set(schema["properties"]))

    def test_tool_definitions_carry_output_schema(self):
        for tool in tool_definitions():
            self.assertEqual(tool["outputSchema"],
                             RESULT_SCHEMAS[tool["name"]])


class JsonConformanceTC(unittest.TestCase):
    """Pure-JSON cases pin exactly what the schema accepts and rejects."""

    def test_valid_cases(self):
        for case in _CASES["valid"]:
            with self.subTest(case=case["name"]):
                self.assertIs(validate_command(case["command"]),
                              case["command"])
                jsonschema.validate(case["command"], SCHEMA,
                                    cls=jsonschema.Draft202012Validator)

    def test_invalid_cases(self):
        for case in _CASES["invalid"]:
            with self.subTest(case=case["name"]):
                with self.assertRaises(CommandError):
                    validate_command(case["command"])

    def test_script_validation(self):
        script = [c["command"] for c in _CASES["valid"]]
        self.assertIs(validate_script(script), script)
        with self.assertRaises(CommandError):
            validate_script([{"op": "clear"}, {"op": "fly"}])
        with self.assertRaises(CommandError):
            validate_script({"op": "clear"})


class ExecutorTC(unittest.TestCase):
    """Application runs validated commands against a real World."""

    def setUp(self):
        self.world = solvcon.WorldFp64()
        self.ex = Executor(self.world)

    def test_add_query_describe(self):
        res = self.ex.run({"op": "add_circle", "cx": 0, "cy": 0, "r": 1})
        self.assertTrue(res.ok)
        sid = res.value["shape_id"]
        self.assertEqual(self.ex.run({"op": "nshape"}).value["nshape"], 1)
        self.assertEqual(
            self.ex.run({"op": "shape_type_of", "shape_id": sid}).
            value["type"], "circle")
        vis = self.ex.run({"op": "query_visible", "min_x": -2, "min_y": -2,
                           "max_x": 2, "max_y": 2})
        self.assertEqual(vis.value["shape_ids"], [sid])
        state = self.ex.run({"op": "describe_state"}).value["state"]
        self.assertEqual(state["shapes"][0]["type"], "circle")

    def test_crud_round_trip(self):
        # Create -> Read-one -> Update -> Read-one -> Delete -> Read-one.
        sid = self.ex.run(
            {"op": "add_rectangle", "x_min": 0, "y_min": 0, "x_max": 2,
             "y_max": 1}).value["shape_id"]
        got = self.ex.run({"op": "get_shape", "shape_id": sid})
        self.assertTrue(got.ok)
        self.assertEqual(got.value["shape"]["bbox"], [0, 0, 2, 1])
        self.assertTrue(self.ex.run(
            {"op": "translate_shape", "shape_id": sid, "dx": 3,
             "dy": 0}).ok)
        moved = self.ex.run({"op": "get_shape", "shape_id": sid})
        self.assertEqual(moved.value["shape"]["bbox"], [3, 0, 5, 1])
        self.assertTrue(
            self.ex.run({"op": "remove_shape", "shape_id": sid}).ok)
        gone = self.ex.run({"op": "get_shape", "shape_id": sid})
        self.assertFalse(gone.ok)
        self.assertIn(str(sid), gone.error)

    def test_translate_remove_clear(self):
        sid = self.ex.run(
            {"op": "add_triangle", "x0": 0, "y0": 0, "x1": 1, "y1": 0,
             "x2": 0, "y2": 1}).value["shape_id"]
        self.assertTrue(self.ex.run(
            {"op": "translate_shape", "shape_id": sid, "dx": 5,
             "dy": 0}).ok)
        self.assertTrue(
            self.ex.run({"op": "remove_shape", "shape_id": sid}).ok)
        self.ex.run({"op": "add_circle", "cx": 0, "cy": 0, "r": 1})
        self.assertTrue(self.ex.run({"op": "clear"}).ok)
        self.assertEqual(self.world.nshape, 0)

    def test_bare_primitives_and_2d_points(self):
        # A 2D point in the command is applied with an implicit zero z.
        seg = self.ex.run(
            {"op": "add_segment", "p0": [0, 0], "p1": [1, 1]})
        self.assertEqual(seg.value["nsegment"], 1)
        bez = self.ex.run(
            {"op": "add_bezier", "p0": [0, 0], "p1": [1, 0],
             "p2": [2, 1], "p3": [3, 1]})
        self.assertEqual(bez.value["nbezier"], 1)

    def test_log(self):
        self.assertTrue(self.ex.run({"op": "log", "message": "hi"}).ok)
        self.assertEqual(self.ex.log, ["hi"])

    def test_render_without_renderer_fails(self):
        res = self.ex.run({"op": "render_png", "width": 16, "height": 16})
        self.assertFalse(res.ok)
        self.assertIn("renderer", res.error)

    def test_render_with_renderer_applies_defaults(self):
        seen = {}

        def fake_renderer(world, view, width, height, antialiasing):
            seen.update(view=view, width=width, height=height,
                        aa=antialiasing)
            return b"PNGBYTES"

        ex = Executor(self.world, renderer=fake_renderer)
        res = ex.run({"op": "render_png", "width": 32, "height": 24})
        self.assertTrue(res.ok)
        # The PNG bytes arrive base64-encoded with a media type, so the result
        # is plain JSON rather than raw bytes.
        image = res.value["image"]
        self.assertEqual(base64.b64decode(image["data"]), b"PNGBYTES")
        self.assertEqual(image["mime_type"], "image/png")
        self.assertEqual((image["width"], image["height"]), (32, 24))
        jsonschema.validate(res.value, RESULT_SCHEMAS["render_png"],
                            cls=jsonschema.Draft202012Validator)
        self.assertEqual(seen["view"],
                         {"pan_x": 0.0, "pan_y": 0.0, "zoom": 1.0})
        self.assertEqual(seen["aa"], False)

    def test_render_partial_view_is_completed(self):
        # A partial view must reach the renderer complete: omitted sub-fields
        # come from the schema default, not arrive missing.
        seen = {}

        def fake_renderer(world, view, width, height, antialiasing):
            seen["view"] = view
            return b"PNG"

        ex = Executor(self.world, renderer=fake_renderer)
        res = ex.run({"op": "render_png", "width": 8, "height": 8,
                      "view": {"zoom": 3}})
        self.assertTrue(res.ok)
        self.assertEqual(seen["view"],
                         {"pan_x": 0.0, "pan_y": 0.0, "zoom": 3})

    def test_invalid_command_becomes_failed_result(self):
        # The executor records validation errors instead of raising, so a
        # harness can log every step.
        res = self.ex.run({"op": "add_circle", "cx": 0, "cy": 0})
        self.assertFalse(res.ok)
        self.assertEqual(res.op, "add_circle")

    def test_results_match_declared_schemas(self):
        # A real result from each op must validate against RESULT_SCHEMAS, so
        # the published output contract matches what the executor returns.
        sid = self.ex.run(
            {"op": "add_rectangle", "x_min": 0, "y_min": 0, "x_max": 2,
             "y_max": 1}).value["shape_id"]
        commands = [
            {"op": "add_point", "x": 0, "y": 0},
            {"op": "add_segment", "p0": [0, 0], "p1": [1, 1]},
            {"op": "add_circle", "cx": 0, "cy": 0, "r": 1},
            {"op": "add_bezier", "p0": [0, 0], "p1": [1, 0], "p2": [2, 1],
             "p3": [3, 1]},
            {"op": "get_shape", "shape_id": sid},
            {"op": "shape_type_of", "shape_id": sid},
            {"op": "nshape"},
            {"op": "query_visible", "min_x": -2, "min_y": -2, "max_x": 2,
             "max_y": 2},
            {"op": "describe_state"},
            {"op": "translate_shape", "shape_id": sid, "dx": 1, "dy": 0},
            {"op": "log", "message": "note"},
            {"op": "remove_shape", "shape_id": sid},
            {"op": "clear"},
        ]
        for command in commands:
            with self.subTest(op=command["op"]):
                res = self.ex.run(command)
                self.assertTrue(res.ok, res.error)
                jsonschema.validate(res.value, RESULT_SCHEMAS[command["op"]],
                                    cls=jsonschema.Draft202012Validator)

    def test_bad_shape_id_is_uniform_command_error(self):
        # Every by-id op fails the same clean way for a missing/dead id: a
        # CommandError naming the id, never a leaked C++ exception type.
        sid = self.ex.run({"op": "add_circle", "cx": 0, "cy": 0,
                           "r": 1}).value["shape_id"]
        self.ex.run({"op": "remove_shape", "shape_id": sid})
        for op in ("get_shape", "shape_type_of", "remove_shape"):
            with self.subTest(op=op):
                res = self.ex.run({"op": op, "shape_id": sid})
                self.assertFalse(res.ok)
                self.assertEqual(res.error, f"no live shape with id {sid}")
        moved = self.ex.run({"op": "translate_shape", "shape_id": 999,
                             "dx": 1, "dy": 0})
        self.assertFalse(moved.ok)
        self.assertEqual(moved.error, "no live shape with id 999")

    def test_validate_results_catches_off_contract_value(self):
        # With result checking on, a handler value that violates the output
        # schema becomes a failed result instead of a false success.
        ex = Executor(self.world, validate_results=True)
        good = ex.run({"op": "add_circle", "cx": 0, "cy": 0, "r": 1})
        self.assertTrue(good.ok)
        # Shadow the instance apply with an off-contract one, then restore it.
        nshape_cmd = COMMANDS["nshape"]
        nshape_cmd.apply = lambda world, args, ctx: {"nshape": "many"}
        try:
            bad = ex.run({"op": "nshape"})
        finally:
            del nshape_cmd.apply
        self.assertFalse(bad.ok)
        self.assertIn("nshape result", bad.error)

    def test_run_script(self):
        results = self.ex.run_script([
            {"op": "clear"},
            {"op": "add_circle", "cx": 0, "cy": 0, "r": 1},
            {"op": "nshape"},
        ])
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual(results[-1].value["nshape"], 1)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
