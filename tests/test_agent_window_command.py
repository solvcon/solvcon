# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The Agent Window lifecycle suite: schema documents and the Executor run
against an in-memory window manager. The view-transform half is covered in
test_agent_view_command.py."""

import unittest

import jsonschema

from solvcon.agent import Command, CommandError, CommandProcessor
from solvcon.agent import window
from solvcon.agent.window import (
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


class FakeWindowManager:
    """In-memory window manager for headless tests; ``save_image`` records
    the request instead of writing a file."""

    def __init__(self):
        self._next_id = 1
        self._windows = []
        self._active = None
        self.saved = []

    def new_canvas(self):
        window_id = self._next_id
        self._next_id += 1
        self._windows.append({"id": window_id, "title": "2D canvas"})
        self._active = window_id
        return window_id

    def list_windows(self):
        return [{"id": w["id"], "title": w["title"],
                 "active": w["id"] == self._active}
                for w in self._windows]

    def activate_window(self, window_id):
        self._active = window_id

    def close_window(self, window_id):
        self._windows = [w for w in self._windows if w["id"] != window_id]
        if self._active == window_id:
            self._active = self._windows[-1]["id"] if self._windows else None

    def save_image(self, window_id, path):
        self.saved.append((window_id, path))


class SchemaDocumentTC(unittest.TestCase):
    """The schema documents themselves are well-formed and consistent."""

    def test_schema_is_derived_from_the_commands(self):
        self.assertEqual(len(schema["oneOf"]), len(commands))
        grouped = commands_by_category()
        self.assertIn("new_canvas", grouped["create"])
        self.assertIn("list_windows", grouped["read"])
        self.assertIn("save_image", grouped["update"])
        self.assertIn("activate_window", grouped["update"])
        self.assertIn("close_window", grouped["delete"])

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
        command = {"op": "save_image", "window_id": 1, "path": "out.png"}
        self.assertIs(validate_command(command), command)
        jsonschema.validate(command, schema,
                            cls=jsonschema.Draft202012Validator)

    def test_wrong_type_is_rejected(self):
        with self.assertRaises(CommandError):
            validate_command({"op": "activate_window", "window_id": "1"})
        with self.assertRaises(CommandError):
            validate_command({"op": "save_image", "window_id": 1, "path": 2})

    def test_missing_required_is_rejected(self):
        with self.assertRaises(CommandError):
            validate_command({"op": "save_image", "window_id": 1})

    def test_unknown_op_and_extra_key_are_rejected(self):
        with self.assertRaises(CommandError):
            validate_command({"op": "minimize_window", "window_id": 1})
        with self.assertRaises(CommandError):
            validate_command({"op": "new_canvas", "extra": 1})

    def test_script_validation(self):
        script = [{"op": "new_canvas"},
                  {"op": "activate_window", "window_id": 1}]
        self.assertIs(validate_script(script), script)
        with self.assertRaises(CommandError):
            validate_script([{"op": "new_canvas"}, {"op": "fly"}])
        with self.assertRaises(CommandError):
            validate_script({"op": "new_canvas"})


class ExecutorTC(unittest.TestCase):
    """Application runs validated commands against a fake window manager."""

    def setUp(self):
        self.manager = FakeWindowManager()
        self.ex = Executor(self.manager)

    def test_new_canvas_opens_and_activates(self):
        res = self.ex.run({"op": "new_canvas"})
        self.assertTrue(res.ok)
        self.assertEqual(res.value["window_id"], 1)
        windows = self.ex.run({"op": "list_windows"}).value["windows"]
        self.assertEqual(len(windows), 1)
        self.assertTrue(windows[0]["active"])
        self.assertEqual(windows[0]["title"], "2D canvas")

    def test_list_windows_reflects_the_active_one(self):
        self.ex.run({"op": "new_canvas"})
        self.ex.run({"op": "new_canvas"})
        windows = self.ex.run({"op": "list_windows"}).value["windows"]
        active = [w["id"] for w in windows if w["active"]]
        self.assertEqual(active, [2])

    def test_activate_window_switches_the_active_one(self):
        self.ex.run({"op": "new_canvas"})
        self.ex.run({"op": "new_canvas"})
        self.assertTrue(
            self.ex.run({"op": "activate_window", "window_id": 1}).ok)
        windows = self.ex.run({"op": "list_windows"}).value["windows"]
        active = [w["id"] for w in windows if w["active"]]
        self.assertEqual(active, [1])

    def test_close_window_removes_it(self):
        self.ex.run({"op": "new_canvas"})
        self.ex.run({"op": "new_canvas"})
        self.assertTrue(
            self.ex.run({"op": "close_window", "window_id": 1}).ok)
        ids = [w["id"]
               for w in self.ex.run({"op": "list_windows"}).value["windows"]]
        self.assertEqual(ids, [2])

    def test_save_image_routes_to_the_manager(self):
        self.ex.run({"op": "new_canvas"})
        res = self.ex.run(
            {"op": "save_image", "window_id": 1, "path": "canvas.png"})
        self.assertTrue(res.ok)
        self.assertEqual(self.manager.saved, [(1, "canvas.png")])

    def test_save_image_reports_a_failed_write(self):
        class FailingManager(FakeWindowManager):
            def save_image(self, window_id, path):
                super().save_image(window_id, path)
                return False

        manager = FailingManager()
        ex = Executor(manager)
        ex.run({"op": "new_canvas"})
        res = ex.run({"op": "save_image", "window_id": 1, "path": "x.png"})
        self.assertFalse(res.ok)
        self.assertIn("x.png", res.error)

    def test_by_id_commands_fail_cleanly_on_a_missing_window(self):
        for op in ("activate_window", "close_window"):
            with self.subTest(op=op):
                res = self.ex.run({"op": op, "window_id": 99})
                self.assertFalse(res.ok)
                self.assertIn("99", res.error)
        res = self.ex.run(
            {"op": "save_image", "window_id": 99, "path": "x.png"})
        self.assertFalse(res.ok)
        self.assertEqual(self.manager.saved, [])

    def test_results_match_declared_schemas(self):
        self.ex.run({"op": "new_canvas"})
        script = [
            {"op": "new_canvas"},
            {"op": "list_windows"},
            {"op": "activate_window", "window_id": 1},
            {"op": "save_image", "window_id": 1, "path": "a.png"},
            {"op": "close_window", "window_id": 1},
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
            {"op": "new_canvas"},
            {"op": "new_canvas"},
            {"op": "activate_window", "window_id": 1},
            {"op": "save_image", "window_id": 2, "path": "b.png"},
            {"op": "close_window", "window_id": 2},
            {"op": "list_windows"},
        ])
        self.assertTrue(all(r.ok for r in results),
                        [r.error for r in results])
        ids = [w["id"] for w in results[-1].value["windows"]]
        self.assertEqual(ids, [1])

    def test_the_module_doubles_as_a_command_set(self):
        # The module delegates its command API to the set, so a bare
        # CommandProcessor drives it like the draw module.
        proc = CommandProcessor(self.manager, window)
        self.assertTrue(proc.run({"op": "new_canvas"}).ok)
        self.assertEqual(len(self.manager.list_windows()), 1)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
