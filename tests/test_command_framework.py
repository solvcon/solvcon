# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests for the generic command framework in solvcon.agent._command,
exercised through a small demo family over a plain target so the base
Command, CommandSet, and CommandExecutor are covered independently of the
drawing family and the C++ World."""

import unittest

import jsonschema

from solvcon.agent import (
    Command,
    CommandSet,
    CommandError,
    CommandResult,
    CommandExecutor,
    CRUD_CATEGORIES,
)


class _Bag:
    """A plain mutable target the demo commands operate on."""

    def __init__(self):
        self.items = []
        self.opts = None


class AddItem(Command):
    op = "add_item"
    category = "create"
    summary = "Add a named item, with an optional tag."
    arguments = {"name": {"type": "string", "description": "Item name."},
                 "tag": {"type": "string", "default": "none",
                         "description": "Optional tag."}}
    optional = ("tag",)
    returns = {"count": {"type": "integer",
                         "description": "Items after the add."}}

    def apply(self, target, args, ctx):
        target.items.append((args["name"], args["tag"]))
        return {"count": len(target.items)}


class Count(Command):
    op = "count"
    category = "read"
    summary = "Count the items."
    returns = {"n": {"type": "integer", "description": "Item count."}}

    def apply(self, target, args, ctx):
        return {"n": len(target.items)}


class Configure(Command):
    op = "configure"
    category = "update"
    summary = "Store an options object."
    arguments = {"opts": {"type": "object",
                          "properties": {"a": {"type": "integer"},
                                         "b": {"type": "integer"}},
                          "additionalProperties": False,
                          "default": {"a": 1, "b": 2},
                          "description": "Options to store."}}
    optional = ("opts",)

    def apply(self, target, args, ctx):
        target.opts = args["opts"]
        return {}


class ClearBag(Command):
    op = "clear"
    category = "delete"
    summary = "Remove every item."

    def apply(self, target, args, ctx):
        target.items.clear()
        return {}


class Note(Command):
    op = "note"
    category = "log"
    summary = "Record a free-text note in the command log."
    arguments = {"text": {"type": "string", "description": "Note text."}}

    def apply(self, target, args, ctx):
        ctx.append_log(args["text"])
        return {}


class BadResult(Command):
    op = "bad_result"
    category = "read"
    summary = "Return a value that violates its own result schema."
    returns = {"value": {"type": "integer", "description": "Should be int."}}

    def apply(self, target, args, ctx):
        return {"value": "not-an-int"}


class Boom(Command):
    op = "boom"
    category = "read"
    summary = "Raise a plain (non-CommandError) exception."

    def apply(self, target, args, ctx):
        raise RuntimeError("kaboom")


class Reject(Command):
    op = "reject"
    category = "read"
    summary = "Raise a CommandError from apply."

    def apply(self, target, args, ctx):
        raise CommandError("nope")


_DEMO_COMMANDS = (AddItem, Count, Configure, ClearBag, Note, BadResult,
                  Boom, Reject)


def _family():
    """A fresh demo family; each test gets its own so registration on one
    never leaks into another."""
    family = CommandSet("demo command", "Any single command in the demo set.")
    for cls in _DEMO_COMMANDS:
        family.register(cls)
    return family


class RegistrationTC(unittest.TestCase):
    """The register guards reject a malformed command at declaration time."""

    def setUp(self):
        self.family = CommandSet("t", "t")

    def test_rejects_empty_op(self):
        class NoOp(Command):
            category = "read"
        with self.assertRaises(ValueError):
            self.family.register(NoOp)

    def test_rejects_duplicate_op(self):
        self.family.register(Count)
        with self.assertRaises(ValueError):
            self.family.register(Count)

    def test_rejects_unknown_category(self):
        class Weird(Command):
            op = "weird"
            category = "frobnicate"
        with self.assertRaises(ValueError):
            self.family.register(Weird)

    def test_rejects_optional_that_is_not_an_argument(self):
        class Ghost(Command):
            op = "ghost"
            category = "read"
            optional = ("missing",)
        with self.assertRaises(ValueError):
            self.family.register(Ghost)

    def test_rejects_defaulted_required_argument(self):
        class Defaulted(Command):
            op = "defaulted"
            category = "read"
            arguments = {"x": {"type": "integer", "default": 0}}
        with self.assertRaises(ValueError):
            self.family.register(Defaulted)

    def test_register_returns_the_class(self):
        self.assertIs(self.family.register(Count), Count)


class SchemaDerivationTC(unittest.TestCase):
    """The derived documents mirror the registered commands."""

    def setUp(self):
        self.family = _family()

    def test_op_sets_match(self):
        ops = {c.op for c in _DEMO_COMMANDS}
        self.assertEqual(set(self.family.commands), ops)
        self.assertEqual(set(self.family.command_schemas), ops)
        self.assertEqual(set(self.family.result_schemas), ops)

    def test_command_schema_shape(self):
        for op, schema in self.family.command_schemas.items():
            jsonschema.Draft202012Validator.check_schema(schema)
            self.assertEqual(schema["properties"]["op"]["const"], op)
            self.assertIn("op", schema["required"])
            self.assertFalse(schema["additionalProperties"])

    def test_result_schemas_are_closed(self):
        for schema in self.family.result_schemas.values():
            jsonschema.Draft202012Validator.check_schema(schema)
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]),
                             set(schema["properties"]))

    def test_combined_schema_carries_title_and_description(self):
        schema = self.family.schema
        jsonschema.Draft202012Validator.check_schema(schema)
        self.assertEqual(schema["title"], "demo command")
        self.assertEqual(len(schema["oneOf"]), len(_DEMO_COMMANDS))

    def test_registration_invalidates_the_cache(self):
        self.assertNotIn("late", self.family.command_schemas)

        class Late(Command):
            op = "late"
            category = "read"
        self.family.register(Late)
        self.assertIn("late", self.family.command_schemas)
        self.assertIn("late", self.family.result_schemas)

    def test_tool_definitions_drop_op_and_carry_output(self):
        tools = {t["name"]: t for t in self.family.tool_definitions()}
        self.assertEqual(set(tools), set(self.family.command_schemas))
        for op, tool in tools.items():
            self.assertNotIn("op", tool["inputSchema"]["properties"])
            self.assertNotIn("op", tool["inputSchema"]["required"])
            self.assertEqual(tool["outputSchema"],
                             self.family.result_schemas[op])

    def test_commands_by_category(self):
        grouped = self.family.commands_by_category()
        self.assertEqual(list(grouped), list(CRUD_CATEGORIES))
        self.assertIn("add_item", grouped["create"])
        self.assertIn("count", grouped["read"])
        self.assertIn("configure", grouped["update"])
        self.assertIn("clear", grouped["delete"])
        self.assertIn("note", grouped["log"])


class DefaultsTC(unittest.TestCase):
    """apply_defaults fills omitted optionals without mutating the input."""

    def setUp(self):
        self.family = _family()

    def test_fills_scalar_default(self):
        filled = self.family.apply_defaults({"op": "add_item", "name": "x"})
        self.assertEqual(filled["tag"], "none")

    def test_keeps_provided_value(self):
        filled = self.family.apply_defaults(
            {"op": "add_item", "name": "x", "tag": "hot"})
        self.assertEqual(filled["tag"], "hot")

    def test_deep_merges_dict_default(self):
        filled = self.family.apply_defaults(
            {"op": "configure", "opts": {"a": 9}})
        self.assertEqual(filled["opts"], {"a": 9, "b": 2})

    def test_does_not_mutate_input(self):
        command = {"op": "add_item", "name": "x"}
        self.family.apply_defaults(command)
        self.assertNotIn("tag", command)


class ValidationTC(unittest.TestCase):
    """validate_command/result/script accept the good and reject the bad."""

    def setUp(self):
        self.family = _family()

    def test_valid_command_returns_input(self):
        command = {"op": "add_item", "name": "x"}
        self.assertIs(self.family.validate_command(command), command)

    def test_non_object_command(self):
        with self.assertRaises(CommandError):
            self.family.validate_command(["not", "a", "dict"])

    def test_missing_string_op(self):
        with self.assertRaises(CommandError):
            self.family.validate_command({"op": 123})

    def test_unknown_op(self):
        with self.assertRaises(CommandError):
            self.family.validate_command({"op": "nope"})

    def test_missing_required_argument(self):
        with self.assertRaises(CommandError):
            self.family.validate_command({"op": "add_item"})

    def test_result_validation(self):
        self.assertEqual(self.family.validate_result("count", {"n": 3}),
                         {"n": 3})
        with self.assertRaises(CommandError):
            self.family.validate_result("count", {"n": "many"})
        with self.assertRaises(CommandError):
            self.family.validate_result("no_such_op", {})

    def test_script_validation(self):
        script = [{"op": "clear"}, {"op": "count"}]
        self.assertIs(self.family.validate_script(script), script)
        with self.assertRaises(CommandError):
            self.family.validate_script({"op": "clear"})
        with self.assertRaises(CommandError):
            self.family.validate_script([{"op": "clear"}, {"op": "nope"}])


class ExecutorTC(unittest.TestCase):
    """CommandExecutor dispatches to the target and records every step."""

    def setUp(self):
        self.bag = _Bag()
        self.ex = CommandExecutor(self.bag, _family())

    def test_run_dispatches_and_applies_defaults(self):
        res = self.ex.run({"op": "add_item", "name": "x"})
        self.assertIsInstance(res, CommandResult)
        self.assertTrue(res.ok)
        self.assertEqual(res.value, {"count": 1})
        self.assertEqual(self.bag.items[-1], ("x", "none"))

    def test_target_is_the_bound_target(self):
        self.ex.run({"op": "add_item", "name": "a"})
        self.assertEqual(self.ex.run({"op": "count"}).value["n"], 1)

    def test_log_command_and_property(self):
        self.assertTrue(self.ex.run({"op": "note", "text": "hi"}).ok)
        self.assertEqual(self.ex.log, ["hi"])

    def test_validation_error_becomes_failed_result(self):
        res = self.ex.run({"op": "add_item"})
        self.assertFalse(res.ok)
        self.assertEqual(res.op, "add_item")

    def test_non_dict_command_failed_result_has_placeholder_op(self):
        res = self.ex.run("garbage")
        self.assertFalse(res.ok)
        self.assertEqual(res.op, "?")

    def test_plain_exception_is_captured_with_type(self):
        res = self.ex.run({"op": "boom"})
        self.assertFalse(res.ok)
        self.assertTrue(res.error.startswith("RuntimeError:"))

    def test_command_error_is_captured(self):
        res = self.ex.run({"op": "reject"})
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "nope")

    def test_result_checking_off_by_default(self):
        self.assertTrue(self.ex.run({"op": "bad_result"}).ok)

    def test_result_checking_catches_off_contract_value(self):
        ex = CommandExecutor(self.bag, _family(), validate_results=True)
        res = ex.run({"op": "bad_result"})
        self.assertFalse(res.ok)
        self.assertIn("bad_result result", res.error)

    def test_run_script_applies_in_order(self):
        results = self.ex.run_script([
            {"op": "clear"},
            {"op": "add_item", "name": "a"},
            {"op": "add_item", "name": "b"},
            {"op": "count"},
        ])
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual(results[-1].value["n"], 2)

    def test_run_script_rejects_bad_script_before_running(self):
        with self.assertRaises(CommandError):
            self.ex.run_script([{"op": "add_item", "name": "a"},
                                {"op": "nope"}])
        self.assertEqual(self.bag.items, [])


class CommandBaseTC(unittest.TestCase):
    """The base Command is an unimplemented contract."""

    def test_apply_is_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Command().apply(object(), {}, None)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
