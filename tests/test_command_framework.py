# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""A basic go-through of the generic command framework in
solvcon.agent._command, exercised through a small demo family over a plain
target so the base Command, CommandSet, and CommandProcessor are covered
independently of the drawing family and the C++ World. The comprehensive
suite lands in a later change."""

import unittest

from solvcon import agent


class _Bag:
    """A plain mutable target the demo commands operate on."""

    def __init__(self):
        self.items = []
        self.style = None


class AddItem(agent.Command):
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


class Count(agent.Command):
    op = "count"
    category = "read"
    summary = "Count the items."
    returns = {"n": {"type": "integer", "description": "Item count."}}

    def apply(self, target, args, ctx):
        return {"n": len(target.items)}


class ClearBag(agent.Command):
    op = "clear"
    category = "delete"
    summary = "Remove every item."

    def apply(self, target, args, ctx):
        target.items.clear()
        return {}


class SetStyle(agent.Command):
    op = "style"
    category = "update"
    summary = "Set nested style options."
    arguments = {"style": {
        "type": "object", "description": "Nested style options.",
        "default": {"stroke": {"color": "black", "width": 1}}}}
    optional = ("style",)

    def apply(self, target, args, ctx):
        target.style = args["style"]
        return {}


class Boom(agent.Command):
    op = "boom"
    category = "update"
    summary = "Raise to exercise error capture."

    def apply(self, target, args, ctx):
        raise RuntimeError("boom")


def _family():
    family = agent.CommandSet("demo command",
                              "Any single command in the demo set.")
    for cls in (AddItem, Count, ClearBag, SetStyle, Boom):
        family.register(cls)
    return family


class CommandFrameworkGoThroughTC(unittest.TestCase):
    """Walk the framework once end to end on the demo family."""

    def setUp(self):
        self.family = _family()

    def test_schema_is_derived_from_registered_commands(self):
        ops = {"add_item", "count", "clear", "style", "boom"}
        self.assertEqual(set(self.family.commands), ops)
        self.assertEqual(set(self.family.command_schemas), ops)
        self.assertEqual(len(self.family.schema["oneOf"]), len(ops))
        grouped = self.family.commands_by_category()
        self.assertEqual(list(grouped), list(agent.CRUD_CATEGORIES))
        self.assertIn("add_item", grouped["create"])

    def test_validation_accepts_good_and_rejects_bad(self):
        command = {"op": "add_item", "name": "x"}
        self.assertIs(self.family.validate_command(command), command)
        with self.assertRaises(agent.CommandError):
            self.family.validate_command({"op": "add_item"})
        with self.assertRaises(agent.CommandError):
            self.family.validate_command({"op": "nope"})

    def test_apply_defaults_fills_omitted_optional(self):
        filled = self.family.apply_defaults({"op": "add_item", "name": "x"})
        self.assertEqual(filled["tag"], "none")

    def test_executor_dispatches_and_applies_defaults(self):
        ex = agent.CommandProcessor(_Bag(), self.family)
        res = ex.run({"op": "add_item", "name": "x"})
        self.assertIsInstance(res, agent.CommandResult)
        self.assertTrue(res.ok)
        self.assertEqual(res.value, {"count": 1})

    def test_executor_captures_a_validation_failure(self):
        ex = agent.CommandProcessor(_Bag(), self.family)
        res = ex.run({"op": "add_item"})
        self.assertFalse(res.ok)
        self.assertEqual(res.op, "add_item")

    def test_run_script_applies_in_order(self):
        ex = agent.CommandProcessor(_Bag(), self.family)
        results = ex.run_script([
            {"op": "clear"},
            {"op": "add_item", "name": "a"},
            {"op": "add_item", "name": "b"},
            {"op": "count"},
        ])
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual(results[-1].value["n"], 2)

    def test_apply_defaults_merges_nested_dict_default(self):
        filled = self.family.apply_defaults(
            {"op": "style", "style": {"stroke": {"width": 2}}})
        self.assertEqual(filled["style"],
                         {"stroke": {"color": "black", "width": 2}})

    def test_register_accepts_an_instance(self):
        family = agent.CommandSet("demo", "Demo.")
        family.register(AddItem())
        self.assertIn("add_item", family.commands)

    def test_family_specific_categories(self):
        family = agent.CommandSet("pilot", "Pilot ops.",
                                  categories=("action", "view"))

        class Screenshot(agent.Command):
            op = "screenshot"
            category = "view"
            summary = "Grab the viewport."

            def apply(self, target, args, ctx):
                return {}

        family.register(Screenshot)
        self.assertEqual(list(family.commands_by_category()),
                         ["action", "view"])
        with self.assertRaises(ValueError):
            family.register(AddItem)

    def test_command_from_tool_call_rebuilds_the_command(self):
        command = self.family.command_from_tool_call(
            "add_item", {"name": "x"})
        self.assertEqual(command, {"op": "add_item", "name": "x"})
        self.assertIs(self.family.validate_command(command), command)

    def test_run_script_stop_on_error_halts(self):
        bag = _Bag()
        ex = agent.CommandProcessor(bag, self.family)
        results = ex.run_script([
            {"op": "add_item", "name": "a"},
            {"op": "boom"},
            {"op": "add_item", "name": "b"},
        ], stop_on_error=True)
        self.assertEqual([r.ok for r in results], [True, False])
        self.assertEqual(len(bag.items), 1)

    def test_reraise_propagates_unexpected_exceptions(self):
        ex = agent.CommandProcessor(_Bag(), self.family, reraise=True)
        with self.assertRaises(RuntimeError):
            ex.run({"op": "boom"})
        res = ex.run({"op": "add_item"})
        self.assertFalse(res.ok)


class CommandDispatcherTC(unittest.TestCase):
    """Route ops across two families with distinct targets."""

    def setUp(self):
        self.bag = _Bag()
        pilot = agent.CommandSet("pilot", "Pilot ops.", categories=("action",))

        class Ping(agent.Command):
            op = "ping"
            category = "action"
            summary = "Acknowledge."
            returns = {"pong": {"type": "boolean",
                                "description": "Always true."}}

            def apply(self, target, args, ctx):
                return {"pong": True}

        pilot.register(Ping)
        self.dispatcher = agent.CommandDispatcher([
            agent.CommandProcessor(self.bag, _family()),
            agent.CommandProcessor(object(), pilot),
        ])

    def test_routes_each_op_to_its_family(self):
        self.assertTrue(self.dispatcher.run(
            {"op": "add_item", "name": "x"}).ok)
        self.assertEqual(len(self.bag.items), 1)
        self.assertEqual(self.dispatcher.run({"op": "ping"}).value,
                         {"pong": True})

    def test_unknown_op_is_a_failed_result(self):
        res = self.dispatcher.run({"op": "nope"})
        self.assertFalse(res.ok)
        self.assertIn("ping", res.error)

    def test_tool_surface_is_the_concatenation(self):
        names = [t["name"] for t in self.dispatcher.tool_definitions()]
        self.assertEqual(sorted(names),
                         ["add_item", "boom", "clear", "count", "ping",
                          "style"])

    def test_duplicate_op_across_families_is_rejected(self):
        with self.assertRaises(ValueError):
            agent.CommandDispatcher([
                agent.CommandProcessor(_Bag(), _family()),
                agent.CommandProcessor(_Bag(), _family())])

    def test_run_script_across_families(self):
        results = self.dispatcher.run_script([
            {"op": "add_item", "name": "a"},
            {"op": "ping"},
            {"op": "count"},
        ])
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual(results[-1].value["n"], 1)
        with self.assertRaises(agent.CommandError):
            self.dispatcher.run_script([{"op": "nope"}])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
