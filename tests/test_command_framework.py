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


class BadCount(agent.Command):
    op = "bad_count"
    category = "read"
    summary = "Return a value that violates the result schema."
    returns = {"n": {"type": "integer", "description": "Item count."}}

    def apply(self, target, args, ctx):
        return {"n": "not-an-integer"}


class LogNote(agent.Command):
    op = "note"
    category = "log"
    summary = "Record a note in the processor log."
    arguments = {"text": {"type": "string", "description": "Note text."}}

    def apply(self, target, args, ctx):
        ctx.append_log(args["text"])
        return {}


class CommandRegistrationTC(unittest.TestCase):
    """Guards that reject a malformed command declaration."""

    def setUp(self):
        self.family = agent.CommandSet("demo", "Demo.")

    def test_rejects_command_without_an_op(self):
        class NoOp(agent.Command):
            category = "read"

        with self.assertRaises(ValueError):
            self.family.register(NoOp)

    def test_rejects_a_duplicate_op(self):
        self.family.register(AddItem)
        with self.assertRaises(ValueError):
            self.family.register(AddItem)

    def test_rejects_optional_that_is_not_an_argument(self):
        class Ghost(agent.Command):
            op = "ghost"
            category = "read"
            optional = ("missing",)

        with self.assertRaises(ValueError):
            self.family.register(Ghost)

    def test_rejects_a_defaulted_required_argument(self):
        class Defaulted(agent.Command):
            op = "defaulted"
            category = "create"
            arguments = {"x": {"type": "integer", "default": 1}}

        with self.assertRaises(ValueError):
            self.family.register(Defaulted)


class ResultValidationTC(unittest.TestCase):
    """The processor's optional result-schema check."""

    def _family(self):
        family = agent.CommandSet("demo", "Demo.")
        family.register(BadCount)
        return family

    def test_off_by_default_lets_an_off_contract_result_through(self):
        ex = agent.CommandProcessor(_Bag(), self._family())
        self.assertTrue(ex.run({"op": "bad_count"}).ok)

    def test_on_catches_an_off_contract_result(self):
        ex = agent.CommandProcessor(_Bag(), self._family(),
                                    validate_results=True)
        res = ex.run({"op": "bad_count"})
        self.assertFalse(res.ok)
        self.assertEqual(res.op, "bad_count")

    def test_on_accepts_a_conforming_result(self):
        ex = agent.CommandProcessor(_Bag(), _family(),
                                    validate_results=True)
        self.assertTrue(ex.run({"op": "add_item", "name": "x"}).ok)


class ProcessorLogTC(unittest.TestCase):
    """The per-processor log the recording harness reads."""

    def test_append_log_accumulates_and_log_returns_a_copy(self):
        ex = agent.CommandProcessor(_Bag(), _family())
        ex.append_log("a")
        ex.append_log("b")
        snapshot = ex.log
        self.assertEqual(snapshot, ["a", "b"])
        snapshot.append("c")
        self.assertEqual(ex.log, ["a", "b"])

    def test_a_command_logs_through_its_ctx(self):
        family = agent.CommandSet("demo", "Demo.")
        family.register(LogNote)
        ex = agent.CommandProcessor(_Bag(), family)
        ex.run({"op": "note", "text": "hello"})
        self.assertEqual(ex.log, ["hello"])


class ProcessorErrorCaptureTC(unittest.TestCase):
    """A failing command becomes a failed result, never a raise."""

    def setUp(self):
        self.ex = agent.CommandProcessor(_Bag(), _family())

    def test_a_plain_exception_is_captured_with_its_type(self):
        res = self.ex.run({"op": "boom"})
        self.assertFalse(res.ok)
        self.assertIn("RuntimeError", res.error)

    def test_a_non_dict_command_fails_with_a_placeholder_op(self):
        res = self.ex.run("not a command")
        self.assertFalse(res.ok)
        self.assertEqual(res.op, "?")


class ValidationRobustnessTC(unittest.TestCase):
    """Friendly errors on structurally invalid input."""

    def setUp(self):
        self.family = _family()

    def test_a_non_object_command_is_rejected(self):
        with self.assertRaises(agent.CommandError):
            self.family.validate_command(["op", "count"])

    def test_a_command_without_a_string_op_is_rejected(self):
        with self.assertRaises(agent.CommandError):
            self.family.validate_command({"name": "x"})

    def test_a_script_that_is_not_a_list_is_rejected(self):
        with self.assertRaises(agent.CommandError):
            self.family.validate_script({"op": "count"})


class DerivedSchemaTC(unittest.TestCase):
    """Behavior of the lazily derived, cached schema surface."""

    def test_registration_invalidates_the_schema_cache(self):
        family = _family()
        self.assertNotIn("late", family.command_schemas)

        class Late(agent.Command):
            op = "late"
            category = "read"

            def apply(self, target, args, ctx):
                return {}

        family.register(Late)
        self.assertIn("late", family.command_schemas)

    def test_apply_defaults_does_not_mutate_its_input(self):
        command = {"op": "style", "style": {"stroke": {"width": 2}}}
        _family().apply_defaults(command)
        self.assertEqual(command,
                         {"op": "style", "style": {"stroke": {"width": 2}}})

    def test_tool_definitions_drop_op_and_carry_the_output_schema(self):
        tools = {t["name"]: t for t in _family().tool_definitions()}
        add = tools["add_item"]
        self.assertNotIn("op", add["inputSchema"]["properties"])
        self.assertEqual(add["category"], "create")
        self.assertIn("count", add["outputSchema"]["properties"])


class CommandBaseTC(unittest.TestCase):
    """The base command contract."""

    def test_apply_is_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            agent.Command().apply(None, {}, None)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
