# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""A basic go-through of the Agent Draw command vocabulary in
solvcon.agent.draw, exercised end to end against a real WorldFp64 through the
generic CommandProcessor. Only a representative slice is covered here; the
comprehensive per-command suite lives in test_agent_draw_schema.py."""

import unittest

import solvcon
from solvcon.agent import CommandProcessor
from solvcon.agent import draw


class DrawVocabularyGoThroughTC(unittest.TestCase):
    """Walk the draw family once: schema derivation, a CRUD flow, errors."""

    def setUp(self):
        self.world = solvcon.WorldFp64()
        self.proc = CommandProcessor(self.world, draw)

    def test_schema_is_derived_from_the_commands(self):
        # The family speaks for itself; nothing enumerates ops by hand.
        self.assertEqual(len(draw.schema["oneOf"]),
                         len(draw.commands))
        grouped = draw.commands_by_category()
        self.assertIn("add_circle", grouped["create"])
        self.assertIn("translate_shape", grouped["update"])
        self.assertIn("clear", grouped["delete"])
        tool = next(t for t in draw.tool_definitions()
                    if t["name"] == "add_circle")
        self.assertIn("r", tool["inputSchema"]["properties"])
        self.assertIn("shape_id", tool["outputSchema"]["properties"])

    def test_crud_flow_mutates_the_world(self):
        created = self.proc.run(
            {"op": "add_circle", "cx": 0.0, "cy": 0.0, "r": 2.0})
        self.assertTrue(created.ok)
        shape_id = created.value["shape_id"]
        # A read reports the type the create just made, so the id is live.
        self.assertEqual(
            self.proc.run(
                {"op": "shape_type_of", "shape_id": shape_id}).value,
            {"type": "circle"})
        self.assertEqual(self.proc.run({"op": "nshape"}).value["nshape"], 1)
        self.assertTrue(self.proc.run(
            {"op": "translate_shape",
             "shape_id": shape_id, "dx": 1.0, "dy": 1.0}).ok)
        # The delete is real: the shape count falls back to zero.
        self.assertTrue(
            self.proc.run({"op": "remove_shape", "shape_id": shape_id}).ok)
        self.assertEqual(self.proc.run({"op": "nshape"}).value["nshape"], 0)

    def test_bad_commands_fail_cleanly_without_touching_the_world(self):
        # A schema violation is a failed result, not a raise or a mutation.
        bad = self.proc.run(
            {"op": "add_circle", "cx": 0.0, "cy": 0.0, "r": -1.0})
        self.assertFalse(bad.ok)
        self.assertEqual(self.proc.run({"op": "nshape"}).value["nshape"], 0)
        # A by-id op on a dead shape reports the one clean error, not a
        # C++-flavored exception.
        missing = self.proc.run(
            {"op": "translate_shape", "shape_id": 999, "dx": 0.0, "dy": 0.0})
        self.assertFalse(missing.ok)
        self.assertIn("no live shape", missing.error)

    def test_render_png_without_a_renderer_is_a_clean_failure(self):
        # render_png never touches a GUI; with no injected renderer it fails
        # rather than raising, so a batch is not aborted.
        result = self.proc.run(
            {"op": "render_png", "width": 8, "height": 8})
        self.assertFalse(result.ok)
        self.assertIn("renderer", result.error)

    def test_log_records_a_note_in_the_processor_log(self):
        self.assertTrue(self.proc.run(
            {"op": "log", "message": "hello"}).ok)
        self.assertEqual(self.proc.log, ["hello"])


class DrawPrimitiveCommandsTC(unittest.TestCase):
    """The polyline, polygon, and text commands end to end."""

    def setUp(self):
        self.world = solvcon.WorldFp64()
        self.proc = CommandProcessor(self.world, draw)

    def test_polyline_and_polygon_create_shapes(self):
        line = self.proc.run(
            {"op": "add_polyline", "vertices": [[0, 0], [1, 0], [1, 1]]})
        self.assertTrue(line.ok)
        self.assertEqual(
            self.proc.run({"op": "shape_type_of",
                           "shape_id": line.value["shape_id"]}).value,
            {"type": "polyline"})
        poly = self.proc.run(
            {"op": "add_polygon", "vertices": [[0, 0], [2, 0], [2, 2]]})
        self.assertTrue(poly.ok)
        self.assertEqual(
            self.proc.run({"op": "shape_type_of",
                           "shape_id": poly.value["shape_id"]}).value,
            {"type": "polygon"})

    def test_polygon_too_short_is_a_schema_failure(self):
        short = self.proc.run(
            {"op": "add_polygon", "vertices": [[0, 0], [1, 1]]})
        self.assertFalse(short.ok)
        self.assertEqual(self.proc.run({"op": "nshape"}).value["nshape"], 0)

    def test_text_round_trips_through_describe_state(self):
        made = self.proc.run(
            {"op": "add_text", "text": "SOLVCON",
             "x": -5.0, "y": -5.0, "height": 2.0})
        self.assertTrue(made.ok)
        shape = self.proc.run(
            {"op": "get_shape",
             "shape_id": made.value["shape_id"]}).value["shape"]
        self.assertEqual(shape["type"], "text")
        self.assertEqual(shape["text"], "SOLVCON")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
