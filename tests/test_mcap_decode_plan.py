# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import types
import unittest

import solvcon as sc


class DecodePlanTC(unittest.TestCase):

    SCHEMA_TEXT = """\
================================================================================
IDL: vhclsim_msgs/msg/Status
#include "vhclsim_msgs/msg/State.idl"
module vhclsim_msgs { module msg {
  enum Mode { OFF, ON };
  struct Status {
    @default (value=0.0) double dt;
    State current;
    sequence<float, 8> samples;
    string<64> label;
    Mode mode;
    int32 sim_end;
  };
}; };
================================================================================
IDL: vhclsim_msgs/msg/State
module vhclsim_msgs { module msg {
  struct State {
    double x;
    double y;
    boolean active;
  };
}; };
"""

    def schema(self, encoding="ros2idl"):
        return types.SimpleNamespace(
            name="vhclsim_msgs/msg/Status",
            encoding=encoding,
            data=self.SCHEMA_TEXT.encode(),
        )

    def test_nested_struct_fields(self):
        plan = sc.mcap.DecodePlan(
            self.schema(), fields=["dt", "current.x", "current.active"])
        self.assertEqual(plan.fields,
                         ("dt", "current.x", "current.active"))
        self.assertEqual(plan.types, ("float64", "float64", "bool"))
        self.assertEqual(
            plan.instructions,
            (
                ("align", 8), ("read", "float64", 0),
                ("align", 8), ("read", "float64", 1),
                ("align", 8), ("skip", 8),
                ("read", "bool", 2),
            ),
        )

    def test_requested_order_is_column_order(self):
        plan = sc.mcap.DecodePlan(
            self.schema(), fields=["current.active", "current.x"])
        self.assertEqual(plan.types, ("bool", "float64"))
        self.assertEqual(
            plan.instructions,
            (
                ("align", 8), ("skip", 8),
                ("align", 8), ("read", "float64", 1),
                ("align", 8), ("skip", 8),
                ("read", "bool", 0),
            ),
        )

    def test_walks_nested_struct_sequence_string_and_enum(self):
        plan = sc.mcap.DecodePlan(self.schema(), fields=["sim_end"])
        self.assertEqual(
            plan.instructions,
            (
                ("align", 8), ("skip", 8),
                ("align", 8), ("skip", 8),
                ("align", 8), ("skip", 8),
                ("skip", 1),
                ("align", 4), ("skip_sequence", "float32"),
                ("align", 4), ("skip_string",),
                ("align", 4), ("skip", 4),
                ("align", 4), ("read", "int32", 0),
            ),
        )

    def test_fixed_primitive_array(self):
        text = """\
module p { module msg { struct S {
  double matrix[4];
  boolean valid;
}; }; };
"""
        schema = types.SimpleNamespace(
            name="p/msg/S", encoding="ros2idl", data=text.encode())
        plan = sc.mcap.DecodePlan(schema, fields=["valid"])
        self.assertEqual(
            plan.instructions,
            (("align", 8), ("skip", 32), ("read", "bool", 0)),
        )

    def test_enum_reads_as_its_ordinal(self):
        plan = sc.mcap.DecodePlan(self.schema(), fields=["mode"])
        self.assertEqual(plan.types, ("uint32",))
        self.assertEqual(plan.instructions[-1], ("read", "uint32", 0))

    CONTAINER_TEXT = """\
module p { module msg {
  struct Item {
    double weight;
    string label;
  };
  struct Grid {
    sequence<Item> items;
    Item corners[4];
    sequence<string> tags;
    sequence<double> samples;
    int32 tail;
  };
}; };
"""

    def container_schema(self):
        return types.SimpleNamespace(
            name="p/msg/Grid", encoding="ros2idl",
            data=self.CONTAINER_TEXT.encode())

    def test_walks_every_container_to_reach_a_later_field(self):
        plan = sc.mcap.DecodePlan(self.container_schema(), fields=["tail"])
        self.assertEqual(
            plan.instructions,
            (
                ("align", 4), ("skip_sequence_body", 4),
                ("align", 8), ("skip", 8),
                ("align", 4), ("skip_string",),
                ("skip_array_body", 4, 4),
                ("align", 8), ("skip", 8),
                ("align", 4), ("skip_string",),
                ("align", 4), ("skip_sequence_body", 2),
                ("align", 4), ("skip_string",),
                ("align", 4), ("skip_sequence", "float64"),
                ("align", 4), ("read", "int32", 0),
            ),
        )

    def test_rejects_a_struct_that_contains_itself(self):
        text = """\
module p { module msg { struct Node {
  sequence<Node> children;
  int32 value;
}; }; };
"""
        schema = types.SimpleNamespace(
            name="p/msg/Node", encoding="ros2idl", data=text.encode())
        with self.assertRaisesRegex(sc.mcap.DecodePlanError,
                                    "contains itself"):
            sc.mcap.DecodePlan(schema, fields=["value"])

    def test_accepts_an_uppercase_field_name(self):
        text = """\
module p { module msg { struct S {
  int32 N;
  double N_min;
}; }; };
"""
        schema = types.SimpleNamespace(
            name="p/msg/S", encoding="ros2idl", data=text.encode())
        plan = sc.mcap.DecodePlan(schema, fields=["N", "N_min"])
        self.assertEqual(plan.types, ("int32", "float64"))

    def test_rejects_non_scalar_output(self):
        cases = (
            ("label", "numeric or bool scalar"),
            ("samples", "numeric or bool scalar"),
            ("current", "numeric or bool scalar"),
            ("current[0].x", "indexed or invalid"),
        )
        for field, message in cases:
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                        sc.mcap.DecodePlanError, message):
                    sc.mcap.DecodePlan(self.schema(), fields=[field])

    def test_rejects_missing_paths_and_dependencies(self):
        with self.assertRaisesRegex(sc.mcap.DecodePlanError, "no field path"):
            sc.mcap.DecodePlan(self.schema(), fields=["current.z"])
        incomplete = self.SCHEMA_TEXT.rsplit("=" * 80, 1)[0]
        schema = types.SimpleNamespace(
            name="vhclsim_msgs/msg/Status",
            encoding="ros2idl",
            data=incomplete.encode(),
        )
        with self.assertRaisesRegex(sc.mcap.DecodePlanError,
                                    "lacks nested struct"):
            sc.mcap.DecodePlan(schema, fields=["current.x"])

    # rosidl writes every .msg comment as a @verbatim annotation, and the
    # stock builtin_interfaces/msg/Time carries a URL in one.
    STAMPED_TEXT = '''\
================================================================================
IDL: p/msg/Stamped
module p { module msg {
  struct Stamped {
    builtin_interfaces::msg::Time stamp;
    double value;
  };
}; };
================================================================================
IDL: builtin_interfaces/msg/Time
module builtin_interfaces { module msg {
  @verbatim (language="comment", text=
    " This message communicates ROS Time defined here:" "\\n"
    " https://design.ros2.org/articles/clock_and_time.html")
  struct Time {
    int32 sec;
    uint32 nanosec;
  };
}; };
'''

    def test_annotation_url_keeps_the_dependency_section(self):
        schema = types.SimpleNamespace(
            name="p/msg/Stamped", encoding="ros2idl",
            data=self.STAMPED_TEXT.encode())
        plan = sc.mcap.DecodePlan(
            schema, fields=["stamp.nanosec", "value"])
        self.assertEqual(plan.types, ("uint32", "float64"))
        self.assertEqual(
            plan.instructions,
            (
                ("align", 4), ("skip", 4),
                ("align", 4), ("read", "uint32", 0),
                ("align", 8), ("read", "float64", 1),
            ),
        )

    def test_rejects_an_annotation_that_moves_bytes(self):
        text = """\
module p { module msg { struct S {
  @optional int32 maybe;
  double value;
}; }; };
"""
        schema = types.SimpleNamespace(
            name="p/msg/S", encoding="ros2idl", data=text.encode())
        with self.assertRaisesRegex(sc.mcap.DecodePlanError,
                                    "@optional changes the CDR layout"):
            sc.mcap.DecodePlan(schema, fields=["value"])

    def test_keeps_an_unknown_metadata_annotation(self):
        text = """\
module p { module msg { struct S {
  @defalut (value=0) @unit (value="m") double span;
  int32 tail;
}; }; };
"""
        schema = types.SimpleNamespace(
            name="p/msg/S", encoding="ros2idl", data=text.encode())
        plan = sc.mcap.DecodePlan(schema, fields=["tail"])
        self.assertEqual(
            plan.instructions,
            (("align", 8), ("skip", 8), ("align", 4), ("read", "int32", 0)))

    def test_rejects_other_mcap_schema_encodings(self):
        for encoding in ("ros2msg", "apex_json"):
            with self.subTest(encoding=encoding):
                with self.assertRaisesRegex(
                        sc.mcap.DecodePlanError, "ros2idl"):
                    sc.mcap.DecodePlan(
                        self.schema(encoding), fields=["dt"])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
