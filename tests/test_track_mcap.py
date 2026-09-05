# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Go through the MCAP reader on a generated recording.

The Foxglove ``mcap`` package writes the fixture and serves as the oracle
for raw message iteration.  The reader never imports it.
"""

import os
import struct
import time
import tempfile
import unittest

import solvcon as sc
from solvcon.track import mcap

try:
    from mcap import reader as foxglove_mcap_reader
    from mcap import writer as foxglove_mcap_writer
except ImportError:
    foxglove_mcap_writer = None

STATUS_IDL = b"""
================================================================================
IDL: vehicle_msgs/msg/Status
#include "vehicle_msgs/msg/Header.idl"

module vehicle_msgs {
  enum Mode {
    OFF,
    ON
  };
  module msg {
    @verbatim (language="comment", text=
      "Vehicle status (speed in m/s)")
    struct Status {
      vehicle_msgs::msg::Header header;
      double longitudinal_speed;  // m/s
      @default (value=FALSE)
      boolean brake_active;
      Mode mode;
    };
  };
};

================================================================================
IDL: vehicle_msgs/msg/Header

module vehicle_msgs {
  module msg {
    struct Header {
      uint32 seq; /* wraps */
    };
  };
};
"""

BRAKE_IDL = b"""
module vehicle_msgs {
  module msg {
    struct Brake {
      boolean active;
    };
  };
};
"""

SIGNALS_IDL = b"""
module demo_msgs { module msg {
  const uint32 kMax = 4;
  const string kUrl = "http://x/y"; // a string constant
  const long kNeg = -1;
  typedef double Cov[2][2];
  enum Kind { A, B, C };
  union Extra switch(Kind) {
    case A: long number;
    case B: string<16> text;
    default: double value;
  };
  union Tail switch(long) {
    case kNeg: case 'x': octet small;
    case 7: double big;
  };
  struct Tag { string<8> name; uint8 level; };
  struct Signals{
    string<256> frame_id;
    sequence<Tag, kMax> tags;
    Cov cov;
    unsigned long count;
    Kind kind;
    Extra extra;
    sequence<float> samples;
    Tag corners[2];
    boolean ok;
    Tail tail;
    double speed;
  };
}; };
"""

SUMMARY_IDL = b"""
========================================
IDL: monitor_msgs/msg/Summary
#include "monitor_msgs/msg/Header.idl"

module monitor_msgs {
  module msg {
    const uint32 kReasonCapacity = 32;
    const unsigned long MAX_LANE_ID_LENGTH = 255;

    struct Summary{
      monitor_msgs::msg::Header header;

      Version version;

      sequence<Reason, kReasonCapacity> reasons;

      sequence<string<256>, 32> notes;

      uint8 flags[3];

      sequence<double, 1> brake_duration;

      geometry_msgs::msg::Point polygon_point[2];

      sequence<geometry_msgs::msg::Point, 8> polygon;

      //! [Fields]
      @verbatim (language="comment", text=
        " target speed" "\\n"
        " in m/s")
      double v_target;
      //! [Fields]
    };
  };
};

========================================
IDL: monitor_msgs/msg/Header
module monitor_msgs {
  module msg {
    struct Header {
      @verbatim (language="comment", text=
        " Publishing module name")
      string<256> module_name;

      uint32 sequence_number;
    };
  };
};

========================================
IDL: monitor_msgs/msg/Version
module monitor_msgs {
    module msg {
        struct Version {
            uint16 major;
            uint16 minor;
        };
    };
};

========================================
IDL: monitor_msgs/msg/Reason
module monitor_msgs {
  module msg {
    struct Reason {
      uint64 timestamp;
      string<64> text;
    };
  };
};

========================================
IDL: geometry_msgs/msg/Point
module geometry_msgs {
  module msg {
    struct Point {
      double x;

      double y;

      double z;
    };
  };
};
"""

TOPIC = "/vehicle/status"
BRAKE_TOPIC = "/vehicle/brake"
LOG_TIMES = [30, 10, 20, 40]


class CdrPacker:
    """Pack an XCDR1 body, where each scalar aligns to its size."""

    def __init__(self, order):
        self.order = order
        self.body = b""

    def scalar(self, code, *values):
        size = struct.calcsize(code)
        self.body += b"\0" * (-len(self.body) % size)
        self.body += struct.pack(self.order + code * len(values), *values)
        return self

    def blob(self, data):
        self.scalar("I", len(data))
        self.body += data
        return self

    def string(self, text):
        return self.blob(text.encode() + b"\0")

    def payload(self):
        return b"\0" + (b"\x01" if self.order == "<" else b"\0") + b"\0\0" + \
            self.body


class SchemaParseTC(unittest.TestCase):

    def test_signals(self):
        """One IDL exercises every construct the parser must recognize."""
        schema = mcap.Schema(1, "demo_msgs/msg/Signals", "ros2idl",
                             SIGNALS_IDL)
        registry = mcap.parse_schema(schema)
        self.assertEqual(registry.enums["demo_msgs::msg::Kind"],
                         ("A", "B", "C"))
        self.assertEqual(registry.consts["demo_msgs::msg::kMax"], 4)
        self.assertEqual(registry.consts["demo_msgs::msg::kNeg"], -1)
        self.assertEqual(registry.typedefs["demo_msgs::msg::Cov"],
                         ("array", ("scalar", "float64"), 4))
        self.assertEqual([name for name, _ in
                          registry.structs["demo_msgs::msg::Signals"]],
                         ["frame_id", "tags", "cov", "count", "kind",
                          "extra", "samples", "corners", "ok", "tail",
                          "speed"])
        self.assertEqual(registry.unions["demo_msgs::msg::Extra"][1],
                         [((0,), ("scalar", "int32")), ((1,), ("string",)),
                          ((None,), ("scalar", "float64"))])
        self.assertEqual(registry.unions["demo_msgs::msg::Tail"][1],
                         [((-1, 120), ("scalar", "uint8")),
                          ((7,), ("scalar", "float64"))])

        plan = mcap.DecodePlan(schema)
        self.assertEqual(plan.fields, ("count", "kind", "ok", "speed"))
        self.assertEqual(plan.types, ("uint32", "uint32", "bool", "float64"))
        self.assertEqual(plan.enums, {"kind": ("A", "B", "C")})
        for fields in (["extra.number"], ["extra.value"]):
            with self.assertRaisesRegex(mcap.McapError, "unsupported"):
                mcap.DecodePlan(schema, fields)

        for order in "<>":
            for kind in range(3):
                self.assertEqual(plan.decode(pack_signals(order, kind)),
                                 (7, kind, True, 3.5), (order, kind))
        with self.assertRaisesRegex(mcap.McapError, "ends before"):
            plan.decode(pack_signals("<", 2)[:-12])

    def test_union_cases(self):
        """A union walks the case its discriminator selects, or none."""
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl", b"""
            module x { module msg {
              union Flag switch(boolean) {
                case TRUE: double when;
                case FALSE: octet why;
              };
              union Bare switch(short) { case 1: double d; };
              struct Y { Flag flag; Bare bare; uint8 tail; };
            }; };""")
        plan = mcap.DecodePlan(schema)
        self.assertEqual(plan.fields, ("tail",))
        packer = CdrPacker("<").scalar("?", True).scalar("d", 1.5)
        packer.scalar("h", 1).scalar("d", 2.5).scalar("B", 9)
        self.assertEqual(plan.decode(packer.payload()), (9,))
        packer = CdrPacker("<").scalar("?", False).scalar("B", 3)
        packer.scalar("h", 2).scalar("B", 8)
        self.assertEqual(plan.decode(packer.payload()), (8,))

        schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                             b"module x { module msg { enum Kind { A, B };"
                             b" typedef Kind Switch;"
                             b" union U switch(Switch) { case Kind::B:"
                             b" double d; }; struct Y { U u; uint8 t; };"
                             b" }; };")
        plan = mcap.DecodePlan(schema)
        packer = CdrPacker("<").scalar("I", 1).scalar("d", 1.5)
        self.assertEqual(plan.decode(packer.scalar("B", 6).payload()), (6,))

        for switch in (b"struct S { long v; }; union U switch(S)",
                       b"union U switch(double)"):
            schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                                 b"module x { module msg { " + switch +
                                 b" { case 1: long a; };"
                                 b" struct Y { U u; }; }; };")
            with self.assertRaisesRegex(mcap.McapError, "bad discriminator"):
                mcap.DecodePlan(schema)
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                             b"module x { module msg {"
                             b" struct S { U u; };"
                             b" union U switch(long) { case 1: S s; };"
                             b" struct Y { U u; }; }; };")
        with self.assertRaisesRegex(mcap.McapError, "union .* contains"):
            mcap.DecodePlan(schema)

    def test_bundle(self):
        """A recording bundles several IDL blocks under one schema."""
        schema = mcap.Schema(1, "monitor_msgs/msg/Summary", "ros2idl",
                             SUMMARY_IDL)
        registry = mcap.parse_schema(schema)
        self.assertEqual(registry.consts["monitor_msgs::msg::"
                                         "MAX_LANE_ID_LENGTH"], 255)
        self.assertEqual(registry.structs["monitor_msgs::msg::Version"],
                         [("major", ("scalar", "uint16")),
                          ("minor", ("scalar", "uint16"))])
        self.assertEqual(registry.structs["monitor_msgs::msg::Summary"][2],
                         ("reasons",
                          ("sequence", ("named", "Reason",
                                        ("monitor_msgs", "msg")))))

    def test_walk_steps_over_containers(self):
        """The auto plan steps over a string, a sequence, or an array."""
        schema = mcap.Schema(1, "monitor_msgs/msg/Summary", "ros2idl",
                             SUMMARY_IDL)
        plan = mcap.DecodePlan(schema)
        self.assertEqual(plan.fields, ("header.sequence_number",
                                       "version.major", "version.minor",
                                       "v_target"))
        for fields in (["reasons"], ["polygon_point"], ["polygon"]):
            with self.assertRaisesRegex(mcap.McapError, "unsupported"):
                mcap.DecodePlan(schema, fields)
        for order in "<>":
            self.assertEqual(plan.decode(pack_summary(order)),
                             (3, 1, 2, 22.5), order)
        payload = pack_summary("<")
        with self.assertRaisesRegex(mcap.McapError, "ends before"):
            plan.decode(payload[:-3])
        with self.assertRaisesRegex(mcap.McapError, "ends before"):
            plan.decode(CdrPacker("<").scalar("I", 999).payload())
        with self.assertRaisesRegex(mcap.McapError, "encapsulation"):
            plan.decode(b"\0\x02\0\0" + payload[4:])

    def test_named_containers_decode(self):
        """A named string, sequence, or array of scalars is a column."""
        schema = mcap.Schema(1, "monitor_msgs/msg/Summary", "ros2idl",
                             SUMMARY_IDL)
        plan = mcap.DecodePlan(schema, ["header.module_name", "notes",
                                        "flags", "brake_duration",
                                        "v_target"])
        self.assertEqual(plan.types, ("str", "str[]", "uint8[]",
                                      "float64[]", "float64"))
        for order in "<>":
            self.assertEqual(plan.decode(pack_summary(order)),
                             ("mod", ["n"], [7, 8, 9], [0.5], 22.5), order)

        schema = mcap.Schema(1, "demo_msgs/msg/Signals", "ros2idl",
                             SIGNALS_IDL)
        plan = mcap.DecodePlan(schema, ["frame_id", "cov", "samples"])
        self.assertEqual(plan.types, ("str", "float64[]", "float32[]"))
        self.assertEqual(plan.decode(pack_signals(">", 1)),
                         ("abc", [1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5]))
        with self.assertRaisesRegex(mcap.McapError, "ends before"):
            mcap.DecodePlan(schema, ["samples"]).decode(
                pack_signals("<", 1)[:-40])

    def test_string_edges(self):
        """An empty, NUL-bearing, or non-UTF-8 string, and an empty list."""
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                             b"module x { module msg { enum E { P, Q };"
                             b" struct Y { sequence<E> es; string s;"
                             b" sequence<string> ss; }; }; };")
        plan = mcap.DecodePlan(schema, ["es", "s", "ss"])
        self.assertEqual(plan.fields, ("es", "s", "ss"))
        self.assertEqual(plan.types, ("uint32[]", "str", "str[]"))
        self.assertEqual(plan.enums, {"es": ("P", "Q")})
        packer = CdrPacker("<").scalar("I", 0).string("a\0b").scalar("I", 0)
        self.assertEqual(plan.decode(packer.payload()), ([], "a\0b", []))
        packer = CdrPacker("<").scalar("I", 2, 1, 0).blob(b"\0")
        self.assertEqual(plan.decode(packer.scalar("I", 0).payload()),
                         ([1, 0], "", []))
        packer = CdrPacker("<").scalar("I", 0).blob(b"\xff\0")
        with self.assertRaisesRegex(mcap.McapError, "not UTF-8"):
            plan.decode(packer.scalar("I", 0).payload())
        self.assertEqual(mcap.DecodePlan(schema).decode(packer.payload()), ())
        for data in (b"", b"abc"):
            packer = CdrPacker("<").scalar("I", 0).blob(data).scalar("I", 0)
            with self.assertRaisesRegex(mcap.McapError, "NUL terminator"):
                plan.decode(packer.payload())

    def test_huge_array_bound_fails_fast(self):
        """A short payload fails before the walk iterates a huge array."""
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                             b"module x { module msg { struct P {"
                             b" sequence<double> a; }; struct Y {"
                             b" P p[4000000000]; double v; }; }; };")
        plan = mcap.DecodePlan(schema)
        self.assertEqual(plan.fields, ("v",))
        started = time.monotonic()
        with self.assertRaisesRegex(mcap.McapError, "ends before"):
            plan.decode(CdrPacker("<").scalar("I", 0).payload())
        self.assertLess(time.monotonic() - started, 5.0)

    def test_nested_structs_flatten(self):
        """A struct of structs, enums, and scalars still gives columns."""
        schema = mcap.Schema(1, "vehicle_msgs/msg/Status", "ros2idl",
                             STATUS_IDL)
        plan = mcap.DecodePlan(schema)
        self.assertEqual(plan.fields, ("header.seq", "longitudinal_speed",
                                       "brake_active", "mode"))
        self.assertEqual(plan.types, ("uint32", "float64", "bool", "uint32"))
        self.assertEqual(plan.enums, {"mode": ("OFF", "ON")})
        with self.assertRaisesRegex(mcap.McapError, "duplicate fields"):
            mcap.DecodePlan(schema, ["mode", "mode"])

    def test_unsupported(self):
        """An IDL construct the plan cannot decode raises ``McapError``."""
        for field in (b"wchar c;", b"wstring s;"):
            schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                                 b"module x { module msg { struct Y { " +
                                 field + b" }; }; };")
            with self.assertRaisesRegex(mcap.McapError, "unsupported"):
                mcap.DecodePlan(schema)
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                             b"module x { module msg { typedef B A; "
                             b"typedef A B; struct Y { A a; }; }; };")
        with self.assertRaisesRegex(mcap.McapError, "cyclic"):
            mcap.DecodePlan(schema)
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                             b"module x { module msg { enum E { A, "
                             b"@value(5) B }; struct Y { E e; }; }; };")
        with self.assertRaisesRegex(mcap.McapError, "declaration order"):
            mcap.DecodePlan(schema)
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                             b"module x { module msg { enum E { "
                             b"@value(0) A, @value(1) B }; "
                             b"struct Y { E e; }; }; };")
        self.assertEqual(mcap.DecodePlan(schema).enums, {"e": ("A", "B")})
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                             b"module x { module msg { struct Y { "
                             b"Z z; }; }; };")
        with self.assertRaisesRegex(mcap.McapError, "unknown type 'Z'"):
            mcap.DecodePlan(schema)
        schema = mcap.Schema(1, "x/msg/Y", "protobuf", b"")
        with self.assertRaisesRegex(mcap.McapError, "protobuf"):
            mcap.DecodePlan(schema)
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl", b"\xff\xfe")
        with self.assertRaisesRegex(mcap.McapError, "not UTF-8"):
            mcap.DecodePlan(schema)

    def test_malformed(self):
        """Every parse failure raises ``McapError``, never a raw builtin."""
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                             b"module x { module msg { enum E { A =")
        with self.assertRaisesRegex(mcap.McapError, "ends inside a block"):
            mcap.DecodePlan(schema)
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                             b"module x { module msg { const long V = 1; "
                             b"enum E { A, B = V }; struct Y { E e; }; "
                             b"}; };")
        self.assertEqual(mcap.DecodePlan(schema).enums, {"e": ("A", "B")})

    def test_const_spans_lines(self):
        """A ``const`` broken before its ``=`` keeps its value."""
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                             b"module x { module msg {\nconst long N\n"
                             b"= 3;\nstruct Y { double v[N]; uint8 t; };\n"
                             b"}; };")
        registry = mcap.parse_schema(schema)
        self.assertEqual(registry.consts["x::msg::N"], 3)
        self.assertEqual(registry.structs["x::msg::Y"][0][1],
                         ("array", ("scalar", "float64"), 3))

    def test_string_const_does_not_shadow(self):
        """A string ``const`` does not hide an integer one further out."""
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                             b"module x { const long N = 3; module msg {"
                             b" const string N = \"s\";"
                             b" struct Y { double v[N]; uint8 t; }; }; };")
        registry = mcap.parse_schema(schema)
        self.assertNotIn("x::msg::N", registry.consts)
        self.assertEqual(registry.structs["x::msg::Y"][0][1],
                         ("array", ("scalar", "float64"), 3))


def pack_signals(order, kind):
    """Pack a ``Signals`` message with ``count`` 7 and ``speed`` 3.5."""
    packer = CdrPacker(order).string("abc")
    packer.scalar("I", 2).string("x").scalar("B", 9)
    packer.string("yz").scalar("B", 8)
    packer.scalar("d", 1.0, 2.0, 3.0, 4.0).scalar("I", 7).scalar("I", kind)
    packer.scalar("I", kind)
    if kind == 0:
        packer.scalar("i", -5)
    elif kind == 1:
        packer.string("h")
    else:
        packer.scalar("d", 6.5)
    packer.scalar("I", 3).scalar("f", 1.5, 2.5, 3.5)
    packer.string("p").scalar("B", 1).string("q").scalar("B", 2)
    packer.scalar("?", True)
    if kind == 0:
        packer.scalar("i", -1).scalar("B", 4)
    elif kind == 1:
        packer.scalar("i", ord("x")).scalar("B", 4)
    else:
        packer.scalar("i", 7).scalar("d", 8.5)
    return packer.scalar("d", 3.5).payload()


def pack_summary(order):
    """Pack a ``Summary`` message with ``sequence_number`` 3."""
    packer = CdrPacker(order).string("mod").scalar("I", 3)
    packer.scalar("H", 1, 2)
    packer.scalar("I", 2).scalar("Q", 10).string("a")
    packer.scalar("Q", 11).string("bb")
    packer.scalar("I", 1).string("n")
    packer.scalar("B", 7, 8, 9)
    packer.scalar("I", 1).scalar("d", 0.5)
    packer.scalar("d", *range(6))
    packer.scalar("I", 1).scalar("d", 1.0, 2.0, 3.0)
    return packer.scalar("d", 22.5).payload()


def pack_status(seq, speed, brake_active, mode):
    """Pack a ``Status`` message as little-endian XCDR1."""
    body = struct.pack("<I", seq) + b"\0" * 4
    body += struct.pack("<d?", speed, brake_active) + b"\0" * 3
    body += struct.pack("<I", mode)
    return b"\0\x01\0\0" + body


def write_fixture(path, compression):
    compression_type = getattr(foxglove_mcap_writer.CompressionType,
                               compression.upper())
    with open(path, "wb") as fp:
        writer = foxglove_mcap_writer.Writer(fp, compression=compression_type,
                                             chunk_size=64)
        writer.start(profile="ros2")
        schema_id = writer.register_schema("vehicle_msgs/msg/Status",
                                           "ros2idl", STATUS_IDL)
        channel_id = writer.register_channel(TOPIC, "cdr", schema_id)
        for seq, log_time in enumerate(LOG_TIMES):
            payload = pack_status(seq, float(log_time), log_time >= 20,
                                  log_time % 20 == 0)
            writer.add_message(channel_id, log_time, payload, log_time)
        writer.finish()


@unittest.skipIf(foxglove_mcap_writer is None,
                 "the Foxglove mcap package is not installed")
class McapInOutTC(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "vehicle.mcap")
        write_fixture(self.path, "zstd")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_go_through(self):
        with mcap.Reader(self.path) as reader:
            # Summary.
            self.assertEqual(reader.profile, "ros2")
            self.assertEqual(reader.topics(),
                             {TOPIC: "vehicle_msgs/msg/Status"})
            self.assertEqual(reader.time_range(), (10, 40))
            self.assertEqual(reader.message_count(), 4)
            self.assertEqual(reader.message_counts(), {TOPIC: 4})
            self.assertEqual(reader.path, self.path)
            self.assertEqual(reader.size, os.path.getsize(self.path))

            # Raw messages come in file order; the range is half open.
            self.assertEqual([t for t, _ in reader.messages(TOPIC)],
                             LOG_TIMES)
            self.assertEqual(
                [t for t, _ in reader.messages(TOPIC, start_ns=20,
                                               end_ns=40)],
                [30, 20])

            # Schema and plan.
            schema = reader.schema(TOPIC)
            self.assertEqual(mcap.DecodePlan(schema).fields,
                             ("header.seq", "longitudinal_speed",
                              "brake_active", "mode"))
            plan = mcap.DecodePlan(schema, ["mode", "longitudinal_speed"])
            self.assertEqual(plan.types, ("uint32", "float64"))
            self.assertEqual(plan.enums, {"mode": ("OFF", "ON")})
            with self.assertRaisesRegex(mcap.McapError, "header"):
                mcap.DecodePlan(schema, ["header"])

            # Columns sorted by log time.
            ext = reader.extract(TOPIC, plan)
            self.assertIsInstance(ext.time, sc.SimpleArrayUint64)
            self.assertIsInstance(ext.columns["mode"], sc.SimpleArrayUint32)
            self.assertEqual(ext.time.ndarray.tolist(), [10, 20, 30, 40])
            self.assertEqual(ext.columns["mode"].ndarray.tolist(),
                             [0, 1, 0, 1])

            # Frames, one pass for every topic.
            frame = reader.extract_frame_many({TOPIC: None})[TOPIC]
            self.assertEqual(frame.columns, list(mcap.DecodePlan(schema)
                                                 .fields))
            self.assertEqual(frame.index.tolist(), [10, 20, 30, 40])
            self.assertEqual(frame["longitudinal_speed"].tolist(),
                             [10.0, 20.0, 30.0, 40.0])
            self.assertEqual(frame["brake_active"].tolist(),
                             [False, True, True, True])
            self.assertEqual(frame["header.seq"].tolist(), [1, 2, 0, 3])

    def test_extract_frame_many_mixed_specs(self):
        path = os.path.join(self.tmpdir.name, "two_topics.mcap")
        with open(path, "wb") as fp:
            writer = foxglove_mcap_writer.Writer(fp)
            writer.start(profile="ros2")
            schema_id = writer.register_schema("vehicle_msgs/msg/Status",
                                               "ros2idl", STATUS_IDL)
            channel_id = writer.register_channel(TOPIC, "cdr", schema_id)
            for seq, log_time in enumerate(LOG_TIMES):
                payload = pack_status(seq, float(log_time), log_time >= 20,
                                      log_time % 20 == 0)
                writer.add_message(channel_id, log_time, payload, log_time)
            schema_id = writer.register_schema("vehicle_msgs/msg/Brake",
                                               "ros2idl", BRAKE_IDL)
            channel_id = writer.register_channel(BRAKE_TOPIC, "cdr",
                                                 schema_id)
            for log_time in (15, 35):
                payload = b"\0\x01\0\0" + struct.pack("<?", log_time >= 20)
                writer.add_message(channel_id, log_time, payload, log_time)
            writer.finish()

        with mcap.Reader(path) as reader:
            self.assertEqual(reader.message_counts(),
                             {TOPIC: 4, BRAKE_TOPIC: 2})
            plan = mcap.DecodePlan(reader.schema(TOPIC),
                                   ["longitudinal_speed", "mode"])
            frames = reader.extract_frame_many({TOPIC: plan,
                                                BRAKE_TOPIC: ["active"]})

        status = frames[TOPIC]
        self.assertEqual(status.columns, ["longitudinal_speed", "mode"])
        self.assertEqual(status.index.tolist(), [10, 20, 30, 40])
        self.assertEqual(status["longitudinal_speed"].tolist(),
                         [10.0, 20.0, 30.0, 40.0])
        brake = frames[BRAKE_TOPIC]
        self.assertEqual(brake.columns, ["active"])
        self.assertEqual(brake.index.tolist(), [15, 35])
        self.assertEqual(brake["active"].tolist(), [False, True])

    def test_extract_container_columns(self):
        """A string or container column is an object array, not a frame."""
        path = os.path.join(self.tmpdir.name, "signals.mcap")
        with open(path, "wb") as fp:
            writer = foxglove_mcap_writer.Writer(fp)
            writer.start(profile="ros2")
            schema_id = writer.register_schema("demo_msgs/msg/Signals",
                                               "ros2idl", SIGNALS_IDL)
            channel_id = writer.register_channel("/signals", "cdr",
                                                 schema_id)
            for log_time, kind in ((20, 0), (10, 1)):
                writer.add_message(channel_id, log_time,
                                   pack_signals("<", kind), log_time)
            writer.finish()

        with mcap.Reader(path) as reader:
            ext = reader.extract("/signals", ["frame_id", "samples",
                                              "count"])
        self.assertEqual(ext.time.ndarray.tolist(), [10, 20])
        self.assertIsInstance(ext.columns["count"], sc.SimpleArrayUint32)
        self.assertEqual(ext.columns["frame_id"].dtype, object)
        self.assertEqual(ext.columns["frame_id"].tolist(), ["abc", "abc"])
        self.assertEqual(ext.columns["samples"].tolist(),
                         [[1.5, 2.5, 3.5], [1.5, 2.5, 3.5]])
        with self.assertRaisesRegex(mcap.McapError, "frame_id has unsup"):
            ext.to_frame()

    def test_messages_match_foxglove(self):
        for compression in ("none", "lz4", "zstd"):
            path = os.path.join(self.tmpdir.name, compression + ".mcap")
            write_fixture(path, compression)
            with mcap.Reader(path) as reader:
                ours = sorted(reader.messages(TOPIC))
            with open(path, "rb") as fp:
                foxglove = foxglove_mcap_reader.make_reader(fp)
                theirs = sorted((msg.log_time, msg.data) for _, _, msg
                                in foxglove.iter_messages(topics=[TOPIC]))
            self.assertEqual(ours, theirs, compression)

    def test_a_truncated_file_raises_mcap_error(self):
        path = os.path.join(self.tmpdir.name, "truncated.mcap")
        with open(path, "wb") as fp:
            fp.write(b"\x89MCAP0\r\n" * 2)
        with self.assertRaisesRegex(mcap.McapError, "malformed record"):
            mcap.Reader(path)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
