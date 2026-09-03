# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Go through the MCAP reader on a generated recording.

The Foxglove ``mcap`` package writes the fixture and serves as the oracle
for raw message iteration.  The reader never imports it.
"""

import os
import struct
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
"""

TOPIC = "/vehicle/status"
BRAKE_TOPIC = "/vehicle/brake"
LOG_TIMES = [30, 10, 20, 40]


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

    def test_plan_rejects_what_it_cannot_flatten(self):
        """A string, sequence, array, or union field has no column yet."""
        schema = mcap.Schema(1, "monitor_msgs/msg/Summary", "ros2idl",
                             SUMMARY_IDL)
        with self.assertRaisesRegex(mcap.McapError, "unsupported field"):
            mcap.DecodePlan(schema)
        schema = mcap.Schema(1, "x/msg/Y", "ros2idl",
                             b"module x { module msg { struct Y {"
                             b" uint32 a; double b[3]; }; }; };")
        with self.assertRaisesRegex(mcap.McapError, "unsupported field 'b'"):
            mcap.DecodePlan(schema)

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
