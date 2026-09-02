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

TOPIC = "/vehicle/status"
BRAKE_TOPIC = "/vehicle/brake"
LOG_TIMES = [30, 10, 20, 40]


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
class McapPrototypeTC(unittest.TestCase):

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

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
