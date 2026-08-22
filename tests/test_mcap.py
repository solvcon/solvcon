# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import os
import struct
import tempfile

import unittest

import solvcon as sc

# tests/data/make_mcap_fixtures.py writes the fixtures. A chunked fixture
# takes its name from its chunk compression.
CHUNKED = ("vehicle_none", "vehicle_lz4", "vehicle_zstd")
UNCHUNKED = "vehicle_unchunked"
FIXTURES = CHUNKED + (UNCHUNKED,)

STATUS = "/vehicle/status"
IMU = "/vehicle/imu"
TOPICS = (STATUS, IMU)

# What the generator recorded: status messages this far apart in nanoseconds,
# starting here, and one imu message per twelve of them.
START_TIME = 1700000000000000000
PERIOD = 10000000
MESSAGE_COUNT = 96
END_TIME = START_TIME + (MESSAGE_COUNT - 1) * PERIOD
# The 256-byte chunk size of the generator packs the messages into this many
# chunks. Regenerating the fixtures with another chunk size changes it.
CHUNK_COUNT = 19


MAGIC = b"\x89MCAP0\r\n"


def _record(opcode, content):
    return bytes([opcode]) + struct.pack("<Q", len(content)) + content


def _prefixed(raw):
    return struct.pack("<I", len(raw)) + raw


def _text(value):
    return _prefixed(value.encode())


def _schema(schema_id, name, encoding="ros2msg", data=b"float64 x\n"):
    return _record(0x03, struct.pack("<H", schema_id) + _text(name)
                   + _text(encoding) + _prefixed(data))


def _channel(channel_id, schema_id, topic, encoding="cdr", metadata=b""):
    return _record(0x04, struct.pack("<HH", channel_id, schema_id)
                   + _text(topic) + _text(encoding) + _prefixed(metadata))


def _chunk_index(start, end, channel_ids=(1,)):
    offsets = b"".join(struct.pack("<HQ", cid, 0) for cid in channel_ids)
    return _record(0x08, struct.pack("<QQQQ", start, end, 0, 0)
                   + _prefixed(offsets) + struct.pack("<Q", 0) + _text("")
                   + struct.pack("<QQ", 0, 0))


def _statistics(start, end):
    return _record(0x0B, struct.pack("<QHIIII", 0, 0, 0, 0, 0, 0)
                   + struct.pack("<QQ", start, end) + _prefixed(b""))


def _assemble(records, trailing=b""):
    """The smallest file the reader accepts, carrying the given summary.

    The data section holds no message, because the reader under test reads
    the footer and the summary and nothing else.
    """
    head = (MAGIC + _record(0x01, _text("") + _text("solvcon test"))
            + _record(0x0F, struct.pack("<I", 0)))
    summary = b"".join(records) + trailing
    footer = _record(0x02, struct.pack("<QQI", len(head), 0, 0))
    return head + summary + footer + MAGIC


@unittest.skipUnless(sc.mcap.HAS_MCAP, "built without BUILD_MCAP")
class McapReaderTB(unittest.TestCase):
    TESTDIR = os.path.abspath(os.path.dirname(__file__))
    DATADIR = os.path.join(TESTDIR, "data")

    def path(self, name):
        return os.path.join(self.DATADIR, "%s.mcap" % name)


class McapSummaryTC(McapReaderTB):
    def test_topics(self):
        for name in FIXTURES:
            with self.subTest(name=name):
                reader = sc.mcap.Reader(self.path(name))
                self.assertEqual(reader.topics(),
                                 {STATUS: "vhcl_msgs/msg/Status",
                                  IMU: "vhcl_msgs/msg/Imu"})

    def test_time_range(self):
        for name in FIXTURES:
            with self.subTest(name=name):
                reader = sc.mcap.Reader(self.path(name))
                self.assertTrue(reader.has_time_range())
                self.assertEqual(reader.time_range(),
                                 (START_TIME, END_TIME))

    def test_schema(self):
        expect = {
            STATUS: ("vhcl_msgs/msg/Status",
                     b"float64 longitudinal_speed\nbool brake_active\n"),
            IMU: ("vhcl_msgs/msg/Imu",
                  b"float64 ax\nfloat64 ay\nfloat64 az\n"),
        }
        reader = sc.mcap.Reader(self.path("vehicle_zstd"))
        for topic, (name, data) in expect.items():
            with self.subTest(topic=topic):
                schema = reader.schema(topic)
                self.assertEqual(schema.name, name)
                self.assertEqual(schema.encoding, "ros2msg")
                self.assertEqual(schema.data, data)

    def test_schema_of_unknown_topic(self):
        reader = sc.mcap.Reader(self.path("vehicle_zstd"))
        with self.assertRaisesRegex(RuntimeError, "no such topic"):
            reader.schema("/vehicle/nothing")

    def test_path(self):
        path = self.path("vehicle_zstd")
        self.assertEqual(sc.mcap.Reader(path).path, path)

    def test_not_an_mcap_file(self):
        path = os.path.join(self.DATADIR, "rectangle.msh")
        with self.assertRaisesRegex(RuntimeError, "not an MCAP file"):
            sc.mcap.Reader(path)

    def test_missing_file(self):
        with self.assertRaisesRegex(RuntimeError, "cannot open"):
            sc.mcap.Reader(os.path.join(self.DATADIR, "no_such.mcap"))

    def test_chunk_count(self):
        for name in CHUNKED:
            with self.subTest(name=name):
                reader = sc.mcap.Reader(self.path(name))
                self.assertEqual(reader.chunk_count(), CHUNK_COUNT)

    def test_unchunked_file_has_no_chunk_index(self):
        reader = sc.mcap.Reader(self.path(UNCHUNKED))
        self.assertEqual(reader.chunk_count(), 0)

    def test_summary_offset_past_the_footer(self):
        """A corrupt offset must not size a read from an underflow."""
        # The footer holds summary_start as its first uint64, and the tail is
        # the footer record plus the closing magic.
        tail = 9 + 20 + 8
        with open(self.path("vehicle_none"), "rb") as stream:
            data = bytearray(stream.read())
        for summary_start in (len(data) - tail, len(data), 2 ** 63):
            with self.subTest(summary_start=summary_start):
                struct.pack_into("<Q", data, len(data) - tail + 9,
                                 summary_start)
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "corrupt.mcap")
                    with open(path, "wb") as stream:
                        stream.write(bytes(data))
                    with self.assertRaisesRegex(RuntimeError, "MCAP"):
                        sc.mcap.Reader(path)


class McapSyntheticTC(McapReaderTB):
    """Summaries the fixture writer never produces, assembled byte by byte."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def build(self, *records, **kw):
        path = os.path.join(self.tmp.name, "synthetic.mcap")
        with open(path, "wb") as stream:
            stream.write(_assemble(records, **kw))

        return path

    def test_time_range_from_chunk_indexes(self):
        reader = sc.mcap.Reader(self.build(
            _schema(1, "S"), _channel(1, 1, "/t"),
            _chunk_index(300, 400), _chunk_index(100, 200)))
        self.assertTrue(reader.has_time_range())
        self.assertEqual(reader.time_range(), (100, 400))

    def test_empty_chunk_keeps_the_start_time(self):
        reader = sc.mcap.Reader(self.build(
            _schema(1, "S"), _channel(1, 1, "/t"),
            _chunk_index(0, 0), _chunk_index(100, 200)))
        self.assertEqual(reader.time_range(), (100, 200))

    def test_summary_stating_no_time(self):
        reader = sc.mcap.Reader(self.build(_schema(1, "S"),
                                           _channel(1, 1, "/t")))
        self.assertFalse(reader.has_time_range())
        self.assertEqual(reader.time_range(), (0, 0))

    def test_two_channels_on_one_topic(self):
        # Written in descending id, so answering with the last record rather
        # than the highest id would name A.
        reader = sc.mcap.Reader(self.build(
            _schema(1, "A"), _schema(2, "B"),
            _channel(2, 2, "/t"), _channel(1, 1, "/t"),
            _statistics(100, 200)))
        self.assertEqual(reader.topics(), {"/t": "B"})
        self.assertEqual(reader.schema("/t").name, "B")

    def test_schema_id_zero_is_no_schema(self):
        reader = sc.mcap.Reader(self.build(_schema(0, "ignored"),
                                           _channel(1, 0, "/t")))
        self.assertEqual(reader.topics(), {"/t": ""})
        with self.assertRaisesRegex(RuntimeError, "no schema"):
            reader.schema("/t")

    def test_duplicate_records_must_agree(self):
        reader = sc.mcap.Reader(self.build(_schema(1, "A"), _schema(1, "A"),
                                           _channel(1, 1, "/t")))
        self.assertEqual(reader.topics(), {"/t": "A"})
        conflicts = (
            ((_schema(1, "A"), _schema(1, "B")), "two schemas"),
            ((_schema(1, "A"), _schema(1, "A", data=b"other\n")),
             "two schemas"),
            ((_schema(1, "A"), _channel(1, 1, "/a"), _channel(1, 1, "/b")),
             "two channels"),
            ((_schema(1, "A"), _channel(1, 1, "/t"),
              _channel(1, 1, "/t", metadata=_text("k") + _text("v"))),
             "two channels"),
        )
        for records, message in conflicts:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    sc.mcap.Reader(self.build(*records))

    def test_record_missing_a_mandatory_field(self):
        # A channel record that stops before its metadata map.
        short = _record(0x04, struct.pack("<HH", 1, 1) + _text("/t")
                        + _text("cdr"))
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            sc.mcap.Reader(self.build(_schema(1, "S"), short))

    def test_summary_ending_inside_a_record_header(self):
        with self.assertRaisesRegex(RuntimeError, "ends inside a record"):
            sc.mcap.Reader(self.build(_schema(1, "S"), trailing=b"\x03\x00"))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
