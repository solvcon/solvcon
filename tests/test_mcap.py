# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import itertools
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
# One imu message per this many status messages.
IMU_PERIOD = 12
# The imu messages reach only this many of the chunks, so a query for the
# topic leaves the rest unread.
IMU_CHUNK_COUNT = 8

# Little-endian CDR encapsulation header, as a ROS 2 recording carries it.
CDR_HEADER = b"\x00\x01\x00\x00"

# A cursor shared between two iterators still answers the first two reads
# correctly, so the interleaving tests alternate this many times.
INTERLEAVED_READS = 4


class McapBytes:
    """Builders for the records of an MCAP file, as raw bytes.

    The fixture writer cannot produce a malformed or hand-tuned summary, so
    the tests that need one assemble the file byte by byte.
    """

    MAGIC = b"\x89MCAP0\r\n"

    @staticmethod
    def record(opcode, content):
        return bytes([opcode]) + struct.pack("<Q", len(content)) + content

    @staticmethod
    def prefixed(raw):
        return struct.pack("<I", len(raw)) + raw

    @classmethod
    def text(cls, value):
        return cls.prefixed(value.encode())

    @classmethod
    def schema(cls, schema_id, name, encoding="ros2msg",
               data=b"float64 x\n"):
        return cls.record(0x03, struct.pack("<H", schema_id)
                          + cls.text(name) + cls.text(encoding)
                          + cls.prefixed(data))

    @classmethod
    def channel(cls, channel_id, schema_id, topic, encoding="cdr",
                metadata=b""):
        return cls.record(0x04, struct.pack("<HH", channel_id, schema_id)
                          + cls.text(topic) + cls.text(encoding)
                          + cls.prefixed(metadata))

    @classmethod
    def chunk_index(cls, start, end, channel_ids=(1,), offset=0, length=0):
        offsets = b"".join(struct.pack("<HQ", cid, 0) for cid in channel_ids)
        return cls.record(0x08, struct.pack("<QQQQ", start, end, offset,
                                            length)
                          + cls.prefixed(offsets) + struct.pack("<Q", 0)
                          + cls.text("") + struct.pack("<QQ", 0, 0))

    @classmethod
    def message(cls, channel_id, log_time, payload):
        return cls.record(0x05, struct.pack("<HIQQ", channel_id, 0, log_time,
                                            log_time) + payload)

    @classmethod
    def chunk(cls, *records):
        """Return an uncompressed chunk record holding the given records."""
        body = b"".join(records)
        return cls.record(0x06, struct.pack("<QQQI", 0, 0, len(body), 0)
                          + cls.text("") + struct.pack("<Q", len(body))
                          + body)

    @classmethod
    def statistics(cls, start, end):
        return cls.record(0x0B, struct.pack("<QHIIII", 0, 0, 0, 0, 0, 0)
                          + struct.pack("<QQ", start, end)
                          + cls.prefixed(b""))

    @classmethod
    def preamble(cls):
        """Return the magic and the header record.

        They precede the data section, so their length is the file offset
        of the first data record.
        """
        return cls.MAGIC + cls.record(0x01, cls.text("")
                                      + cls.text("solvcon test"))

    @classmethod
    def assemble(cls, records, trailing=b"", data=b""):
        """Return the smallest file the reader accepts, with that summary.

        The data section holds only the records the data argument carries.
        Most of these tests read the footer and the summary and nothing
        else.
        """
        head = cls.preamble() + data + cls.record(0x0F, struct.pack("<I", 0))
        summary = b"".join(records) + trailing
        footer = cls.record(0x02, struct.pack("<QQI", len(head), 0, 0))
        return head + summary + footer + cls.MAGIC


@unittest.skipUnless(sc.mcap.HAS_MCAP, "built without BUILD_MCAP")
class McapReaderTB(unittest.TestCase):
    TESTDIR = os.path.abspath(os.path.dirname(__file__))
    DATADIR = os.path.join(TESTDIR, "data")

    def path(self, name):
        return os.path.join(self.DATADIR, "%s.mcap" % name)

    def expected_messages(self, topic):
        """Return the log time and payload of every message on a topic.

        The generator tests/data/make_mcap_fixtures.py wrote them, and this
        method repeats the arithmetic it used.
        """
        out = []
        for index in range(MESSAGE_COUNT):
            log_time = START_TIME + index * PERIOD
            if STATUS == topic:
                out.append((log_time, CDR_HEADER
                            + struct.pack("<dB", 1.5 * index, index % 2)))
            elif 0 == index % IMU_PERIOD:
                out.append((log_time, CDR_HEADER
                            + struct.pack("<ddd", index, -index, 9.81)))

        return out


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


class McapSyntheticTC(McapReaderTB, McapBytes):
    """Summaries the fixture writer never produces, assembled byte by byte."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def build(self, *records, **kw):
        path = os.path.join(self.tmp.name, "synthetic.mcap")
        with open(path, "wb") as stream:
            stream.write(self.assemble(records, **kw))

        return path

    def test_time_range_from_chunk_indexes(self):
        reader = sc.mcap.Reader(self.build(
            self.schema(1, "S"), self.channel(1, 1, "/t"),
            self.chunk_index(300, 400), self.chunk_index(100, 200)))
        self.assertTrue(reader.has_time_range())
        self.assertEqual(reader.time_range(), (100, 400))

    def test_empty_chunk_keeps_the_start_time(self):
        reader = sc.mcap.Reader(self.build(
            self.schema(1, "S"), self.channel(1, 1, "/t"),
            self.chunk_index(0, 0), self.chunk_index(100, 200)))
        self.assertEqual(reader.time_range(), (100, 200))

    def test_summary_stating_no_time(self):
        reader = sc.mcap.Reader(self.build(self.schema(1, "S"),
                                           self.channel(1, 1, "/t")))
        self.assertFalse(reader.has_time_range())
        self.assertEqual(reader.time_range(), (0, 0))

    def test_two_channels_on_one_topic(self):
        # Written in descending id, so answering with the last record rather
        # than the highest id would name A.
        reader = sc.mcap.Reader(self.build(
            self.schema(1, "A"), self.schema(2, "B"),
            self.channel(2, 2, "/t"), self.channel(1, 1, "/t"),
            self.statistics(100, 200)))
        self.assertEqual(reader.topics(), {"/t": "B"})
        self.assertEqual(reader.schema("/t").name, "B")

    def test_schema_id_zero_is_no_schema(self):
        reader = sc.mcap.Reader(self.build(self.schema(0, "ignored"),
                                           self.channel(1, 0, "/t")))
        self.assertEqual(reader.topics(), {"/t": ""})
        with self.assertRaisesRegex(RuntimeError, "no schema"):
            reader.schema("/t")

    def test_duplicate_records_must_agree(self):
        reader = sc.mcap.Reader(self.build(
            self.schema(1, "A"), self.schema(1, "A"),
            self.channel(1, 1, "/t")))
        self.assertEqual(reader.topics(), {"/t": "A"})
        conflicts = (
            ((self.schema(1, "A"), self.schema(1, "B")), "two schemas"),
            ((self.schema(1, "A"), self.schema(1, "A", data=b"other\n")),
             "two schemas"),
            ((self.schema(1, "A"), self.channel(1, 1, "/a"),
              self.channel(1, 1, "/b")),
             "two channels"),
            ((self.schema(1, "A"), self.channel(1, 1, "/t"),
              self.channel(1, 1, "/t",
                           metadata=self.text("k") + self.text("v"))),
             "two channels"),
        )
        for records, message in conflicts:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    sc.mcap.Reader(self.build(*records))

    def test_record_missing_a_mandatory_field(self):
        # A channel record that stops before its metadata map.
        short = self.record(0x04, struct.pack("<HH", 1, 1)
                            + self.text("/t") + self.text("cdr"))
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            sc.mcap.Reader(self.build(self.schema(1, "S"), short))

    def test_an_index_without_channel_ids_cannot_prune(self):
        """An empty channel list says nothing, so the chunk must be read.

        The specification gives the empty map of message index offsets the
        meaning "no message indexing is available", and the reader derives
        the channel ids from that map.  Reading the emptiness as "holds no
        wanted channel" would drop every message of a conforming file.
        """
        chunk = self.chunk(self.message(1, 10, b"payload"))
        path = self.build(
            self.schema(1, "S"), self.channel(1, 1, "/t"),
            self.chunk_index(10, 10, channel_ids=(),
                             offset=len(self.preamble()),
                             length=len(chunk)),
            data=chunk)
        reader = sc.mcap.Reader(path)
        self.assertEqual(reader.messages("/t").selected_chunk_count(), 1)
        self.assertEqual(list(reader.messages("/t")), [(10, b"payload")])

    def test_a_message_beside_a_chunk(self):
        """A chunk index names the chunks and must not hide the rest.

        The walk that picks up the lone message must also not yield the
        chunk a second time.
        """
        chunk = self.chunk(self.message(1, 10, b"in"))
        path = self.build(
            self.schema(1, "S"), self.channel(1, 1, "/t"),
            self.chunk_index(10, 10, offset=len(self.preamble()),
                             length=len(chunk)),
            data=chunk + self.message(1, 20, b"out"))
        reader = sc.mcap.Reader(path)
        self.assertEqual(reader.messages("/t").selected_chunk_count(), 1)
        self.assertEqual(list(reader.messages("/t")),
                         [(10, b"in"), (20, b"out")])

    def test_summary_ending_inside_a_record_header(self):
        with self.assertRaisesRegex(RuntimeError, "ends inside a record"):
            sc.mcap.Reader(self.build(self.schema(1, "S"),
                                      trailing=b"\x03\x00"))


class McapMessageTC(McapReaderTB):
    def test_messages(self):
        for name, topic in itertools.product(FIXTURES, TOPICS):
            with self.subTest(name=name, topic=topic):
                reader = sc.mcap.Reader(self.path(name))
                self.assertEqual(list(reader.messages(topic)),
                                 self.expected_messages(topic))

    def test_interleaved_iterators_do_not_share_a_cursor(self):
        """Two iterators on one topic each walk it from the start."""
        expected = self.expected_messages(STATUS)
        reader = sc.mcap.Reader(self.path("vehicle_zstd"))
        first = reader.messages(STATUS)
        second = reader.messages(STATUS)
        for index in range(INTERLEAVED_READS):
            first_message = next(first)
            second_message = next(second)
            self.assertEqual(first_message, expected[index])
            self.assertEqual(second_message, expected[index])

    def test_reading_one_topic_leaves_another_alone(self):
        """A read on one topic must not advance another topic's iterator."""
        status_expected = self.expected_messages(STATUS)
        imu_expected = self.expected_messages(IMU)
        reader = sc.mcap.Reader(self.path("vehicle_zstd"))
        status = reader.messages(STATUS)
        imu = reader.messages(IMU)
        for index in range(INTERLEAVED_READS):
            status_message = next(status)
            imu_message = next(imu)
            self.assertEqual(status_message, status_expected[index])
            self.assertEqual(imu_message, imu_expected[index])

    def test_sparse_topic_prunes_chunks(self):
        for name in CHUNKED:
            with self.subTest(name=name):
                reader = sc.mcap.Reader(self.path(name))
                self.assertEqual(
                    reader.messages(STATUS).selected_chunk_count(),
                    CHUNK_COUNT)
                self.assertEqual(reader.messages(IMU).selected_chunk_count(),
                                 IMU_CHUNK_COUNT)

    def test_unchunked_file_has_no_chunk_to_prune(self):
        reader = sc.mcap.Reader(self.path(UNCHUNKED))
        self.assertEqual(reader.messages(IMU).selected_chunk_count(), 0)

    def test_messages_of_unknown_topic(self):
        reader = sc.mcap.Reader(self.path("vehicle_zstd"))
        with self.assertRaisesRegex(RuntimeError, "no such topic"):
            reader.messages("/vehicle/nothing")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
