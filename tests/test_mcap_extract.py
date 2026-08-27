# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import os
import struct
import tempfile

import solvcon as sc

# The reader tests own the test base, the byte builders, and the fixture
# constants this module reuses.
import test_mcap

SAMPLE = "/signals/sample"
WHEEL = "/signals/wheel"

# What tests/data/make_mcap_fixtures.py recorded into signals.mcap: this many
# sample messages, and one wheel message per WHEEL_PERIOD of them.
SIGNAL_COUNT = 40
WHEEL_PERIOD = 5

# Requested field path to the array type of its column and to the value the
# generator wrote for the message of each index.
SAMPLE_COLUMNS = {
    "flags": (sc.SimpleArrayUint8, lambda index: index % 256),
    "speed": (sc.SimpleArrayFloat64, lambda index: 1.5 * index),
    "header.seq": (sc.SimpleArrayUint32, lambda index: index),
    "header.stamp": (sc.SimpleArrayFloat64, lambda index: 0.001 * index),
    "valid": (sc.SimpleArrayBool, lambda index: 0 == index % 2),
    "gear": (sc.SimpleArrayInt16, lambda index: index % 8 - 4),
    "odometer": (sc.SimpleArrayUint64, lambda index: 1000 * index + 7),
}
SAMPLE_FIELDS = tuple(SAMPLE_COLUMNS)

# IDL of the one-field struct the two-channel files carry.
IDL = ("=" * 80 + "\nIDL: p/msg/S\n"
       "module p { module msg { struct S { double x; }; }; };\n")


def sample_values(field):
    """Return the value of a field of every sample message."""
    value_of = SAMPLE_COLUMNS[field][1]
    return [value_of(index) for index in range(SIGNAL_COUNT)]


class McapExtractTB(test_mcap.McapReaderTB, test_mcap.McapBytes):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def reader(self):
        return sc.mcap.Reader(self.path("signals"))

    def extract(self, reader, topic, column_count, instructions):
        """Return the time column and the list of field columns."""
        plan = sc.core.McapDecodePlan(instructions, column_count)
        return reader.extract(topic, plan)

    def extract_fields(self, topic, fields):
        reader = self.reader()
        plan = sc.mcap.DecodePlan(reader.schema(topic), fields=fields)
        return self.extract(reader, topic, len(plan.fields),
                            plan.instructions)

    def extract_two_channels(self, schema_id, encoding):
        """Extract from a topic two channels carry.

        The first channel carries schema 1 as CDR; the second carries the
        given schema id and message encoding.
        """
        payload = test_mcap.CDR_HEADER + struct.pack("<d", 1.0)
        chunk = self.chunk(self.message(1, 10, payload),
                           self.message(2, 20, payload))
        raw = self.assemble(
            (self.schema(1, "p/msg/S", "ros2idl", IDL.encode()),
             self.schema(2, "p/msg/T", "ros2idl", IDL.encode()),
             self.channel(1, 1, "/t"),
             self.channel(2, schema_id, "/t", encoding=encoding),
             self.chunk_index(10, 20, channel_ids=(1, 2),
                              offset=len(self.preamble()),
                              length=len(chunk))),
            data=chunk)
        path = os.path.join(self.tmp.name, "two_channels.mcap")
        with open(path, "wb") as stream:
            stream.write(raw)
        plan = sc.mcap.DecodePlan(IDL, fields=["x"])
        return self.extract(sc.mcap.Reader(path), "/t", 1, plan.instructions)


class McapExtractTC(McapExtractTB):
    def test_columns_hold_every_message(self):
        _, columns = self.extract_fields(SAMPLE, SAMPLE_FIELDS)
        self.assertEqual(len(columns), len(SAMPLE_FIELDS))
        for index, field in enumerate(SAMPLE_FIELDS):
            with self.subTest(field=field):
                self.assertEqual(list(columns[index]), sample_values(field))

    def test_time_column(self):
        time, _ = self.extract_fields(SAMPLE, ["speed"])
        expected = [test_mcap.START_TIME + index * test_mcap.PERIOD
                    for index in range(SIGNAL_COUNT)]
        self.assertEqual(list(time), expected)

    def test_column_types(self):
        """Each column is the array type the plan states for the field."""
        time, columns = self.extract_fields(SAMPLE, SAMPLE_FIELDS)
        for index, field in enumerate(SAMPLE_FIELDS):
            with self.subTest(field=field):
                self.assertIsInstance(columns[index],
                                      SAMPLE_COLUMNS[field][0])
        self.assertIsInstance(time, sc.SimpleArrayUint64)

    def test_requested_order_is_column_order(self):
        _, columns = self.extract_fields(SAMPLE, ["gear", "speed"])
        self.assertEqual(list(columns[0]), sample_values("gear"))
        self.assertEqual(list(columns[1]), sample_values("speed"))

    def test_walk_over_sequence_and_array_of_structs(self):
        """The field after the containers must land on the right offset.

        The wheel message holds a sequence of structs whose length changes
        from message to message, then a fixed array of the same struct.
        """
        _, columns = self.extract_fields(WHEEL, ["slip"])
        expected = [0.125 * index
                    for index in range(0, SIGNAL_COUNT, WHEEL_PERIOD)]
        self.assertEqual(list(columns[0]), expected)

    def test_extract_of_unknown_topic(self):
        reader = self.reader()
        plan = sc.mcap.DecodePlan(reader.schema(SAMPLE), fields=["speed"])
        with self.assertRaisesRegex(RuntimeError, "no such topic"):
            self.extract(reader, "/signals/nothing", 1, plan.instructions)

    def test_topic_carried_by_two_schemas(self):
        """One plan cannot walk two layouts, so the extraction must stop.

        ``schema()`` answers with the channel of highest id, and
        ``messages()`` walks every channel of the topic.  The messages of
        the other channel would then decode against a schema of another
        id, which need not describe them.
        """
        with self.assertRaisesRegex(RuntimeError, "another schema"):
            self.extract_two_channels(schema_id=2, encoding="cdr")

    def test_topic_carried_by_a_channel_of_another_encoding(self):
        """A plan walks CDR, so a channel of another encoding must stop it.

        Such a payload would either fail on the encapsulation header or,
        when its first bytes look like one, decode into wrong columns.
        """
        with self.assertRaisesRegex(RuntimeError, "no CDR encoding"):
            self.extract_two_channels(schema_id=1, encoding="json")


class McapExtractGuardTC(McapExtractTB):
    """Walks the executor must stop, or must not let run away.

    The plan check accepts these plans; what stops them is the payload.
    """

    def test_container_of_elements_the_plan_walks_over_with_no_step(self):
        """A count of an element of no width must not spin the walk.

        The count comes from the payload.  This plan reads it where the low
        half of a float64 sits, which states a count in the billions for
        most messages.  A body of no instruction advances nothing, so running
        it once per counted element would hold the walk for minutes.
        """
        reader = self.reader()
        stamp = sc.mcap.DecodePlan(reader.schema(SAMPLE),
                                   fields=["header.stamp"])
        self.assertEqual(stamp.instructions[-1], ("read", "float64", 0))
        steps = stamp.instructions[:-1] + (("skip_sequence_body", 0),
                                           ("read", "uint8", 0))
        _, columns = self.extract(reader, SAMPLE, 1, steps)
        self.assertEqual(len(columns[0]), SIGNAL_COUNT)

    def test_container_of_elements_that_advance_nothing(self):
        """A body with an instruction that consumes no byte must not spin.

        An alignment to one byte never moves the cursor, so a count in the
        billions would run the body that many times for nothing.
        """
        _, columns = self.extract(
            self.reader(), SAMPLE, 1,
            (("skip_array_body", 4294967295, 1), ("align", 1),
             ("read", "uint8", 0)))
        self.assertEqual(len(columns[0]), SIGNAL_COUNT)

    def test_walk_past_the_end_of_the_payload(self):
        with self.assertRaisesRegex(RuntimeError, "too short"):
            self.extract(self.reader(), SAMPLE, 1,
                         (("skip", 10000), ("read", "uint8", 0)))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
