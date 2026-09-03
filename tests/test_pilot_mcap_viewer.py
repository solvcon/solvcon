# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""The MCAP dock filled from a generated recording."""

import os
import tempfile
import unittest

import solvcon
from solvcon.track import mcap

try:
    from solvcon import pilot
    from solvcon.pilot.track import _mcap_viewer
except ImportError:
    pilot = None

try:
    from mcap import writer as foxglove_mcap_writer
except ImportError:
    foxglove_mcap_writer = None

BRAKE_IDL = b"""
module vehicle_msgs {
  module msg {
    struct Brake {
      boolean active;
    };
  };
};
"""

BRAKE_TOPIC = "/vehicle/brake"
BRAKE_PAYLOAD = b"\0\x01\0\0\x01"
DIAG_TOPIC = "/diagnostics"
SECOND_NS = 1_000_000_000


def write_fixture(path):
    """One CDR topic with an IDL schema and one JSON topic."""
    with open(path, "wb") as fp:
        writer = foxglove_mcap_writer.Writer(fp)
        writer.start(profile="ros2")
        schema_id = writer.register_schema("vehicle_msgs/msg/Brake",
                                           "ros2idl", BRAKE_IDL)
        channel_id = writer.register_channel(BRAKE_TOPIC, "cdr", schema_id)
        for log_time in (3 * SECOND_NS, 5 * SECOND_NS, 8 * SECOND_NS):
            writer.add_message(channel_id, log_time, BRAKE_PAYLOAD, log_time)
        schema_id = writer.register_schema("diagnostics", "jsonschema",
                                           b"{}")
        channel_id = writer.register_channel(DIAG_TOPIC, "json", schema_id)
        writer.add_message(channel_id, 4 * SECOND_NS, b"{}", 4 * SECOND_NS)
        writer.finish()


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
@unittest.skipIf(foxglove_mcap_writer is None,
                 "the Foxglove mcap package is not installed")
class McapDockTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "drive.mcap")
        write_fixture(self.path)
        self.reader = mcap.Reader(self.path)

    def tearDown(self):
        self.reader.close()
        self.tmpdir.cleanup()

    def test_dock_lists_the_file_its_topics_and_the_one_selected(self):
        dock = _mcap_viewer.McapDock()
        dock.set_reader(self.reader)
        self.assertEqual(dock._file.text(), "drive.mcap")
        self.assertEqual(dock._file.toolTip(), self.path)
        self.assertEqual(dock._size.text(),
                         _mcap_viewer.format_size(self.reader.size))
        self.assertEqual(dock._duration.text(), "5s")
        self.assertEqual(dock._messages.text(), "4")
        self.assertEqual(dock._caption.text(), "TOPICS \u00b7 2")

        topics = dock._topics
        self.assertEqual(
            [[topics.item(i).data(role) for role in (
                _mcap_viewer._TOPIC_ROLE, _mcap_viewer._COUNT_ROLE,
                _mcap_viewer._TYPE_ROLE)]
             for i in range(topics.count())],
            [[BRAKE_TOPIC, "3", "vehicle_msgs/msg/Brake"],
             [DIAG_TOPIC, "1", "diagnostics"]])

        dock.set_reader(None)
        self.assertEqual([label.text() for label in dock._values], ["-"] * 4)
        self.assertEqual(dock._caption.text(), "TOPICS")
        self.assertEqual(topics.count(), 0)

    def test_a_file_without_statistics_reports_unknown(self):
        path = os.path.join(self.tmpdir.name, "nostat.mcap")
        with open(path, "wb") as fp:
            writer = foxglove_mcap_writer.Writer(fp, use_statistics=False)
            writer.start(profile="ros2")
            schema_id = writer.register_schema("vehicle_msgs/msg/Brake",
                                               "ros2idl", BRAKE_IDL)
            channel_id = writer.register_channel(BRAKE_TOPIC, "cdr", schema_id)
            writer.add_message(channel_id, SECOND_NS, BRAKE_PAYLOAD,
                               SECOND_NS)
            writer.finish()

        with mcap.Reader(path) as reader:
            size = _mcap_viewer.format_size(os.path.getsize(path))
            self.assertEqual(_mcap_viewer.summarize(reader),
                             (size, "unknown", "unknown"))
            dock = _mcap_viewer.McapDock()
            dock.set_reader(reader)
            self.assertEqual(dock._topics.item(0).data(
                _mcap_viewer._COUNT_ROLE), "?")

    def test_a_file_that_will_not_open_is_reported_in_the_dock(self):
        path = os.path.join(self.tmpdir.name, "broken.mcap")
        with open(path, "wb") as fp:
            fp.write(b"not an MCAP file")

        dock = _mcap_viewer.McapDock()
        dock.set_reader(self.reader)
        with self.assertRaisesRegex(mcap.McapError, "bad magic"):
            mcap.Reader(path)
        dock.set_error(path, "bad magic")
        self.assertEqual(dock._file.text(), "broken.mcap")
        self.assertEqual(dock._error.text(), "bad magic")
        self.assertFalse(dock._error.isHidden())
        self.assertEqual(dock._topics.count(), 0)
        dock.set_reader(self.reader)
        self.assertTrue(dock._error.isHidden())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
