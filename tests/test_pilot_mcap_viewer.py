# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""The MCAP dock and main window filled from a generated recording."""

import os
import struct
import tempfile
import unittest

import solvcon
from solvcon.track import mcap

try:
    from solvcon import pilot
    from solvcon.pilot.track import _mcap_viewer
    from solvcon.pilot.track import _mcap_viewer_main_window as _main_window
    from PySide6.QtCore import Qt
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

STATUS_IDL = b"""
module vehicle_msgs {
  module msg {
    struct Status {
      uint32 seq;
      double speed;
      boolean active;
    };
  };
};
"""

BRAKE_TOPIC = "/vehicle/brake"
BRAKE_PAYLOAD = b"\0\x01\0\0\x01"
DIAG_TOPIC = "/diagnostics"
STATUS_TOPIC = "/vehicle/status"
STATUS_COUNT = 120
SECOND_NS = 1_000_000_000
START_NS = 10 * SECOND_NS


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
        schema_id = writer.register_schema("vehicle_msgs/msg/Status",
                                           "ros2idl", STATUS_IDL)
        channel_id = writer.register_channel(STATUS_TOPIC, "cdr", schema_id)
        for i in range(STATUS_COUNT):
            log_time = START_NS + i * 10_000_000
            fields = struct.pack("<I4xd?", i, i / 4, i % 2 == 0)
            payload = b"\0\x01\0\0" + fields
            writer.add_message(channel_id, log_time, payload, log_time)
        writer.finish()


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
@unittest.skipIf(foxglove_mcap_writer is None,
                 "the Foxglove mcap package is not installed")
class _RecordingTC(unittest.TestCase):
    """A generated recording open in a reader for the length of a test."""

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


class McapDockTC(_RecordingTC):

    def test_dock_lists_the_file_its_topics_and_the_one_selected(self):
        dock = _mcap_viewer.McapDock()
        dock.set_reader(self.reader)
        self.assertEqual(dock._file.text(), "drive.mcap")
        self.assertEqual(dock._file.toolTip(), self.path)
        self.assertEqual(dock._size.text(),
                         _mcap_viewer.format_size(self.reader.size))
        self.assertEqual(dock._duration.text(), "8s")
        self.assertEqual(dock._messages.text(), "124")
        self.assertEqual(dock._caption.text(), "TOPICS \u00b7 3")

        topics = dock._topics
        self.assertEqual(
            [[topics.item(i).data(role) for role in (
                _mcap_viewer._TOPIC_ROLE, _mcap_viewer._COUNT_ROLE,
                _mcap_viewer._TYPE_ROLE)]
             for i in range(topics.count())],
            [[BRAKE_TOPIC, "3", "vehicle_msgs/msg/Brake"],
             [DIAG_TOPIC, "1", "diagnostics"],
             [STATUS_TOPIC, "120", "vehicle_msgs/msg/Status"]])
        selected = []
        dock.topic_selected.connect(selected.append)
        topics.itemClicked.emit(topics.item(1))
        topics.itemActivated.emit(topics.item(2))
        self.assertEqual(selected, [DIAG_TOPIC, STATUS_TOPIC])

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


class McapMainWindowTC(_RecordingTC):

    def test_go_through(self):
        window = _main_window.McapMainWindow(self.reader)
        self.assertEqual(window.title, "MCAP viewer - drive.mcap")
        self.assertIsNone(window.model)
        self.assertIsNone(window.page(1))
        self.assertIs(window._pages.currentWidget(), window._notice)

        # The topic table, its headers, and its first page.
        model = window.table(STATUS_TOPIC)
        self.assertEqual(window.topic, STATUS_TOPIC)
        self.assertEqual(window._type.text(), "vehicle_msgs/msg/Status")
        self.assertEqual(window._summary.text(),
                         "120 messages \u00b7 3 columns")
        self.assertEqual(
            [(model.headerData(i, Qt.Horizontal),
              model.headerData(i, Qt.Horizontal, Qt.ToolTipRole))
             for i in range(model.columnCount())],
            [("#", ""), ("log_time", ""), ("seq", "uint32"),
             ("speed", "float64"), ("active", "bool")])

        def row(irow):
            return [model.data(model.index(irow, i))
                    for i in range(model.columnCount())]

        self.assertEqual(model.rowCount(), 50)
        self.assertEqual(row(0), ["0", "1970-01-01 00:00:10.000000", "0",
                                  "0.0", "true"])
        self.assertEqual(row(49), ["49", "1970-01-01 00:00:10.490000", "49",
                                   "12.25", "false"])
        self.assertEqual(window._range.text(), "Rows 1\u201350 of 120")
        self.assertEqual(window._page_total.text(), "of 3")
        self.assertFalse(window._prev.isEnabled())

        # Paging by API, by the input field, and by the buttons.
        self.assertEqual(window.page(3), 3)
        self.assertEqual(model.rowCount(), 20)
        self.assertEqual(row(0)[:2], ["100", "1970-01-01 00:00:11.000000"])
        self.assertEqual(window._range.text(), "Rows 101\u2013120 of 120")
        self.assertFalse(window._next.isEnabled())
        self.assertEqual(window.page(99), 3)
        self.assertEqual(window.page(0), 1)
        window._page_input.setText("2")
        window._page_input.returnPressed.emit()
        self.assertEqual(model.page, 2)
        window._next.click()
        self.assertEqual(model.page, 3)
        window._prev.click()
        self.assertEqual(model.page, 2)
        self.assertEqual(window._page_input.text(), "2")

        # A topic the decoder cannot read says why.
        self.assertIsNone(window.table(DIAG_TOPIC))
        self.assertIsNone(window.model)
        self.assertEqual(window._type.text(), "diagnostics")
        self.assertIs(window._pages.currentWidget(), window._notice)
        self.assertEqual(
            window._notice.text(),
            "Cannot decode /diagnostics: "
            "schema encoding 'jsonschema' of diagnostics")

        self.assertEqual(
            _main_window.format_time(1755765120 * SECOND_NS + 123_456_789),
            "2025-08-21 08:32:00.123456")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
