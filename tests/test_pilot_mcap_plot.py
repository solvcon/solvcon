# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""The plot page of the MCAP viewer, fed from a generated recording."""

import os
import struct
import tempfile
import unittest

import solvcon
from solvcon.track import mcap

try:
    from solvcon import pilot
    from solvcon.pilot.track import _mcap_plot
except ImportError:
    pilot = _mcap_plot = None

try:
    from mcap import writer as foxglove_mcap_writer
except ImportError:
    foxglove_mcap_writer = None

STATE_IDL = b"""
module vehicle_msgs {
  module msg {
    enum Gear { PARK, DRIVE };
    struct State {
      double speed_mps;
      float motor_current_a;
      boolean active;
      Gear gear;
      string name;
      sequence<int32> codes;
    };
  };
};
"""

STATE_TOPIC = "/vehicle/state"
DIAG_TOPIC = "/diagnostics"
STATE_COUNT = 50
SECOND_NS = 1_000_000_000
# An epoch start: float64 would round off the nanosecond of each offset.
FILE_START_NS = 1_700_000_000 * SECOND_NS + 1
STATE_START_NS = FILE_START_NS + SECOND_NS - 1


def write_fixture(path):
    """Write a JSON topic at the file start, then one CDR topic at 10 Hz."""
    with open(path, "wb") as fp:
        writer = foxglove_mcap_writer.Writer(fp)
        writer.start(profile="ros2")
        schema_id = writer.register_schema("diagnostics", "jsonschema",
                                           b"{}")
        channel_id = writer.register_channel(DIAG_TOPIC, "json", schema_id)
        writer.add_message(channel_id, FILE_START_NS, b"{}", FILE_START_NS)
        schema_id = writer.register_schema("vehicle_msgs/msg/State",
                                           "ros2idl", STATE_IDL)
        channel_id = writer.register_channel(STATE_TOPIC, "cdr", schema_id)
        for i in range(STATE_COUNT):
            log_time = STATE_START_NS + i * 100_000_000
            # The gear, an empty string, and an empty sequence follow the
            # three scalars the plot draws.
            fields = struct.pack("<df?3xIII", i / 2, 10.0 + i, i % 2 == 0,
                                 i % 2, 1, 0) + b"\0" * 4
            writer.add_message(channel_id, log_time, b"\0\x01\0\0" + fields,
                               log_time)
        writer.finish()


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PlotHelperTC(unittest.TestCase):

    def test_time_ticks_fit_at_most_eight(self):
        self.assertEqual(_mcap_plot.time_ticks(0, 7), list(range(8)))
        self.assertEqual(_mcap_plot.time_ticks(0, 8), [0, 2, 4, 6, 8])
        self.assertEqual(_mcap_plot.time_ticks(5, 125),
                         [15, 30, 45, 60, 75, 90, 105, 120])
        self.assertEqual(_mcap_plot.time_ticks(0, 1200),
                         [it * 120 for it in range(0, 11)])
        self.assertEqual(_mcap_plot.time_ticks(3, 3), [])
        self.assertEqual([_mcap_plot.format_mss(s) for s in (0, 5, 75, 120)],
                         ["0:00", "0:05", "1:15", "2:00"])

    def test_value_ticks_are_nice_and_pad_the_data(self):
        ymin, ymax, _ = _mcap_plot.value_ticks(0.0, 24.5)
        self.assertAlmostEqual(ymin, -1.225)
        self.assertAlmostEqual(ymax, 25.725)
        ymin, ymax, _ = _mcap_plot.value_ticks(4.0, 4.0)
        self.assertEqual((ymin, ymax), (2.0, 6.0))
        ymin, ymax, ticks = _mcap_plot.value_ticks(0.0, 0.0)
        self.assertEqual((ymin, ymax), (-1.0, 1.0))


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
@unittest.skipIf(foxglove_mcap_writer is None,
                 "the Foxglove mcap package is not installed")
class PlotPageTC(unittest.TestCase):
    """The page fed a decoded topic and the span of the file."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "drive.mcap")
        write_fixture(self.path)
        self.reader = mcap.Reader(self.path)
        self.page = _mcap_plot.McapPlotPage()

    def tearDown(self):
        self.reader.close()
        self.tmpdir.cleanup()

    def show(self, topic):
        plan = mcap.DecodePlan(self.reader.schema(topic))
        extraction = self.reader.extract(topic, plan)
        self.page.show_topic(topic, extraction, plan,
                             self.reader.time_range())

    def test_a_topic_offers_its_fields_and_draws_the_first(self):
        self.show(STATE_TOPIC)
        self.assertEqual(self.page.topic, STATE_TOPIC)
        self.assertEqual(self.page.field, "speed_mps")
        self.assertEqual(
            [self.page._field.itemText(i)
             for i in range(self.page._field.count())],
            ["speed_mps \u00b7 float64", "motor_current_a \u00b7 float32",
             "active \u00b7 bool"])
        plot = self.page.plot
        self.assertEqual(plot.title, "speed_mps")
        # The file is shorter than the default span, so the range is
        # clamped to it; the abscissa counts from the file start.
        self.assertAlmostEqual(plot.range[1], 5.9)
        times, values = plot._visible
        self.assertEqual(len(times), STATE_COUNT)
        self.assertEqual(times[0], 0.999999999)
        self.assertAlmostEqual(values[-1], 24.5)
        self.assertEqual(plot._limits[2],
                         [0.0, 5.0, 10.0, 15.0, 20.0, 25.0])

        # Samples the range ends on stay in, twins included; a value
        # that is not finite is dropped before it can wreck the scale.
        plot.set_range(0.0, 1.5)
        plot.set_series("x", False, [0.0, 1.0, 1.5, 1.5, 2.0],
                        [1.0, float("nan"), 2.0, 3.0, 4.0])
        self.assertEqual(list(plot._visible[1]), [1.0, 2.0, 3.0])
        plot.set_series("x", False, [0.0, 1.0], [1.0, float("inf")])
        self.assertEqual(plot._limits[:2], (0.5, 1.5))

    def test_the_dropdown_switches_the_field_and_says_so(self):
        chosen = []
        self.page.field_changed.connect(chosen.append)
        self.show(STATE_TOPIC)
        self.assertEqual(chosen, [])

        self.page._field.setCurrentIndex(1)
        self.page._field.activated.emit(1)
        self.assertEqual(chosen, ["motor_current_a"])
        plot = self.page.plot
        self.assertEqual(plot.title, "motor_current_a")
        self.assertFalse(plot._boolean)
        self.assertEqual(plot._limits[2],
                         [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        self.page.set_field("active")
        self.assertEqual(chosen, ["motor_current_a"])
        self.assertEqual(self.page.field, "active")
        self.assertTrue(plot._boolean)
        self.assertEqual(plot.title, "active")
        self.assertEqual(plot._limits,
                         (-0.25, 1.25, [0.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "'gear' is not"):
            self.page.set_field("gear")
        self.assertEqual(self.page.field, "active")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
