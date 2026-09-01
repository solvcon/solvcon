# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Walk through the MCAP reader: from a recording to signal verdicts.

The steps are the ones a drive-log checker takes:

1. Open the recording and list its topics.
2. Compile a decode plan per topic and extract the selected fields into
   columns in one pass over the file.
3. Build a sorted ``DataFrame`` per topic.
4. Align the brake topic onto the speed clock with ``DataFrame.asof``.
5. Run the time-series kernels: derivative, moving average, threshold,
   and interval detection.

Run without arguments to generate a synthetic recording in a temporary
directory with the Foxglove ``mcap`` writer, or pass ``--path`` to read an
existing one that carries the same two topics::

    PYTHONPATH=. python3 examples/track_mcap_signals.py
"""

import argparse
import os
import struct
import tempfile

import numpy as np

import solvcon as sc
from solvcon import timeseries
from solvcon.track import mcap

NS = 1_000_000_000
SPEED_TOPIC = "/vehicle/status"
BRAKE_TOPIC = "/vehicle/brake"

STATUS_IDL = b"""
module vehicle_msgs {
  module msg {
    enum Mode {
      OFF,
      MANUAL,
      AUTONOMOUS
    };
    struct Status {
      double longitudinal_speed;
      vehicle_msgs::msg::Mode mode;
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


def pack_status(speed, mode):
    return b"\0\x01\0\0" + struct.pack("<dI", speed, mode)


def pack_brake(active):
    return b"\0\x01\0\0" + struct.pack("<?", active)


def foxglove_write_data(path):
    """
    Record 20 s of driving: speed at 10 Hz, brake at 2 Hz.

    The Foxglove ``mcap`` writer writes the file.

    The car accelerates for 8 s, cruises, and brakes hard from 14 s to
    17 s.  The brake is held from 14 s to 18 s.
    """
    from mcap import writer as foxglove_mcap_writer
    with open(path, "wb") as fp:
        writer = foxglove_mcap_writer.Writer(
            fp, compression=foxglove_mcap_writer.CompressionType.ZSTD)
        writer.start(profile="ros2")

        schema_id = writer.register_schema("vehicle_msgs/msg/Status",
                                           "ros2idl", STATUS_IDL)
        channel_id = writer.register_channel(SPEED_TOPIC, "cdr", schema_id)
        for tick in range(200):
            t = tick / 10.0
            if t < 8.0:
                speed = 2.5 * t
            elif t < 14.0:
                speed = 20.0
            else:
                speed = max(0.0, 20.0 - 6.0 * (t - 14.0))
            mode = 2 if 2.0 <= t < 18.0 else 1
            writer.add_message(channel_id, int(t * NS),
                               pack_status(speed, mode), int(t * NS))

        schema_id = writer.register_schema("vehicle_msgs/msg/Brake",
                                           "ros2idl", BRAKE_IDL)
        channel_id = writer.register_channel(BRAKE_TOPIC, "cdr", schema_id)
        for tick in range(40):
            t = tick / 2.0
            writer.add_message(channel_id, int(t * NS),
                               pack_brake(14.0 <= t < 18.0), int(t * NS))
        writer.finish()


def seconds(times):
    return np.asarray(times) / NS


def report_intervals(title, times, flags):
    rows = timeseries.true_intervals(times, flags).ndarray
    print("{}: {} interval(s)".format(title, len(rows)))
    for start, end, duration in rows:
        print("  {:6.2f} s to {:6.2f} s ({:.2f} s)".format(
            start / NS, end / NS, duration / NS))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--path", help="an existing recording to read")
    args = parser.parse_args()

    # Without --path, write a synthetic recording into a temporary directory.
    tmpdir = None
    path = args.path
    if path is None:
        tmpdir = tempfile.TemporaryDirectory()
        path = os.path.join(tmpdir.name, "drive.mcap")
        foxglove_write_data(path)

    # Step 1: the constructor reads only the summary section.
    with mcap.Reader(path) as reader:
        print("profile:", reader.profile)
        for topic, schema_name in sorted(reader.topics().items()):
            print("  {} : {}".format(topic, schema_name))
        start_ns, end_ns = reader.time_range()
        print("time range: {:.1f} s to {:.1f} s".format(
            start_ns / NS, end_ns / NS))

        # Step 2: compile once per topic; no fields selects every scalar leaf.
        # the plan can be reused for every message of the topic.
        plan = mcap.DecodePlan(reader.schema(SPEED_TOPIC))
        print("plan of {}: fields {} types {} enums {}\n".format(
            SPEED_TOPIC, plan.fields, plan.types, plan.enums))

        # Steps 2 and 3: one pass over the file yields a sorted frame per
        # topic.  A plan and a path list are both valid selections.
        frames = reader.extract_frame_many({SPEED_TOPIC: plan,
                                            BRAKE_TOPIC: ["active"]})

    if tmpdir is not None:
        tmpdir.cleanup()

    speed_frame = frames[SPEED_TOPIC]
    brake_frame = frames[BRAKE_TOPIC]
    print("{}: {} rows, columns {}".format(SPEED_TOPIC, speed_frame.shape[0],
                                           speed_frame.columns))
    print("{}: {} rows, columns {}\n".format(BRAKE_TOPIC, brake_frame.shape[0],
                                             brake_frame.columns))

    # Step 4: the 10 Hz speed index is the clock.  asof takes the last brake
    # sample at or before each tick; brake_known is false before the first.
    clock = sc.SimpleArrayUint64(array=speed_frame.index)
    speed = sc.SimpleArrayFloat64(array=speed_frame["longitudinal_speed"])
    brake, brake_known = speed_frame.asof(
        sc.SimpleArrayUint64(array=brake_frame.index),
        sc.SimpleArrayBool(array=brake_frame["active"]))
    print("brake aligned onto the speed clock: {} of {} samples known\n"
          .format(int(np.count_nonzero(brake_known.ndarray)), len(clock)))

    # Step 5: deriv differentiates per nanosecond; scale to per second.
    accel_times, accel = timeseries.deriv(clock, speed)
    accel_per_s = accel.ndarray * NS
    print("acceleration: min {:.2f} m/s^2, max {:.2f} m/s^2".format(
        accel_per_s.min(), accel_per_s.max()))

    # Smooth over the trailing 1 s window before the threshold test.
    smooth_times, smooth = timeseries.movavg(
        accel_times, sc.SimpleArrayFloat64(array=accel_per_s), NS)
    print("1 s moving average: min {:.2f} m/s^2, max {:.2f} m/s^2\n".format(
        smooth.ndarray.min(), smooth.ndarray.max()))

    # A threshold turns a series into a flag; true_intervals reports the
    # stretches where the flag holds.
    hard_braking = sc.SimpleArrayBool(array=smooth.ndarray < -3.0)
    report_intervals("hard braking (average below -3 m/s^2)", smooth_times,
                     hard_braking)

    # held is true only after the brake stays on for the whole 2 s window.
    held_times, held = timeseries.held(clock, brake, 2 * NS)
    report_intervals("brake held for 2 s", held_times, held)

    # The enum column is numeric; plan.enums maps a member name to it.
    mode = speed_frame["mode"]
    names = plan.enums["mode"]
    autonomous = sc.SimpleArrayBool(
        array=mode == names.index("AUTONOMOUS"))
    report_intervals("mode == AUTONOMOUS", clock, autonomous)


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
