# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Write ``fake_recording.mcap`` with the Foxglove ``mcap`` writer.
The fixture imitates the shape of a real vehicle recording.

The chunks stay uncompressed, because the reader imports ``zstandard``
and ``lz4`` only when a chunk needs them and neither is a dependency of
solvcon.  A small chunk size spreads the messages over several chunks,
so a reader must go through the chunk index.  Each schema bundles its
IDL blocks under a separator line of 80 ``=``, the width a ROS 2
decoder expects.

Regenerate with::

    python3 tests/data/mcap/make_fake_recording.py
"""

import math
import os
import struct

from mcap.writer import CompressionType, Writer

SEPARATOR = b"=" * 80 + b"\n"

HEADER_BLOCK = b"""\
IDL: sim_msgs/msg/Header
module sim_msgs {
  module msg {
    struct Header {
      string<64> module_name;

      uint32 sequence_number;

      uint64 stamp_ns;
    };
  };
};
"""

POINT2_BLOCK = b"""\
IDL: geom2_msgs/msg/Point2
module geom2_msgs {
  module msg {
    struct Point2 {
      double x;

      double y;
    };
  };
};
"""

EGO_STATE_BLOCK = b"""\
IDL: sim_msgs/msg/EgoState
#include "sim_msgs/msg/Header.idl"
#include "geom2_msgs/msg/Point2.idl"

module sim_msgs {
  module msg {
    struct EgoState {
      sim_msgs::msg::Header header;

      @verbatim (language="comment", text=
        " Vehicle reference point in the track frame, in meter.")
      geom2_msgs::msg::Point2 position;

      @verbatim (language="comment", text=
        " Velocity of the reference point, in meter per second.")
      geom2_msgs::msg::Point2 velocity;

      double heading;  /* radian, zero along the x axis */

      float steering;  /* radian at the wheel */
    };
  };
};
"""

DRIVE_MODE_BLOCK = b"""\
IDL: sim_msgs/msg/DriveMode
#include "sim_msgs/msg/Header.idl"

module sim_msgs {
  module msg {
    /**
     * How much of the driving the stack is doing.
     */
    enum ModeKind {
      MANUAL,
      ASSIST,
      AUTO,
      DEGRADED
    };

    struct DriveMode {
      sim_msgs::msg::Header header;

      ModeKind mode;

      boolean brake_active;

      float target_speed;  /* meter per second */
    };
  };
};
"""

FAULT_REPORT_BLOCK = b"""\
IDL: sim_msgs/msg/FaultReport
#include "sim_msgs/msg/Header.idl"

module sim_msgs {
  module msg {
    enum FaultCode {
      FAULT_NONE,
      SENSOR_TIMEOUT,
      ACTUATOR_LIMIT,
      PLAN_INFEASIBLE,
      LOCALIZATION_DRIFT
    };

    const uint16 kFaultCapacity = 8;

    struct FaultReport {
      sim_msgs::msg::Header header;

      sequence<FaultCode, kFaultCapacity> codes;

      string<128> note;
    };
  };
};
"""


def bundle(*blocks):
    """Return the IDL blocks joined under separator lines."""
    return b"".join(SEPARATOR + block for block in blocks)


EGO_STATE_IDL = bundle(EGO_STATE_BLOCK, HEADER_BLOCK, POINT2_BLOCK)
DRIVE_MODE_IDL = bundle(DRIVE_MODE_BLOCK, HEADER_BLOCK)
FAULT_REPORT_IDL = bundle(FAULT_REPORT_BLOCK, HEADER_BLOCK)

EGO_TOPIC = "/sim/ego/state"
FILTERED_TOPIC = "/sim/ego/state_filtered"
MODE_TOPIC = "/sim/ego/drive_mode"
FAULT_TOPIC = "/sim/diag/fault_report"

COUNT = 6
START_NS = 1_000_000_000
PERIOD_NS = 10_000_000
MODE_OFFSET_NS = 1_000_000
FAULT_OFFSET_NS = 3_000_000

FAULT_SEQUENCES = [[], [1], [], [2, 4], [2], [3, 4, 1]]
NOTES = ["nominal", "nominal", "nominal", "sensor recheck",
         "sensor recheck", "replan requested"]


class Cdr:
    """Build a little-endian XCDR1 body and align every field."""

    def __init__(self):
        self._buf = bytearray()

    def scalar(self, fmt, value):
        self._buf += b"\0" * (-len(self._buf) % struct.calcsize(fmt))
        self._buf += struct.pack("<" + fmt, value)
        return self

    def string(self, text):
        data = text.encode() + b"\0"
        self.scalar("I", len(data))
        self._buf += data
        return self

    def sequence(self, fmt, values):
        self.scalar("I", len(values))
        for value in values:
            self.scalar(fmt, value)
        return self

    def header(self, module_name, sequence_number, stamp_ns):
        return (self.string(module_name).scalar("I", sequence_number)
                .scalar("Q", stamp_ns))

    def payload(self):
        return b"\0\x01\0\0" + bytes(self._buf)


def stamp_of(index, offset_ns=0):
    return START_NS + index * PERIOD_NS + offset_ns


def seconds_of(index):
    return index * PERIOD_NS / 1e9


def speed_of(index):
    """Return the cruise speed in meter per second."""
    return 9.0 + 3.0 * math.sin(0.02 * seconds_of(index))


def emit(writer, channel, index, offset_ns, payload):
    """Add one message whose log time and publish time are the same."""
    stamp = stamp_of(index, offset_ns)
    writer.add_message(channel, stamp, payload, stamp)


def pack_ego_state(index, module_name, drift):
    """Return one ego state of a vehicle weaving down the x axis."""
    time = seconds_of(index)
    speed = speed_of(index)
    heading = 0.05 * math.sin(0.01 * time)
    return (Cdr()
            .header(module_name, index, stamp_of(index))
            .scalar("d", 9.0 * time + 150.0 * (1.0 - math.cos(0.02 * time))
                    + drift)
            .scalar("d", 1.5 * math.sin(0.01 * time))
            .scalar("d", speed * math.cos(heading))
            .scalar("d", speed * math.sin(heading))
            .scalar("d", heading)
            .scalar("f", 0.02 * math.sin(0.01 * time))
            .payload())


def pack_drive_mode(index):
    """Return one drive mode: assist, then auto, then degraded."""
    if 10 * index < COUNT:
        mode = 1
    elif 10 * (index + 1) > 9 * COUNT:
        mode = 3
    else:
        mode = 2
    return (Cdr()
            .header("mode_arbiter", index, stamp_of(index, MODE_OFFSET_NS))
            .scalar("I", mode)
            .scalar("?", index in (0, COUNT - 1))
            .scalar("f", speed_of(index))
            .payload())


def pack_fault_report(index):
    """Return one fault report from a repeating cycle of codes."""
    return (Cdr()
            .header("diagnostics", index, stamp_of(index, FAULT_OFFSET_NS))
            .sequence("I", FAULT_SEQUENCES[index % len(FAULT_SEQUENCES)])
            .string(NOTES[index % len(NOTES)])
            .payload())


def main():
    """Write the recording next to this script."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fake_recording.mcap")
    with open(path, "wb") as fp:
        writer = Writer(fp, chunk_size=2048,
                        compression=CompressionType.NONE)
        writer.start(profile="ros2", library="solvcon fixture generator")
        writer.add_metadata("recording", {"track": "figure_eight",
                                          "run": "0007"})

        ego_schema = writer.register_schema("sim_msgs/msg/EgoState",
                                            "ros2idl", EGO_STATE_IDL)
        ego = writer.register_channel(EGO_TOPIC, "cdr", ego_schema)
        filtered = writer.register_channel(FILTERED_TOPIC, "cdr", ego_schema)
        mode_schema = writer.register_schema("sim_msgs/msg/DriveMode",
                                             "ros2idl", DRIVE_MODE_IDL)
        mode = writer.register_channel(MODE_TOPIC, "cdr", mode_schema)
        fault_schema = writer.register_schema("sim_msgs/msg/FaultReport",
                                              "ros2idl", FAULT_REPORT_IDL)
        fault = writer.register_channel(FAULT_TOPIC, "cdr", fault_schema)

        for index in range(COUNT):
            emit(writer, ego, index, 0,
                 pack_ego_state(index, "odometry", 0.0))
            emit(writer, filtered, index, 0,
                 pack_ego_state(index, "odometry_filter", 0.12))
            emit(writer, mode, index, MODE_OFFSET_NS,
                 pack_drive_mode(index))
            emit(writer, fault, index, FAULT_OFFSET_NS,
                 pack_fault_report(index))
        writer.finish()

    print("{} ({} bytes)".format(path, os.path.getsize(path)))


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
