# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Regenerate the tiny MCAP recordings that the tests read.

Run it from the repository root::

  python3 tests/data/make_mcap_fixtures.py

It needs the pip packages mcap, lz4, and zstandard.

The script writes the vehicle recordings that tests/test_mcap.py reads, one
per chunk compression the reader supports. It writes one more without chunks,
for the path that has no chunk index to prune with. The chunk size is 256
bytes, so the 96 status messages spread over 19 chunks. The imu topic carries
one message per twelve status messages. It reaches only 8 of the 19 chunks,
so a query for it leaves the rest unread.

The script also writes the signals recording that tests/test_mcap_extract.py
reads. That recording carries IDL schemas instead, because a decode plan
compiles from IDL. Its two topics hold the field layouts a plan must walk:

- primitives of every width, a sequence, a string, and a nested struct in
  the sample topic;
- a struct inside a sequence and inside a fixed array in the wheel topic.

The payloads use the Common Data Representation (CDR) that a ROS 2 recording
carries.
"""

import os
import struct

from mcap.writer import CompressionType, Writer

# Topic, schema name, and the message definition text of the schema.
TOPICS = (
    ("/vehicle/status", "vhcl_msgs/msg/Status",
     "float64 longitudinal_speed\nbool brake_active\n"),
    ("/vehicle/imu", "vhcl_msgs/msg/Imu",
     "float64 ax\nfloat64 ay\nfloat64 az\n"),
)

IDL_TOPICS = (
    ("/signals/sample", "sig_msgs/msg/Sample", """\
================================================================================
IDL: sig_msgs/msg/Sample
module sig_msgs { module msg {
  struct Sample {
    octet flags;
    double speed;
    sequence<float> taps;
    string label;
    Header header;
    boolean valid;
    int16 gear;
    uint64 odometer;
  };
}; };
================================================================================
IDL: sig_msgs/msg/Header
module sig_msgs { module msg {
  struct Header {
    uint32 seq;
    double stamp;
  };
}; };
"""),
    ("/signals/wheel", "sig_msgs/msg/Wheel", """\
================================================================================
IDL: sig_msgs/msg/Wheel
module sig_msgs { module msg {
  struct Corner {
    double load;
    string tag;
  };
  struct Wheel {
    sequence<Corner> corners;
    Corner fixed[2];
    double slip;
  };
}; };
"""),
)

SIGNAL_COUNT = 40
# One wheel message per this many sample messages.
WHEEL_PERIOD = 5

COMPRESSIONS = (
    ("none", CompressionType.NONE),
    ("lz4", CompressionType.LZ4),
    ("zstd", CompressionType.ZSTD),
)

MESSAGE_COUNT = 96
# One imu message per this many status messages.
IMU_PERIOD = 12
# Log time of the first message and the interval between them, in
# nanoseconds.
START_TIME_NS = 1700000000000000000
PERIOD_NS = 10000000

# Little-endian CDR encapsulation header.
CDR_HEADER = b"\x00\x01\x00\x00"


class Cdr:
    """Writer of a CDR body, padding every field to its own width.

    The encapsulation header is four bytes, and the alignment of a field
    counts from the byte after the header.  The buffer therefore holds the
    body alone, and ``payload()`` puts the header in front of the body.
    """

    def __init__(self):
        self.body = bytearray()

    def pack(self, fmt, *values):
        size = struct.calcsize("<" + fmt)
        self.body += b"\x00" * (-len(self.body) % size)
        self.body += struct.pack("<" + fmt, *values)

    def pack_string(self, text):
        # The stated length counts the terminating byte the text carries.
        raw = text.encode() + b"\x00"
        self.pack("I", len(raw))
        self.body += raw

    def pack_sequence(self, fmt, values):
        self.pack("I", len(values))
        for value in values:
            self.pack(fmt, value)

    def payload(self):
        return CDR_HEADER + bytes(self.body)


def sample_payload(index):
    cdr = Cdr()
    cdr.pack("B", index % 256)
    cdr.pack("d", 1.5 * index)
    cdr.pack_sequence("f", [0.5 * tap for tap in range(index % 3)])
    cdr.pack_string("sample-%d" % index)
    cdr.pack("I", index)
    cdr.pack("d", 0.001 * index)
    cdr.pack("B", int(0 == index % 2))
    cdr.pack("h", index % 8 - 4)
    cdr.pack("Q", 1000 * index + 7)
    return cdr.payload()


def wheel_payload(index):
    cdr = Cdr()
    cdr.pack("I", index % 3)
    for corner in range(index % 3):
        cdr.pack("d", index + 0.25 * corner)
        cdr.pack_string("c%d" % corner)
    for corner in range(2):
        cdr.pack("d", -float(index + corner))
        cdr.pack_string("f%d" % corner)
    cdr.pack("d", 0.125 * index)
    return cdr.payload()


def status_payload(index):
    return CDR_HEADER + struct.pack("<dB", 1.5 * index, index % 2)


def imu_payload(index):
    return CDR_HEADER + struct.pack("<ddd", index, -index, 9.81)


def write_messages(writer, topics, encoding, payloads, count, period):
    """Write the messages of two topics.

    The first topic gets one message per index; the second gets one per
    ``period`` indices.
    """
    channels = []
    for topic, name, text in topics:
        schema_id = writer.register_schema(
            name=name, encoding=encoding, data=text.encode())
        channels.append(writer.register_channel(
            topic=topic, message_encoding="cdr", schema_id=schema_id))

    for index in range(count):
        log_time = START_TIME_NS + index * PERIOD_NS
        messages = [(channels[0], payloads[0](index))]
        if 0 == index % period:
            messages.append((channels[1], payloads[1](index)))
        for channel_id, payload in messages:
            writer.add_message(
                channel_id=channel_id, log_time=log_time,
                publish_time=log_time, sequence=index, data=payload)


def write(path, compression, chunked):
    with open(path, "wb") as stream:
        writer = Writer(stream, chunk_size=256, compression=compression,
                        use_chunking=chunked)
        writer.start(profile="ros2", library="solvcon fixture")
        write_messages(writer, TOPICS, "ros2msg",
                       (status_payload, imu_payload), MESSAGE_COUNT,
                       IMU_PERIOD)
        writer.finish()


def write_signals(path):
    with open(path, "wb") as stream:
        writer = Writer(stream, chunk_size=1024,
                        compression=CompressionType.ZSTD)
        writer.start(profile="ros2", library="solvcon fixture")
        write_messages(writer, IDL_TOPICS, "ros2idl",
                       (sample_payload, wheel_payload), SIGNAL_COUNT,
                       WHEEL_PERIOD)
        writer.finish()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    plan = [("vehicle_%s" % name, compression, True)
            for name, compression in COMPRESSIONS]
    plan.append(("vehicle_unchunked", CompressionType.NONE, False))
    for name, compression, chunked in plan:
        path = os.path.join(here, "%s.mcap" % name)
        write(path, compression, chunked)
        print("wrote %s (%d bytes)" % (path, os.path.getsize(path)))
    path = os.path.join(here, "signals.mcap")
    write_signals(path)
    print("wrote %s (%d bytes)" % (path, os.path.getsize(path)))


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
