# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Regenerate the tiny MCAP recordings that tests/test_mcap.py reads.

Run it from the repository root::

  python3 tests/data/make_mcap_fixtures.py

It needs the pip packages mcap, lz4, and zstandard.

The script writes one recording per chunk compression the reader supports.
It writes one more without chunks, for the path that has no chunk index to
prune with. The chunk size is 256 bytes, so the 96 status messages spread
over 19 chunks. The imu topic carries one message per twelve status
messages. It reaches only 8 of the 19 chunks, so a query for it leaves the
rest unread.

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


def status_payload(index):
    return CDR_HEADER + struct.pack("<dB", 1.5 * index, index % 2)


def imu_payload(index):
    return CDR_HEADER + struct.pack("<ddd", index, -index, 9.81)


def write(path, compression, chunked):
    with open(path, "wb") as stream:
        writer = Writer(stream, chunk_size=256, compression=compression,
                        use_chunking=chunked)
        writer.start(profile="ros2", library="solvcon fixture")
        channels = []
        for topic, name, text in TOPICS:
            schema_id = writer.register_schema(
                name=name, encoding="ros2msg", data=text.encode())
            channels.append(writer.register_channel(
                topic=topic, message_encoding="cdr", schema_id=schema_id))

        for index in range(MESSAGE_COUNT):
            log_time = START_TIME_NS + index * PERIOD_NS
            messages = [(channels[0], status_payload(index))]
            if 0 == index % IMU_PERIOD:
                messages.append((channels[1], imu_payload(index)))
            for channel_id, payload in messages:
                writer.add_message(
                    channel_id=channel_id, log_time=log_time,
                    publish_time=log_time, sequence=index, data=payload)

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


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
