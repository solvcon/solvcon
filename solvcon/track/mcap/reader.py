# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Container layer: the MCAP file format.

``Reader`` reads the magic, the header, the footer, and the summary section
the footer points to.  The summary gives the schemas, the channels, the
statistics, and the chunk indexes; the chunks themselves are read on demand
by ``messages()`` and the ``extract*`` methods.
"""

import struct
import collections

import numpy as np

from .. import dataframe
from . import McapError
from .idl import DecodePlan, COLUMN_TYPES

MAGIC = b"\x89MCAP0\r\n"
OP_HEADER = 0x01
OP_FOOTER = 0x02
OP_SCHEMA = 0x03
OP_CHANNEL = 0x04
OP_MESSAGE = 0x05
OP_CHUNK = 0x06
OP_CHUNK_INDEX = 0x08
OP_STATISTICS = 0x0B
FOOTER_SIZE = 1 + 8 + 20

Schema = collections.namedtuple("Schema", "id name encoding data")
Channel = collections.namedtuple("Channel",
                                 "id schema_id topic message_encoding")
ChunkIndex = collections.namedtuple("ChunkIndex",
                                    "start_ns end_ns offset length")


class Extraction(collections.namedtuple("Extraction", "time columns")):
    """
    The columns of one topic sorted by log time.

    ``time`` is a ``SimpleArrayUint64`` of log times in nanoseconds, and
    ``columns`` maps each selected field to its ``SimpleArray``.
    """

    def to_frame(self):
        return dataframe.DataFrame.from_columns(self.time, **self.columns)


def unpack(fmt, buf, pos):
    """Return the values of ``fmt`` at ``pos`` and the position after."""
    return struct.unpack_from(fmt, buf, pos), pos + struct.calcsize(fmt)


def unpack_prefixed(fmt, buf, pos):
    """Return the bytes whose length ``fmt`` gives and the position after."""
    (size,), pos = unpack(fmt, buf, pos)
    return buf[pos:pos + size], pos + size


def unpack_string(buf, pos):
    data, pos = unpack_prefixed("<I", buf, pos)
    return data.decode(), pos


def iter_records(buf):
    """Yield ``(opcode, body)`` for every record in ``buf``."""
    pos = 0
    while pos + 9 <= len(buf):
        (opcode, size), pos = unpack("<BQ", buf, pos)
        yield opcode, buf[pos:pos + size]
        pos += size


def decompress(compression, data, uncompressed_size):
    if compression == "":
        return data
    if compression == "zstd":
        import zstandard
        return zstandard.ZstdDecompressor().decompress(
            data, max_output_size=uncompressed_size)
    if compression == "lz4":
        import lz4.frame
        return lz4.frame.decompress(data)
    raise McapError("unknown chunk compression {!r}".format(compression))


class Reader:
    """
    Read one MCAP file.

    The reader is a context manager; ``close()`` releases the file handle.
    """

    def __init__(self, path):
        self._file = open(path, "rb")
        self._schemas = {}
        self._channels = {}
        self._chunk_indexes = []
        self._statistics = None
        self._read_summary()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._file.close()

    def _read_at(self, offset, size):
        self._file.seek(offset)
        return self._file.read(size)

    def _read_summary(self):
        self._file.seek(0, 2)
        file_size = self._file.tell()
        if self._read_at(0, 8) != MAGIC or \
                self._read_at(file_size - 8, 8) != MAGIC:
            raise McapError("bad magic")

        (opcode, size), _ = unpack("<BQ", self._read_at(8, 9), 0)
        if opcode != OP_HEADER:
            raise McapError("the header record is missing")
        self.profile, _ = unpack_string(self._read_at(17, size), 0)

        footer_offset = file_size - 8 - FOOTER_SIZE
        footer = self._read_at(footer_offset, FOOTER_SIZE)
        (opcode, _, summary_start, _, _), _ = unpack("<BQQQI", footer, 0)
        if opcode != OP_FOOTER or summary_start == 0:
            raise McapError("the footer or the summary section is missing")

        summary = self._read_at(summary_start, footer_offset - summary_start)
        for opcode, body in iter_records(summary):
            if opcode == OP_SCHEMA:
                (schema_id,), pos = unpack("<H", body, 0)
                name, pos = unpack_string(body, pos)
                encoding, pos = unpack_string(body, pos)
                data, pos = unpack_prefixed("<I", body, pos)
                self._schemas[schema_id] = Schema(schema_id, name, encoding,
                                                  bytes(data))
            elif opcode == OP_CHANNEL:
                (channel_id, schema_id), pos = unpack("<HH", body, 0)
                topic, pos = unpack_string(body, pos)
                message_encoding, pos = unpack_string(body, pos)
                self._channels[channel_id] = Channel(
                    channel_id, schema_id, topic, message_encoding)
            elif opcode == OP_CHUNK_INDEX:
                fields, _ = unpack("<QQQQ", body, 0)
                self._chunk_indexes.append(ChunkIndex(*fields))
            elif opcode == OP_STATISTICS:
                (start_ns, end_ns), _ = unpack("<QQ", body, 8 + 2 + 4 * 4)
                self._statistics = (start_ns, end_ns)

    def topics(self):
        """Return a map from topic to schema name."""
        return {ch.topic: self._schemas[ch.schema_id].name
                for ch in self._channels.values()}

    def channel(self, topic):
        found = [ch for ch in self._channels.values() if ch.topic == topic]
        if len(found) != 1:
            raise McapError("topic {!r} matches {} channels".format(
                topic, len(found)))
        return found[0]

    def schema(self, topic):
        return self._schemas[self.channel(topic).schema_id]

    def time_range(self):
        """Return ``(start_ns, end_ns)``, or ``None`` without statistics."""
        return self._statistics

    def _iter_messages(self, channel_ids, start_ns, end_ns):
        """
        Yield ``(channel_id, log_time, payload)`` in file order.

        The chunk indexes skip the chunks outside ``[start_ns, end_ns)``
        before any decompression.
        """
        for index in self._chunk_indexes:
            if start_ns is not None and index.end_ns < start_ns:
                continue
            if end_ns is not None and index.start_ns >= end_ns:
                continue
            chunk = self._read_at(index.offset, index.length)
            (opcode, _), pos = unpack("<BQ", chunk, 0)
            if opcode != OP_CHUNK:
                raise McapError("the chunk index is off a chunk")
            (_, _, uncompressed_size, _), pos = unpack("<QQQI", chunk, pos)
            compression, pos = unpack_string(chunk, pos)
            data, pos = unpack_prefixed("<Q", chunk, pos)

            records = decompress(compression, data, uncompressed_size)
            for opcode, body in iter_records(records):
                if opcode != OP_MESSAGE:
                    continue
                (channel_id, _, log_time, _), pos = unpack("<HIQQ", body, 0)
                if channel_id not in channel_ids:
                    continue
                if start_ns is not None and log_time < start_ns:
                    continue
                if end_ns is not None and log_time >= end_ns:
                    continue
                yield channel_id, log_time, bytes(body[pos:])

    def messages(self, topic, start_ns=None, end_ns=None):
        """
        Yield ``(log_time, payload)`` of ``topic`` in file order.

        The range ``[start_ns, end_ns)`` is half open, the same convention
        as ``DataFrame.window``.
        """
        channel_id = self.channel(topic).id
        for _, log_time, payload in self._iter_messages({channel_id},
                                                        start_ns, end_ns):
            yield log_time, payload

    def extract_many(self, specs, start_ns=None, end_ns=None):
        """
        Extract the columns of several topics in one pass over the file.

        :param specs: a map from topic to a field list or a ``DecodePlan``.
        :return: a map from topic to ``Extraction``.
        """
        plans = {}
        for topic, fields in specs.items():
            channel = self.channel(topic)
            schema = self._schemas[channel.schema_id]
            plan = fields if isinstance(fields, DecodePlan) else \
                DecodePlan(schema, fields)
            if channel.message_encoding != "cdr" or plan.schema != schema:
                raise McapError("cannot decode {!r} with the plan".format(
                    topic))
            plans[channel.id] = (topic, plan)

        rows = {channel_id: [] for channel_id in plans}
        for channel_id, log_time, payload in self._iter_messages(
                set(plans), start_ns, end_ns):
            rows[channel_id].append(
                (log_time,) + plans[channel_id][1].decode(payload))

        extractions = {}
        for channel_id, (topic, plan) in plans.items():
            rows[channel_id].sort(key=lambda row: row[0])
            columns = list(zip(*rows[channel_id])) or \
                [()] * (len(plan.fields) + 1)
            extractions[topic] = Extraction(
                _column(columns[0], "uint64"),
                {field: _column(values, dtype) for field, dtype, values
                 in zip(plan.fields, plan.types, columns[1:])})
        return extractions

    def extract(self, topic, fields=None, start_ns=None, end_ns=None):
        return self.extract_many({topic: fields}, start_ns, end_ns)[topic]

    def extract_frame_many(self, specs, start_ns=None, end_ns=None):
        extractions = self.extract_many(specs, start_ns, end_ns)
        return {topic: ext.to_frame() for topic, ext in extractions.items()}

    def extract_frame(self, topic, fields=None, start_ns=None, end_ns=None):
        return self.extract(topic, fields, start_ns, end_ns).to_frame()


def _column(values, dtype):
    return COLUMN_TYPES[dtype][1](array=np.array(values, dtype=dtype))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
