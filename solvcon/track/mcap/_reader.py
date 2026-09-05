# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Container layer: the MCAP file format.

``Reader`` reads the magic, the header, the footer, and the summary section
the footer points to.  The summary gives the schemas, the channels, the
statistics, and the chunk indexes; the chunks themselves are read on demand
by ``messages()`` and the ``extract*`` methods.  A file without a summary
section gives no statistics, and ``Reader`` builds the index by walking the
data section at open time instead.
"""

import os
import struct
import collections

import numpy as np

from .. import dataframe
from . import McapError
from ._decodeplan import DecodePlan, COLUMN_TYPES

__all__ = ["Reader", "Schema", "Channel", "Extraction"]

MAGIC = b"\x89MCAP0\r\n"
MAGIC_SIZE = len(MAGIC)
OP_HEADER = 0x01
OP_FOOTER = 0x02
OP_SCHEMA = 0x03
OP_CHANNEL = 0x04
OP_MESSAGE = 0x05
OP_CHUNK = 0x06
OP_MESSAGE_INDEX = 0x07
OP_CHUNK_INDEX = 0x08
OP_STATISTICS = 0x0B
RECORD_HEADER_SIZE = 1 + 8
FOOTER_SIZE = RECORD_HEADER_SIZE + 20

Schema = collections.namedtuple("Schema", "id name encoding data")
Channel = collections.namedtuple("Channel",
                                 "id schema_id topic message_encoding")
ChunkIndex = collections.namedtuple("ChunkIndex",
                                    "start_ns end_ns offset length chunked")


class Extraction(collections.namedtuple("Extraction", "time columns")):
    """
    The columns of one topic sorted by log time.

    ``time`` is a ``SimpleArrayUint64`` of log times in nanoseconds, and
    ``columns`` maps each selected field to its ``SimpleArray``.  A string
    or container field maps to a NumPy ``object`` array of Python values
    instead, which ``to_frame`` rejects.
    """

    def to_frame(self):
        try:
            return dataframe.DataFrame.from_columns(self.time, **self.columns)
        except TypeError as error:
            raise McapError("not a frame column: {}".format(error)) from error


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
    while pos + RECORD_HEADER_SIZE <= len(buf):
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

    ``path`` is the file the reader opened and ``size`` its size in
    bytes. The reader is a context manager; ``close()`` releases the
    file handle.
    """

    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        self._file = open(path, "rb")
        self._schemas = {}
        self._channels = {}
        self._chunk_indexes = []
        self._statistics = None
        self._message_count = None
        self._message_counts = None
        try:
            self._read_index()
        except (struct.error, UnicodeDecodeError) as error:
            raise McapError("malformed record: {}".format(error)) from error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._file.close()

    def _read_at(self, offset, size):
        self._file.seek(offset)
        return self._file.read(size)

    def _read_index(self):
        """Index the file from its summary section, or from its data."""
        if self._read_at(0, MAGIC_SIZE) != MAGIC:
            raise McapError("bad magic")

        (opcode, size), _ = unpack("<BQ", self._read_at(
            MAGIC_SIZE, RECORD_HEADER_SIZE), 0)
        if opcode != OP_HEADER:
            raise McapError("the header record is missing")
        data_start = MAGIC_SIZE + RECORD_HEADER_SIZE + size
        if data_start > self.size:
            raise McapError("the header record is truncated")
        self.profile, _ = unpack_string(self._read_at(
            MAGIC_SIZE + RECORD_HEADER_SIZE, size), 0)

        summary_start = self._summary_start(data_start)
        if summary_start:
            self._read_summary(summary_start)
        else:
            self._scan_data(data_start)

    def _summary_start(self, data_start):
        """Return the offset of the summary section, or 0 without one."""
        footer_offset = self.size - MAGIC_SIZE - FOOTER_SIZE
        if footer_offset < data_start or \
                self._read_at(self.size - MAGIC_SIZE, MAGIC_SIZE) != MAGIC:
            return 0
        (opcode, _, summary_start, _, _), _ = unpack(
            "<BQQQI", self._read_at(footer_offset, FOOTER_SIZE), 0)
        if opcode != OP_FOOTER:
            raise McapError("the footer record is missing")
        return summary_start

    def _read_summary(self, summary_start):
        """Take the schemas, the channels, and the indexes from the summary."""
        end = self.size - MAGIC_SIZE - FOOTER_SIZE
        for opcode, body in iter_records(
                self._read_at(summary_start, end - summary_start)):
            if opcode == OP_CHUNK_INDEX:
                fields, _ = unpack("<QQQQ", body, 0)
                self._chunk_indexes.append(ChunkIndex(*fields, True))
            elif opcode == OP_STATISTICS:
                (self._message_count,), pos = unpack("<Q", body, 0)
                (start_ns, end_ns), pos = unpack("<QQ", body, pos + 2 + 4 * 4)
                self._statistics = (start_ns, end_ns)
                counts, _ = unpack_prefixed("<I", body, pos)
                self._message_counts = dict(struct.iter_unpack("<HQ", counts))
            else:
                self._register(opcode, body)

    def _register(self, opcode, body):
        """Store the schema or the channel that ``body`` carries."""
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

    def _scan_data(self, offset):
        """
        Index a file that has no summary section.

        MCAP leaves the summary section optional, and a recorder that
        dies leaves the file without one.  The reader walks the data
        section instead and drops a record that the end of the file cuts
        short.  A run of messages outside any chunk becomes one span, so
        an unchunked file reads like a chunked one.
        """
        wanted = set()
        span = None
        while offset + RECORD_HEADER_SIZE <= self.size:
            (opcode, size), _ = unpack("<BQ", self._read_at(
                offset, RECORD_HEADER_SIZE), 0)
            length = RECORD_HEADER_SIZE + size
            if opcode == OP_FOOTER or offset + length > self.size:
                break

            if opcode == OP_MESSAGE:
                span = self._extend_span(span, offset, length)
            else:
                span = self._close_span(span)
                if opcode == OP_CHUNK:
                    (start_ns, end_ns), _ = unpack("<QQ", self._read_at(
                        offset + RECORD_HEADER_SIZE, 16), 0)
                    self._chunk_indexes.append(ChunkIndex(
                        start_ns, end_ns, offset, length, True))
                elif opcode == OP_MESSAGE_INDEX:
                    (channel_id,), _ = unpack("<H", self._read_at(
                        offset + RECORD_HEADER_SIZE, 2), 0)
                    wanted.add(channel_id)
                elif opcode in (OP_SCHEMA, OP_CHANNEL):
                    self._register(opcode, self._read_at(
                        offset + RECORD_HEADER_SIZE, size))
            offset += length
        self._close_span(span)

        # The schemas and the channels usually sit inside the chunks, and
        # the message indexes name every channel that carries a message, so
        # the walk stops once the chunks account for all of them.
        for index in self._chunk_indexes:
            if self._channels_known(wanted):
                break
            if index.chunked:
                for opcode, body in iter_records(self._chunk_records(
                        index.offset, index.length)):
                    self._register(opcode, body)

    def _extend_span(self, span, offset, length):
        """Return ``span`` grown by the message record at ``offset``."""
        (_, _, log_time, _), _ = unpack("<HIQQ", self._read_at(
            offset + RECORD_HEADER_SIZE, 22), 0)
        if span is None:
            return [log_time, log_time, offset, length]
        span[0] = min(span[0], log_time)
        span[1] = max(span[1], log_time)
        span[3] += length
        return span

    def _close_span(self, span):
        """Index ``span`` as a run of messages outside any chunk."""
        if span is not None:
            self._chunk_indexes.append(ChunkIndex(*span, False))
        return None

    def _channels_known(self, wanted):
        """Tell whether every channel in ``wanted`` has its records."""
        if not wanted or not wanted.issubset(self._channels):
            return False
        return all(ch.schema_id == 0 or ch.schema_id in self._schemas
                   for ch in self._channels.values())

    def _chunk_records(self, offset, length):
        """Return the decompressed body of the chunk at ``offset``."""
        chunk = self._read_at(offset, length)
        (opcode, _), pos = unpack("<BQ", chunk, 0)
        if opcode != OP_CHUNK:
            raise McapError("the chunk index is off a chunk")
        (_, _, uncompressed_size, _), pos = unpack("<QQQI", chunk, pos)
        compression, pos = unpack_string(chunk, pos)
        data, pos = unpack_prefixed("<Q", chunk, pos)
        return decompress(compression, data, uncompressed_size)

    def topics(self):
        """Return a map from topic to schema name; "" means no schema."""
        names = {}
        for ch in self._channels.values():
            schema = self._schemas.get(ch.schema_id)
            names[ch.topic] = schema.name if schema else ""
        return names

    def channels(self):
        """Return every channel in file order."""
        return list(self._channels.values())

    def schema_of(self, channel):
        """Return the schema of ``channel``, or ``None`` when it has none."""
        return self._schemas.get(channel.schema_id)

    def channel(self, topic):
        found = [ch for ch in self._channels.values() if ch.topic == topic]
        if len(found) != 1:
            raise McapError("topic {!r} matches {} channels".format(
                topic, len(found)))
        return found[0]

    def schema(self, topic):
        """Return the schema of ``topic``, or ``None`` when it has none."""
        return self._schemas.get(self.channel(topic).schema_id)

    def time_range(self):
        """Return ``(start_ns, end_ns)``, or ``None`` without statistics."""
        return self._statistics

    def message_count(self):
        """Return the file's message count, or ``None`` without statistics."""
        return self._message_count

    def message_counts(self):
        """Return a map from topic to its message count.

        The counts come from the statistics record, which is optional; a
        file without one gives ``None``. Channels that share a topic add
        up.
        """
        if self._message_counts is None:
            return None
        counts = {}
        for ch in self._channels.values():
            counts[ch.topic] = counts.get(ch.topic, 0) + \
                self._message_counts.get(ch.id, 0)
        return counts

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
            if index.chunked:
                records = self._chunk_records(index.offset, index.length)
            else:
                records = self._read_at(index.offset, index.length)
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
            A plan built from another ``Schema`` works only when it
            names the message type the file gives the topic.
        :return: a map from topic to ``Extraction``.
        """
        plans = {}
        for topic, fields in specs.items():
            channel = self.channel(topic)
            schema = self._schemas.get(channel.schema_id)
            if isinstance(fields, DecodePlan):
                plan = fields
            elif schema is None:
                raise McapError("topic {!r} carries no schema".format(topic))
            else:
                plan = DecodePlan(schema, fields)
            if channel.message_encoding != "cdr" or \
                    (schema is not None and plan.schema.name != schema.name):
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
    if dtype not in COLUMN_TYPES:
        return np.fromiter(values, dtype='object', count=len(values))
    return COLUMN_TYPES[dtype][1](array=np.array(values, dtype=dtype))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
