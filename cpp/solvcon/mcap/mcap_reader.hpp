#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Container-level reader for MCAP recordings.
 *
 * @ingroup group_inout
 */

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <solvcon/buffer/small_vector.hpp>

namespace solvcon
{

namespace mcap
{

/**
 * Schema record of the summary section.  The data field holds the schema text
 * verbatim, for example the message definition of a ROS 2 type.
 *
 * @ingroup group_inout
 */
struct SchemaRecord
{
    uint16_t id = 0;
    std::string name;
    std::string encoding;
    std::string data;
}; /* end struct SchemaRecord */

/**
 * Channel record of the summary section.  A channel carries the messages of
 * one topic and names the schema that describes them.
 *
 * @ingroup group_inout
 */
struct ChannelRecord
{
    uint16_t id = 0;
    uint16_t schema_id = 0;
    std::string topic;
    std::string message_encoding;
    std::string metadata; ///< Key-value pairs, as the record serializes them.
}; /* end struct ChannelRecord */

/**
 * Chunk index record of the summary section.  The channel ids name the
 * channels the chunk holds messages of.  A query for a topic skips the chunk
 * when the list omits its channel.
 *
 * @ingroup group_inout
 */
struct ChunkIndexRecord
{
    uint64_t message_start_time = 0; ///< Log time of the first message, in nanoseconds.
    uint64_t message_end_time = 0; ///< Log time of the last message, in nanoseconds.
    uint64_t chunk_start_offset = 0; ///< Byte offset of the chunk record in the file.
    uint64_t chunk_length = 0; ///< Byte length of the chunk record.
    small_vector<uint16_t> channel_ids;
}; /* end struct ChunkIndexRecord */

/**
 * Reader for the container layer of an MCAP recording.
 *
 * Construction validates the file magic and the footer, then parses the
 * summary section only: the schema, channel, chunk index, and statistics
 * records.  No chunk is read, and the message payloads stay where they are.
 *
 * The file must carry a summary section.  Every writer that indexes its output
 * writes one.  Without it, a reader would have to walk the whole recording to
 * collect the topics.
 *
 * @ingroup group_inout
 */
class Reader
{

public:

    explicit Reader(std::string const & path);

    ~Reader() = default;

    Reader(Reader const & other) = delete;
    Reader(Reader && other) = delete;
    Reader & operator=(Reader const & other) = delete;
    Reader & operator=(Reader && other) = delete;

    std::string const & path() const { return m_path; }

    /**
     * Topic name to the name of the schema its messages follow.  A topic
     * carried by more than one channel appears once.  The channel of highest
     * id names it, and schema() answers with that same channel.
     */
    std::map<std::string, std::string> topics() const;

    /**
     * Log time of the first and of the last message, in nanoseconds.  A
     * summary that describes no message at all leaves the range at (0, 0);
     * has_time_range() tells that apart from a recording that starts at zero.
     */
    std::pair<uint64_t, uint64_t> time_range() const { return {m_start_time, m_end_time}; }

    /// Whether the summary stated the log time range.
    bool has_time_range() const { return m_has_time_range; }

    /// Number of chunk index records the summary carries.
    size_t chunk_count() const { return m_chunk_indices.size(); }

    /**
     * Schema the messages of a topic follow, as the summary states it.
     *
     * TODO: The decode plan is the first caller of this, and what it needs
     * from a schema decides the shape returned here. Revisit once it lands
     * (issue #1286).
     */
    SchemaRecord schema(std::string const & topic) const;

private:

    void read_summary();
    /// Walk the summary section one record at a time, from start to end.
    void parse_summary(uint64_t start, uint64_t end);
    void parse_summary_record(uint8_t opcode, std::string_view content);
    std::string read_bytes(uint64_t offset, uint64_t length);
    /// The channel that names a topic, or nullptr when no channel carries it.
    ChannelRecord const * channel_for(std::string const & topic) const;

    std::string m_path;
    std::ifstream m_stream;
    uint64_t m_file_size = 0;
    // TODO: Replace the STL containers below once the buffer subsystem offers
    // an id-keyed map and a collector of records; SimpleCollector holds
    // fundamental types and these hold records (issue #1286).
    std::map<uint16_t, SchemaRecord> m_schemas;
    std::map<uint16_t, ChannelRecord> m_channels;
    std::vector<ChunkIndexRecord> m_chunk_indices;
    bool m_has_time_range = false;
    uint64_t m_start_time = 0;
    uint64_t m_end_time = 0;
}; /* end class Reader */

} /* end namespace mcap */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
