/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/mcap/mcap_reader.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

#include <lz4frame.h>
#include <zstd.h>

namespace solvcon
{

namespace mcap
{

namespace
{

constexpr std::array<char, 8> MAGIC = {'\x89', 'M', 'C', 'A', 'P', '0', '\r', '\n'};
constexpr uint64_t MAGIC_SIZE = MAGIC.size();
// One opcode byte and one uint64 content length.
constexpr uint64_t RECORD_HEADER_SIZE = 9;
// Two uint64 offsets and one uint32 CRC.
constexpr uint64_t FOOTER_CONTENT_SIZE = 20;

enum Opcode : uint8_t
{
    OP_FOOTER = 0x02,
    OP_SCHEMA = 0x03,
    OP_CHANNEL = 0x04,
    OP_MESSAGE = 0x05,
    OP_CHUNK = 0x06,
    OP_CHUNK_INDEX = 0x08,
    OP_STATISTICS = 0x0b,
}; /* end enum Opcode */

/**
 * @internal
 * Cursor reading the fields of one record content.  MCAP writes its integers
 * little-endian, and so does every platform solvcon builds for, so a field is
 * a plain copy out of the buffer.
 */
class FieldReader
{

public:

    explicit FieldReader(std::string_view data)
        : m_data(data)
    {
    }

    uint16_t u16() { return read<uint16_t>(); }
    uint32_t u32() { return read<uint32_t>(); }
    uint64_t u64() { return read<uint64_t>(); }

    /// Length-prefixed string or byte array.
    std::string_view str() { return bytes(u32()); }

    std::string_view bytes(uint64_t length)
    {
        require(length);
        std::string_view const out = m_data.substr(m_pos, length);
        m_pos += length;
        return out;
    }

    void skip(uint64_t length) { bytes(length); }

    /// The bytes after the cursor.
    std::string_view rest() { return bytes(m_data.size() - m_pos); }

    bool done() const { return m_pos >= m_data.size(); }

private:

    template <typename T>
    T read()
    {
        require(sizeof(T));
        T value = 0;
        std::memcpy(&value, m_data.data() + m_pos, sizeof(T));
        m_pos += sizeof(T);
        return value;
    }

    void require(uint64_t length) const
    {
        // The cursor never passes the end, so the room left is a subtraction.
        // Adding the length to the position instead would wrap for a length
        // the file states.
        if (length > m_data.size() - m_pos)
        {
            throw std::runtime_error("MCAP record is truncated");
        }
    }

    std::string_view m_data;
    uint64_t m_pos = 0;
}; /* end class FieldReader */

} /* end namespace */

static bool same_schema(SchemaRecord const & lhs, SchemaRecord const & rhs)
{
    return lhs.name == rhs.name && lhs.encoding == rhs.encoding && lhs.data == rhs.data;
}

static bool same_channel(ChannelRecord const & lhs, ChannelRecord const & rhs)
{
    return lhs.schema_id == rhs.schema_id && lhs.topic == rhs.topic &&
           lhs.message_encoding == rhs.message_encoding && lhs.metadata == rhs.metadata;
}

/// Whether the summary record of this opcode is one the reader keeps.
static bool is_kept_record(uint8_t opcode)
{
    return OP_SCHEMA == opcode || OP_CHANNEL == opcode ||
           OP_CHUNK_INDEX == opcode || OP_STATISTICS == opcode;
}

/**
 * @internal
 * Read the record at the cursor of an in-memory section and advance past it.
 * False means the section holds no further record.
 */
static bool next_record(std::string_view buffer, size_t & pos, uint8_t & opcode, std::string_view & content)
{
    if (pos == buffer.size())
    {
        return false;
    }
    // A remainder shorter than a header is a truncated record, not the end
    // of the section.
    if (buffer.size() - pos < RECORD_HEADER_SIZE)
    {
        throw std::runtime_error("MCAP section ends inside a record header");
    }

    opcode = static_cast<uint8_t>(buffer[pos]);
    uint64_t length = 0;
    std::memcpy(&length, buffer.data() + pos + 1, sizeof(length));
    if (buffer.size() - pos - RECORD_HEADER_SIZE < length)
    {
        throw std::runtime_error("MCAP record runs past the end of its section");
    }

    content = buffer.substr(pos + RECORD_HEADER_SIZE, length);
    pos += RECORD_HEADER_SIZE + length;
    return true;
}

static std::string decompress(std::string_view compression, std::string_view input, uint64_t output_size)
{
    if (compression.empty())
    {
        return std::string(input);
    }

    std::string output(output_size, '\0');
    if ("lz4" == compression)
    {
        // MCAP compresses a chunk as an LZ4 frame, not as a bare LZ4 block.
        LZ4F_dctx * dctx = nullptr;
        if (LZ4F_isError(LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION)))
        {
            throw std::runtime_error("cannot create the LZ4 decompression context");
        }
        size_t dst_size = output.size();
        size_t src_size = input.size();
        size_t const ret = LZ4F_decompress(dctx, output.data(), &dst_size, input.data(), &src_size, nullptr);
        LZ4F_freeDecompressionContext(dctx);
        if (0 != ret || dst_size != output_size || src_size != input.size())
        {
            throw std::runtime_error("cannot decompress the lz4 MCAP chunk");
        }
    }
    else if ("zstd" == compression)
    {
        size_t const ret = ZSTD_decompress(output.data(), output.size(), input.data(), input.size());
        if (ZSTD_isError(ret) || ret != output_size)
        {
            throw std::runtime_error("cannot decompress the zstd MCAP chunk");
        }
    }
    else
    {
        throw std::runtime_error("unsupported MCAP chunk compression: " + std::string(compression));
    }

    return output;
}

Reader::Reader(std::string const & path)
    : m_path(path)
    , m_stream(path, std::ios::binary)
{
    if (!m_stream)
    {
        throw std::runtime_error("cannot open the MCAP file: " + path);
    }

    // A directory opens but does not seek, and an unchecked -1 would become a
    // file size of every byte there is.
    m_stream.seekg(0, std::ios::end);
    std::streamoff const size = m_stream.tellg();
    if (size < 0)
    {
        throw std::runtime_error("cannot open the MCAP file: " + path);
    }
    m_file_size = static_cast<uint64_t>(size);

    read_summary();
}

std::string Reader::read_bytes(uint64_t offset, uint64_t length)
{
    // Both arguments come out of the file.  A sum of the two wraps back into
    // range for a large enough pair, so the check avoids the addition.
    if (offset > m_file_size || length > m_file_size - offset)
    {
        throw std::runtime_error("MCAP read runs past the end of the file: " + m_path);
    }

    std::string out(length, '\0');
    m_stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    m_stream.read(out.data(), static_cast<std::streamsize>(length));
    if (!m_stream)
    {
        throw std::runtime_error("cannot read the MCAP file: " + m_path);
    }

    return out;
}

void Reader::read_summary()
{
    constexpr uint64_t TAIL_SIZE = RECORD_HEADER_SIZE + FOOTER_CONTENT_SIZE + MAGIC_SIZE;
    if (m_file_size < MAGIC_SIZE + TAIL_SIZE)
    {
        throw std::runtime_error("not an MCAP file, it is too short: " + m_path);
    }

    std::string const head = read_bytes(0, MAGIC_SIZE);
    std::string const tail = read_bytes(m_file_size - TAIL_SIZE, TAIL_SIZE);
    if (0 != std::memcmp(head.data(), MAGIC.data(), MAGIC_SIZE) ||
        0 != std::memcmp(tail.data() + tail.size() - MAGIC_SIZE, MAGIC.data(), MAGIC_SIZE))
    {
        throw std::runtime_error("not an MCAP file, the magic does not match: " + m_path);
    }

    // The footer sits at a fixed distance from the end.  Its opcode and
    // length are therefore read in place, not by walking records to it.
    uint64_t footer_length = 0;
    std::memcpy(&footer_length, tail.data() + 1, sizeof(footer_length));
    if (OP_FOOTER != static_cast<uint8_t>(tail[0]) || FOOTER_CONTENT_SIZE != footer_length)
    {
        throw std::runtime_error("the MCAP footer record is malformed: " + m_path);
    }

    FieldReader footer(std::string_view(tail).substr(RECORD_HEADER_SIZE, FOOTER_CONTENT_SIZE));
    uint64_t const summary_start = footer.u64();
    if (0 == summary_start)
    {
        throw std::runtime_error("the MCAP file carries no summary section: " + m_path);
    }
    uint64_t const summary_end = m_file_size - TAIL_SIZE;
    if (summary_start >= summary_end)
    {
        throw std::runtime_error("the MCAP summary section starts at or past the footer: " + m_path);
    }

    m_data_end = summary_start;
    parse_summary(summary_start, summary_end);

    if (m_has_time_range || m_chunk_indices.empty())
    {
        return;
    }
    // No statistics record, so the chunk indexes bound the same range. A chunk
    // holding no message states (0, 0) and would drag the start down to zero.
    for (ChunkIndexRecord const & chunk : m_chunk_indices)
    {
        if (0 == chunk.message_start_time && 0 == chunk.message_end_time)
        {
            continue;
        }
        m_start_time = m_has_time_range
                           ? std::min(m_start_time, chunk.message_start_time)
                           : chunk.message_start_time;
        m_end_time = std::max(m_end_time, chunk.message_end_time);
        m_has_time_range = true;
    }
}

bool Reader::read_record_header(uint64_t pos, uint64_t end, uint8_t & opcode, uint64_t & length)
{
    if (pos == end)
    {
        return false;
    }
    // A length out of the file may not be added to the position, because the
    // sum wraps for a large enough pair.  Every check is a subtraction.
    if (end - pos < RECORD_HEADER_SIZE)
    {
        throw std::runtime_error("an MCAP section ends inside a record header: " + m_path);
    }

    std::string const head = read_bytes(pos, RECORD_HEADER_SIZE);
    opcode = static_cast<uint8_t>(head[0]);
    std::memcpy(&length, head.data() + 1, sizeof(length));
    if (length > end - pos - RECORD_HEADER_SIZE)
    {
        throw std::runtime_error("an MCAP record runs past its section: " + m_path);
    }

    return true;
}

void Reader::parse_summary(uint64_t start, uint64_t end)
{
    // One record is read at a time rather than the section as a whole.  What
    // the walk costs then follows the largest record, not the whole summary.
    uint64_t pos = start;
    uint8_t opcode = 0;
    uint64_t length = 0;
    while (read_record_header(pos, end, opcode, length))
    {
        if (is_kept_record(opcode))
        {
            parse_summary_record(opcode, read_bytes(pos + RECORD_HEADER_SIZE, length));
        }
        pos += RECORD_HEADER_SIZE + length;
    }
}

void Reader::parse_summary_record(uint8_t opcode, std::string_view content)
{
    // Reading a field is what validates it, because the cursor throws on a
    // record too short to hold it.  Fields the specification adds after the
    // ones below stay unread, which is what keeps this reader working against
    // a newer writer.
    FieldReader field(content);
    switch (opcode)
    {
    case OP_SCHEMA:
    {
        SchemaRecord schema;
        schema.id = field.u16();
        schema.name = std::string(field.str());
        schema.encoding = std::string(field.str());
        schema.data = std::string(field.str());
        // Zero is what a channel states for "no schema", so no schema record
        // may claim it.
        if (0 == schema.id)
        {
            break;
        }
        auto const it = m_schemas.find(schema.id);
        if (m_schemas.end() == it)
        {
            m_schemas[schema.id] = std::move(schema);
        }
        else if (!same_schema(it->second, schema))
        {
            throw std::runtime_error("the MCAP file states two schemas for one id: " + m_path);
        }
        break;
    }
    case OP_CHANNEL:
    {
        ChannelRecord channel;
        channel.id = field.u16();
        channel.schema_id = field.u16();
        channel.topic = std::string(field.str());
        channel.message_encoding = std::string(field.str());
        channel.metadata = std::string(field.str());
        auto const it = m_channels.find(channel.id);
        if (m_channels.end() == it)
        {
            m_channels[channel.id] = std::move(channel);
        }
        else if (!same_channel(it->second, channel))
        {
            throw std::runtime_error("the MCAP file states two channels for one id: " + m_path);
        }
        break;
    }
    case OP_CHUNK_INDEX:
    {
        ChunkIndexRecord chunk;
        chunk.message_start_time = field.u64();
        chunk.message_end_time = field.u64();
        chunk.chunk_start_offset = field.u64();
        chunk.chunk_length = field.u64();
        FieldReader offsets(field.str());
        while (!offsets.done())
        {
            chunk.channel_ids.push_back(offsets.u16());
            offsets.u64();
        }
        field.u64(); // message_index_length
        field.str(); // compression
        field.u64(); // compressed_size
        field.u64(); // uncompressed_size
        m_chunk_indices.push_back(std::move(chunk));
        break;
    }
    case OP_STATISTICS:
    {
        field.u64(); // message_count
        field.u16(); // schema_count
        field.u32(); // channel_count
        field.u32(); // attachment_count
        field.u32(); // metadata_count
        field.u32(); // chunk_count
        m_start_time = field.u64();
        m_end_time = field.u64();
        field.skip(field.u32()); // channel_message_counts
        m_has_time_range = true;
        break;
    }
    default:
        break;
    }
}

ChannelRecord const * Reader::channel_for(std::string const & topic) const
{
    // More than one channel may carry a topic.  The last of them names it, and
    // topics() and schema() have to agree on which that is.
    ChannelRecord const * found = nullptr;
    for (auto const & pair : m_channels)
    {
        if (topic == pair.second.topic)
        {
            found = &pair.second;
        }
    }

    return found;
}

SchemaRecord Reader::schema(std::string const & topic) const
{
    ChannelRecord const * const channel = channel_for(topic);
    if (nullptr == channel)
    {
        throw std::runtime_error("no such topic in the MCAP file: " + topic);
    }

    auto const it = m_schemas.find(channel->schema_id);
    if (m_schemas.end() == it)
    {
        throw std::runtime_error("the MCAP topic states no schema: " + topic);
    }

    return it->second;
}

std::map<std::string, std::string> Reader::topics() const
{
    std::map<std::string, std::string> out;
    for (auto const & pair : m_channels)
    {
        ChannelRecord const * const channel = channel_for(pair.second.topic);
        auto const it = m_schemas.find(channel->schema_id);
        out[channel->topic] = m_schemas.end() == it ? std::string() : it->second.name;
    }

    return out;
}

MessageIterator Reader::messages(std::string const & topic)
{
    small_vector<uint16_t> channel_ids;
    for (auto const & pair : m_channels)
    {
        if (topic == pair.second.topic)
        {
            channel_ids.push_back(pair.first);
        }
    }
    if (channel_ids.empty())
    {
        throw std::runtime_error("no such topic in the MCAP file: " + topic);
    }

    return MessageIterator(*this, channel_ids);
}

std::string Reader::read_chunk(uint64_t offset, uint64_t length)
{
    std::string const record = read_bytes(offset, length);
    size_t pos = 0;
    uint8_t opcode = 0;
    std::string_view content;
    if (!next_record(record, pos, opcode, content) || OP_CHUNK != opcode)
    {
        throw std::runtime_error("no MCAP chunk record where the chunk index points");
    }

    FieldReader field(content);
    field.u64(); // message_start_time
    field.u64(); // message_end_time
    uint64_t const uncompressed_size = field.u64();
    field.u32(); // uncompressed_crc
    std::string_view const compression = field.str();
    return decompress(compression, field.bytes(field.u64()), uncompressed_size);
}

MessageIterator::MessageIterator(Reader & reader, small_vector<uint16_t> const & channel_ids)
    : m_reader(&reader)
    , m_channel_ids(channel_ids)
    , m_offset(MAGIC_SIZE)
{
    for (size_t it = 0; it < reader.m_chunk_indices.size(); ++it)
    {
        small_vector<uint16_t> const & held = reader.m_chunk_indices[it].channel_ids;
        // The channel ids come from the map of message index offsets, and an
        // empty map means no message indexing is available.  It says nothing
        // about what the chunk holds, so the pruning must keep the chunk.
        bool const keep = held.empty() || std::any_of(held.begin(), held.end(), [this](uint16_t id)
                                                      { return wanted(id); });
        if (keep)
        {
            m_chunks.push_back(it);
        }
    }

    std::vector<ChunkIndexRecord> const & indices = reader.m_chunk_indices;
    std::sort(m_chunks.begin(), m_chunks.end(), [&indices](size_t lhs, size_t rhs)
              { return indices[lhs].chunk_start_offset < indices[rhs].chunk_start_offset; });
}

bool MessageIterator::wanted(uint16_t channel_id) const
{
    return m_channel_ids.end() != std::find(m_channel_ids.begin(), m_channel_ids.end(), channel_id);
}

bool MessageIterator::next(uint64_t & log_time, std::string_view & payload)
{
    while (true)
    {
        uint8_t opcode = 0;
        std::string_view content;
        if (!next_record(m_buffer, m_cursor, opcode, content))
        {
            if (!load_next_buffer())
            {
                return false;
            }
            continue;
        }

        if (OP_MESSAGE != opcode)
        {
            continue;
        }

        FieldReader field(content);
        if (!wanted(field.u16()))
        {
            continue;
        }
        field.u32(); // sequence
        log_time = field.u64();
        field.u64(); // publish_time
        payload = field.rest();
        return true;
    }
}

bool MessageIterator::load_next_buffer()
{
    m_buffer.clear();
    m_cursor = 0;

    if (m_next_chunk < m_chunks.size())
    {
        ChunkIndexRecord const & chunk = m_reader->m_chunk_indices[m_chunks[m_next_chunk++]];
        m_buffer = m_reader->read_chunk(chunk.chunk_start_offset, chunk.chunk_length);
        return true;
    }

    // A file carrying chunk indexes should hold every message in a chunk, but
    // the specification only recommends it, and a message record is legal in
    // the data section.  The walk therefore runs after the indexed chunks so
    // that a writer ignoring the recommendation cannot make messages vanish;
    // it reads a chunk itself only when no index named it.
    uint64_t const end = m_reader->m_data_end;
    uint8_t opcode = 0;
    uint64_t length = 0;
    while (m_reader->read_record_header(m_offset, end, opcode, length))
    {
        uint64_t const start = m_offset;
        m_offset += RECORD_HEADER_SIZE + length;

        if (OP_MESSAGE == opcode)
        {
            m_buffer = m_reader->read_bytes(start, RECORD_HEADER_SIZE + length);
            return true;
        }
        if (OP_CHUNK == opcode && m_reader->m_chunk_indices.empty())
        {
            m_buffer = m_reader->read_chunk(start, RECORD_HEADER_SIZE + length);
            return true;
        }
    }

    return false;
}

} /* end namespace mcap */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
