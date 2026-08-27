#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Bounds-checked cursor over the bytes of an MCAP record or message.
 *
 * @ingroup group_inout
 */

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace solvcon
{

namespace mcap
{

namespace detail
{

/**
 * @internal
 * Cursor over a run of bytes.  MCAP writes its integers little-endian, the
 * decoder accepts only little-endian CDR, and every platform solvcon builds
 * for is little-endian.  A value is therefore a plain copy out of the buffer.
 */
class ByteCursor
{

public:

    /**
     * Wrap a run of bytes.
     *
     * @param data Bytes to read.
     * @param truncated Message of the std::runtime_error that a read past
     *                  the end throws.  The string must outlive the cursor.
     */
    ByteCursor(std::string_view data, char const * truncated)
        : m_data(data)
        , m_truncated(truncated)
    {
    }

    size_t position() const { return m_pos; }

    void skip(size_t length)
    {
        require(length);
        m_pos += length;
    }

    template <typename T>
    T read()
    {
        require(sizeof(T));
        T value = T();
        std::memcpy(&value, m_data.data() + m_pos, sizeof(T));
        m_pos += sizeof(T);
        return value;
    }

protected:

    void require(size_t length) const
    {
        // The cursor never passes the end, so the room left is a subtraction.
        // Adding the length to the position instead would wrap for a length
        // the data states.
        if (length > m_data.size() - m_pos)
        {
            throw std::runtime_error(m_truncated);
        }
    }

    std::string_view m_data;
    size_t m_pos = 0;

private:

    char const * m_truncated;
}; /* end class ByteCursor */

} /* end namespace detail */

} /* end namespace mcap */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
