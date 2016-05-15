#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <stdexcept>

namespace march
{

/**
 * Untyped and unresizeable memory buffer for data storage.
 */
class Buffer {

private:

    size_t m_length = 0;
    char * m_data = nullptr;
    bool m_own_data = true;

public:

    Buffer() {
        static_assert(sizeof(Buffer) == 24, "Buffer size changes");
    }

    /**
     * \param[in] length Memory buffer length.
     */
    Buffer(size_t length) : m_length(length) { m_data = new char[length](); }

    /**
     * When given an allocated memory block (\p data) from outside, the
     * constructed Buffer object doesn't manage its own memory.
     *
     * This constructor allows the ownership of the memory block can be
     * transferred to an outside system, like NumPy.
     *
     * \param[in] length Memory buffer length.
     * \param[in] data   The memory block.
     */
    Buffer(size_t length, char * data) : m_length(length), m_data(data), m_own_data(false) {}

    ~Buffer() {
        if (m_own_data && nullptr != m_data) { delete m_data; }
        m_data = nullptr;
    }

    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&) = delete;
    Buffer & operator=(const Buffer &) = delete;
    Buffer & operator=(Buffer && other) {
        m_data = other.m_data;
        other.m_data = nullptr;
        m_length = other.m_length;
        m_own_data = other.m_own_data;
        return *this;
    }

    explicit operator bool() const { return nullptr == m_data; }

    size_t nbyte() const { return m_length; }

    template< typename T >
    size_t length() const {
        size_t result = m_length / sizeof(T);
        if (result * sizeof(T) != m_length) {
            throw std::length_error("length not divisible");
        }
        return result;
    }

    template< typename T, size_t LENGTH >
    T (& array() const) [LENGTH] {
        if (LENGTH * sizeof(T) != m_length) {
            throw std::length_error("array byte count mismatches buffer");
        }
        return *reinterpret_cast<T(*)[LENGTH]>(data<T>());
    }

    /** Backdoor */
    template< typename T >
    T * data() const { return reinterpret_cast<T*>(m_data); }

}; /* end class Buffer */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
