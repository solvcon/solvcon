#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <stdexcept>

namespace march
{

class Buffer {

public:

    Buffer() : m_data(nullptr) , m_length(0) {}

    Buffer(size_t length)
        : m_data(nullptr)
        , m_length(length)
    {
        m_data = new char[length];
    }

    Buffer(char * data, size_t length) : m_data(data), m_length(length) {}

    ~Buffer() {
        if (nullptr != m_data) { delete m_data; }
        m_data = nullptr;
    }

    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&) = delete;
    Buffer & operator=(const Buffer &) = delete;
    Buffer & operator=(Buffer && other) {
        m_data = other.m_data;
        other.m_data = nullptr;
        m_length = other.m_length;
        return *this;
    }

    explicit operator bool() const { return nullptr == m_data; }

    size_t bytes() const { return m_length; }

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

private:

    char * m_data;
    size_t m_length;

}; /* end class Buffer */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
