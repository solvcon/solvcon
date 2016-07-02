#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <stdexcept>
#include <memory>

namespace march
{

namespace mesh { class LookupTableCore; }

/**
 * Untyped and unresizeable memory buffer for data storage.
 */
class Buffer: public std::enable_shared_from_this<Buffer> {

private:

    size_t m_length = 0;
    char * m_data = nullptr;

    struct ctor_passkey {};

public:

    Buffer(const ctor_passkey &) { }

    static std::shared_ptr<Buffer> construct() {
        return std::make_shared<Buffer>(ctor_passkey());
    }

    /**
     * \param[in] length Memory buffer length.
     */
    Buffer(size_t length, const ctor_passkey &) : m_length(length) { m_data = new char[length](); }

    static std::shared_ptr<Buffer> construct(size_t length) {
        return std::make_shared<Buffer>(length, ctor_passkey());
    }

    ~Buffer() {
        delete[] m_data;
        m_data = nullptr;
    }

    Buffer(const Buffer &) = delete;

    Buffer(Buffer &&) = delete;

    Buffer & operator=(const Buffer &) = delete;

    Buffer & operator=(Buffer &&) = delete;
    /*Buffer & operator=(Buffer && other) {
        m_data = other.m_data;
        m_data = nullptr;
        m_length = other.m_length;
        m_own_data = other.m_own_data;
        return *this;
    }*/

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
