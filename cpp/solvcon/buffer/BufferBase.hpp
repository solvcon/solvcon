#pragma once

/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/base.hpp>

namespace solvcon
{

/// Validate that alignment is one of the supported values (0, 16, 32, 64).
inline std::size_t validate_alignment(std::size_t alignment, const char * prefix = nullptr)
{
    if (alignment != 0 && alignment != 16 && alignment != 32 && alignment != 64)
    {
        const std::string prefix_str = prefix ? std::string(prefix) + ": " : "";
        throw std::invalid_argument(
            std::format("{}alignment must be 0, 16, 32, or 64, but got {}",
                        prefix_str,
                        alignment));
    }
    return alignment;
}

/// Validate that size is a multiple of alignment (when alignment > 0).
inline void validate_size_alignment(std::size_t size, std::size_t alignment, const char * prefix = nullptr)
{
    if (alignment > 0 && size % alignment != 0)
    {
        const std::string prefix_str = prefix ? std::string(prefix) + ": " : "";
        throw std::invalid_argument(
            std::format("{}size {} must be a multiple of alignment {}",
                        prefix_str,
                        size,
                        alignment));
    }
}

/// Base class for buffer-like objects.
template <typename Derived>
class BufferBase
{
public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    explicit operator bool() const { return static_cast<bool>(m_begin); }
    size_type size() const
    {
        return this->m_begin == this->m_end ? 0 : static_cast<size_type>(this->m_end - this->m_begin);
    }
    size_t nbytes() const { return size() * sizeof(int8_t); }

    int8_t operator[](size_t it) const { return data(it); }
    int8_t & operator[](size_t it) { return data(it); }

    int8_t at(size_t it) const
    {
        validate_range(it);
        return data(it);
    }

    int8_t & at(size_t it)
    {
        validate_range(it);
        return data(it);
    }

    using iterator = int8_t *;
    using const_iterator = int8_t const *;

    iterator begin() noexcept { return m_begin; }
    iterator end() noexcept { return m_end; }
    const_iterator begin() const noexcept { return m_begin; }
    const_iterator end() const noexcept { return m_end; }
    const_iterator cbegin() const noexcept { return m_begin; }
    const_iterator cend() const noexcept { return m_end; }

    /* Backdoor */
    int8_t data(size_type it) const { return data()[it]; }
    int8_t & data(size_type it) { return data()[it]; }
    int8_t const * data() const noexcept { return data<int8_t>(); }
    int8_t * data() noexcept { return data<int8_t>(); }

    // clang-format off
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    template <typename T> T const * data() const noexcept { return reinterpret_cast<T *>(m_begin); }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    template <typename T> T * data() noexcept { return reinterpret_cast<T *>(m_begin); }
    // clang-format on

    constexpr const char * name() const
    {
        return Derived::name();
    }

protected:
    BufferBase() = default;

    BufferBase(int8_t * start, int8_t * end)
        : m_begin(start)
        , m_end(end)
    {
    }

public:
    BufferBase(BufferBase const &) = delete;
    BufferBase(BufferBase &&) = delete;
    BufferBase & operator=(BufferBase const &) = delete;
    BufferBase & operator=(BufferBase &&) = delete;
    ~BufferBase() = default;

protected:
    void validate_range(size_t it) const
    {
        if (it >= size())
        {
            throw std::out_of_range(std::format("{}: index {} is out of bounds with size {}", name(), it, size()));
        }
    }

    int8_t * m_begin; // don't initialize, must be set by derived class
    int8_t * m_end; // don't initialize, must be set by derived class

}; /* end class BufferBase */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
