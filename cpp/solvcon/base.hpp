#pragma once

/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Core foundations shared across the library, e.g., the value-type bases, the
 * Formatter string builder, macros, etc.
 *
 * @ingroup group_core
 */

// Used in this file.
#include <cstdint>

// Shared by all code.
#include <algorithm>
#include <cassert>
#include <format>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

/// Throw exception @p EXC carrying a "@p CLS: @p MSG" message.
#define SOLVCON_EXCEPT(CLS, EXC, MSG) throw EXC(#CLS ": " MSG);

/// Width in bytes of the solvcon integer type: 4 (int32) or 8 (int64).
#ifndef SOLVCON_INTSIZE
#define SOLVCON_INTSIZE 4
#endif // SOLVCON_INTSIZE

namespace solvcon
{

namespace detail
{

/// Helper trait to check if a type is a specialization of a given template
template <template <typename...> class Template, typename T>
struct is_specialization_of : std::false_type
{
}; /* end struct is_specialization_of */

/// Helper trait to check if a type is a specialization of a given template
template <template <typename...> class Template, typename... Args>
struct is_specialization_of<Template, Template<Args...>> : std::true_type
{
}; /* end struct is_specialization_of */

/// Helper trait to check if a type is a specialization of a given template
template <template <typename...> class Template, typename T>
inline constexpr bool is_specialization_of_v = is_specialization_of<Template, T>::value;

inline bool is_whitespace(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

} /* end namespace detail */

using real_type = double;

#if 4 == SOLVCON_INTSIZE
using uint_type = uint32_t;
using int_type = int32_t;
#elif 8 == SOLVCON_INTSIZE
using uint_type = uint64_t;
using int_type = int64_t;
#else // SOLVCON_INTSIZE
#error SOLVCON_INTSIZE is not supported
#endif // SOLVCON_INTSIZE

/**
 * Fixes the integer and real value types for spatial data structures.
 *
 * @tparam I Integral index type.
 * @tparam R Floating-point real type.
 *
 * @ingroup group_core
 */
template <typename I, typename R>
class NumberBase
{

public:

    static_assert(std::is_integral_v<I>, "I must be integral type");
    static_assert(std::is_floating_point_v<R>, "R must be floating-point type");

    using int_type = I;
    using uint_type = std::make_unsigned_t<I>;
    using size_type = uint_type;
    using real_type = R;

}; /* end class NumberBase */

/**
 * Spatial table basic information.  Any table-based data store for spatial
 * data should inherit this class template.
 *
 * @ingroup group_core
 */
template <uint8_t ND, typename I, typename R>
class SpaceBase : public NumberBase<I, R>
{

public:

    using dim_type = uint8_t;
    static constexpr const dim_type NDIM = ND;

    using number_base = NumberBase<I, R>;

    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using size_type = typename number_base::size_type;
    using serial_type = typename number_base::size_type;
    using real_type = typename number_base::real_type;

}; /* end class SpaceBase */

/**
 * Deprecated ostringstream-based string builder; prefer std::format.
 *
 * Streams values with operator<< and yields the accumulated text through
 * str() or an implicit std::string conversion. Taken from
 * https://stackoverflow.com/a/12262626 .
 *
 * @ingroup group_core
 */
class [[deprecated("Use std::format instead")]] Formatter
{

public:

    Formatter() = default; // NOLINT(fuchsia-default-arguments) not sure why it's needed
    Formatter(Formatter const &) = delete;
    Formatter(Formatter &&) = delete;
    Formatter & operator=(Formatter const &) = delete;
    Formatter & operator=(Formatter &&) = delete;
    ~Formatter() = default;

    template <typename T>
    Formatter & operator<<(T const & value)
    {
        m_stream << value;
        return *this;
    }

    std::string str() const { return m_stream.str(); }
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator std::string() const { return m_stream.str(); }

    enum ConvertToString : std::uint8_t
    {
        to_str
    };

    std::string operator>>(ConvertToString const &) { return m_stream.str(); }

private:

    std::ostringstream m_stream;

}; /* end class Formatter */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
