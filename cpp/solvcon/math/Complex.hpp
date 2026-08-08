#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Complex number type with a std::complex-compatible layout and the
 * related type traits.
 *
 * @ingroup group_core
 */

#include <type_traits>
#include <cmath>
#include <stdexcept>
#include <complex>

namespace solvcon
{

namespace detail
{

template <typename T>
struct ComplexImpl
{
    static_assert(std::is_floating_point_v<T>);

    T real_v;
    T imag_v;

    ComplexImpl() = default;

    ComplexImpl(T r, T i)
        : real_v(r)
        , imag_v(i)
    {
    }

    // FIXME: NOLINTNEXTLINE(google-explicit-constructor)
    ComplexImpl(T t)
        : real_v(t)
        , imag_v(0.0)
    {
    }

    // FIXME: NOLINTNEXTLINE(google-explicit-constructor)
    ComplexImpl(std::complex<T> const & c)
        : real_v(c.real())
        , imag_v(c.imag())
    {
    }

    ComplexImpl & operator+=(const ComplexImpl & other)
    {
        real_v += other.real_v;
        imag_v += other.imag_v;
        return *this;
    }

    ComplexImpl & operator+=(T other)
    {
        real_v += other;
        return *this;
    }

    ComplexImpl & operator-=(const ComplexImpl & other)
    {
        real_v -= other.real_v;
        imag_v -= other.imag_v;
        return *this;
    }

    ComplexImpl & operator-=(T other)
    {
        real_v -= other;
        return *this;
    }

    ComplexImpl & operator*=(const ComplexImpl<T> & rhs)
    {
        T real_v_copy = real_v;
        real_v = real_v_copy * rhs.real_v - imag_v * rhs.imag_v;
        imag_v = real_v_copy * rhs.imag_v + imag_v * rhs.real_v;
        return *this;
    }

    ComplexImpl & operator*=(const T & rhs)
    {
        real_v *= rhs;
        imag_v *= rhs;
        return *this;
    }

    ComplexImpl & operator/=(const ComplexImpl<T> & rhs)
    {
        T denominator = rhs.norm();
        T real_v_copy = real_v;

        if (denominator == 0.0)
        {
            throw std::runtime_error("Division by zero in complex number");
        }

        real_v = (real_v * rhs.real_v + imag_v * rhs.imag_v) / denominator;
        imag_v = (imag_v * rhs.real_v - real_v_copy * rhs.imag_v) / denominator;
        return *this;
    }

    ComplexImpl & operator/=(const T & rhs)
    {
        if (rhs == 0.0)
        {
            throw std::runtime_error("Division by zero in complex number");
        }
        real_v /= rhs;
        imag_v /= rhs;
        return *this;
    }

    std::complex<T> to_std_complex() const { return std::complex<T>(real_v, imag_v); }
    T real() const { return real_v; }
    T imag() const { return imag_v; }
    T norm() const { return real_v * real_v + imag_v * imag_v; }
    ComplexImpl<T> conj() const { return {real_v, -imag_v}; }
}; /* end struct ComplexImpl */

/**
 * These comparison operators use lexicographic ordering and would be used in
 * SimpleArray::min() and SimpleArray::max(). The use of lexicographic ordering
 * is to match the numpy behaviors documented in
 * https://numpy.org/devdocs/reference/generated/numpy.sort.html . The
 * following discussions include more details:
 * more details:
 * 1. https://github.com/numpy/numpy/issues/12943
 * 2. https://stackoverflow.com/questions/52481376
 */
template <typename T>
bool operator<(const ComplexImpl<T> & lhs, const ComplexImpl<T> & rhs)
{
    if (lhs.real_v == rhs.real_v)
    {
        return lhs.imag_v < rhs.imag_v;
    }
    return lhs.real_v < rhs.real_v;
}

template <typename T>
bool operator>(const ComplexImpl<T> & lhs, const ComplexImpl<T> & rhs)
{
    if (lhs.real_v == rhs.real_v)
    {
        return lhs.imag_v > rhs.imag_v;
    }
    return lhs.real_v > rhs.real_v;
}

template <typename T>
bool operator==(const ComplexImpl<T> & lhs, const ComplexImpl<T> & rhs)
{
    return lhs.real_v == rhs.real_v && lhs.imag_v == rhs.imag_v;
}

template <typename T>
bool operator!=(const ComplexImpl<T> & lhs, const ComplexImpl<T> & rhs)
{
    return lhs.real_v != rhs.real_v || lhs.imag_v != rhs.imag_v;
}

template <typename T>
ComplexImpl<T> operator+(ComplexImpl<T> lhs, const ComplexImpl<T> & rhs)
{
    return lhs += rhs;
}

template <typename T>
ComplexImpl<T> operator+(ComplexImpl<T> lhs, T rhs)
{
    return lhs += rhs;
}

template <typename T>
ComplexImpl<T> operator+(T lhs, const ComplexImpl<T> & rhs)
{
    return ComplexImpl<T>{lhs, 0.0} += rhs;
}

template <typename T>
ComplexImpl<T> operator-(ComplexImpl<T> lhs, const ComplexImpl<T> & rhs)
{
    return lhs -= rhs;
}

template <typename T>
ComplexImpl<T> operator-(ComplexImpl<T> lhs, T rhs)
{
    return lhs -= rhs;
}

template <typename T>
ComplexImpl<T> operator-(T lhs, const ComplexImpl<T> & rhs)
{
    return ComplexImpl<T>{lhs, 0.0} -= rhs;
}

template <typename T>
ComplexImpl<T> operator*(ComplexImpl<T> lhs, const ComplexImpl<T> & rhs)
{
    return lhs *= rhs;
}

template <typename T>
ComplexImpl<T> operator*(ComplexImpl<T> lhs, T rhs)
{
    return lhs *= rhs;
}

template <typename T>
ComplexImpl<T> operator*(T lhs, const ComplexImpl<T> & rhs)
{
    return ComplexImpl<T>{lhs, 0.0} *= rhs;
}

template <typename T>
ComplexImpl<T> operator/(ComplexImpl<T> lhs, const ComplexImpl<T> & rhs)
{
    return lhs /= rhs;
}

template <typename T>
ComplexImpl<T> operator/(ComplexImpl<T> lhs, const T & rhs)
{
    return lhs /= rhs;
}

template <typename T>
ComplexImpl<T> operator/(T lhs, const ComplexImpl<T> & rhs)
{
    return ComplexImpl<T>{lhs, 0.0} /= rhs;
}

} /* end namespace detail */

template <typename T>
using Complex = detail::ComplexImpl<T>;

template <typename T>
inline constexpr bool is_std_complex_layout_compatible_v = std::is_standard_layout_v<Complex<T>> &&
                                                           sizeof(Complex<T>) == sizeof(std::complex<T>) &&
                                                           alignof(Complex<T>) == alignof(std::complex<T>);

static_assert(is_std_complex_layout_compatible_v<float>);
static_assert(is_std_complex_layout_compatible_v<double>);

template <typename T>
std::complex<T> const * as_std_complex_pointer(Complex<T> const * ptr)
{
    return reinterpret_cast<std::complex<T> const *>(ptr); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

template <typename T>
std::complex<T> * as_std_complex_pointer(Complex<T> * ptr)
{
    return reinterpret_cast<std::complex<T> *>(ptr); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

// clang-format off
/**
 * Type trait that reports whether a type is a solvcon Complex.
 *
 * The primary template inherits std::false_type; the specialization for
 * Complex<T> inherits std::true_type.
 *
 * @ingroup group_core
 */
template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<Complex<T>> : std::true_type {};

/**
 * Type trait that reports whether a type is a real floating-point type.
 *
 * @ingroup group_core
 */
template <typename T>
struct is_real : std::is_floating_point<T> {};
// clang-format on

template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

template <typename T>
constexpr bool is_real_v = is_real<T>::value;

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
