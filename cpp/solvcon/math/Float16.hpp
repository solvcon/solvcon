#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Portable IEEE 754 binary16 storage and conversion.
 *
 * @ingroup group_core
 */

#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#if __has_include(<stdfloat>)
#include <stdfloat>
#endif
#include <type_traits>

#if __has_include(<stdfloat>) && defined(__STDCPP_FLOAT16_T__)
#define SC_FLOAT16_NATIVE_TYPE std::float16_t
#elif defined(__FLT16_MANT_DIG__) && __FLT16_MANT_DIG__ == 11
#define SC_FLOAT16_NATIVE_TYPE _Float16
#endif

namespace solvcon
{

/**
 * Two-byte IEEE 754 binary16 value with stable cross-platform type identity.
 *
 * Native compiler conversions follow the current rounding mode. Other
 * toolchains use a bitwise round-to-nearest-even fallback.
 *
 * @ingroup group_core
 */
class Float16
{

public:

    using storage_type = uint16_t;

    Float16() = default;
    Float16(Float16 const &) = default;
    Float16(Float16 &&) = default;
    Float16 & operator=(Float16 const &) = default;
    Float16 & operator=(Float16 &&) = default;
    ~Float16() = default;

    /**
     * Construct from a single-precision value.
     *
     * @param value Source value.
     */
    constexpr Float16(float value); // NOLINT(google-explicit-constructor)

    /**
     * Construct from a double-precision value.
     *
     * @param value Source value.
     */
    constexpr Float16(double value); // NOLINT(google-explicit-constructor)

    /**
     * Construct from an integral value.
     *
     * @tparam T Integral source type.
     * @param value Source value.
     */
    template <std::integral T>
    constexpr Float16(T value); // NOLINT(google-explicit-constructor)

    /**
     * Convert to a single-precision value.
     *
     * @return The represented value as ``float``.
     */
    explicit constexpr operator float() const { return decode(m_bits); }

    /**
     * Return the raw binary16 representation.
     *
     * @return The sign, exponent, and fraction bits.
     */
    constexpr storage_type bits() const { return m_bits; }

    /**
     * Construct directly from a raw binary16 representation.
     *
     * @param bits The sign, exponent, and fraction bits.
     * @return A value containing the supplied representation.
     */
    static constexpr Float16 from_bits(storage_type bits);

private:

#ifdef SC_FLOAT16_NATIVE_TYPE
    using native_type = SC_FLOAT16_NATIVE_TYPE;
    static_assert(sizeof(native_type) == sizeof(storage_type));
#endif

    template <typename T>
    static constexpr storage_type encode(T value);
    static constexpr float decode(storage_type value_bits);
    template <typename T>
    static constexpr storage_type encode_fallback(T value);
    static constexpr float decode_fallback(storage_type value_bits);

    storage_type m_bits;

}; /* end class Float16 */

constexpr Float16 Float16::from_bits(storage_type bits) { return std::bit_cast<Float16>(bits); }

template <typename T>
constexpr Float16::storage_type Float16::encode(T value)
{
#ifdef SC_FLOAT16_NATIVE_TYPE
    return std::bit_cast<storage_type>(static_cast<native_type>(value));
#else
    return encode_fallback(value);
#endif
}

constexpr Float16::Float16(float value)
    : m_bits(encode(value))
{
}

constexpr Float16::Float16(double value)
    : m_bits(encode(value))
{
}

template <std::integral T>
constexpr Float16::Float16(T value)
    : Float16(static_cast<double>(value))
{
}

constexpr float Float16::decode(storage_type value_bits)
{
#ifdef SC_FLOAT16_NATIVE_TYPE
    return static_cast<float>(std::bit_cast<native_type>(value_bits));
#else
    return decode_fallback(value_bits);
#endif
}

template <typename T>
constexpr Float16::storage_type Float16::encode_fallback(T value)
{
    static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));
    static_assert(std::numeric_limits<T>::is_iec559);

    using bits_type = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint64_t>;
    constexpr uint32_t frac_bits = std::numeric_limits<T>::digits - 1;
    constexpr uint32_t exp_bits = sizeof(T) * 8 - frac_bits - 1;
    constexpr bits_type exp_mask = (bits_type{1} << exp_bits) - 1;
    constexpr bits_type frac_mask = (bits_type{1} << frac_bits) - 1;
    constexpr int32_t exp_bias = (1 << (exp_bits - 1)) - 1;

    auto const value_bits = std::bit_cast<bits_type>(value);
    uint32_t const sign = static_cast<uint32_t>(value_bits >> (frac_bits + exp_bits)) << 15;
    bits_type const exponent = (value_bits >> frac_bits) & exp_mask;
    bits_type const fraction = value_bits & frac_mask;

    if (exponent == exp_mask)
    {
        if (fraction == 0)
        {
            return static_cast<storage_type>(sign | 0x7c00U);
        }
        bits_type const payload = (fraction >> (frac_bits - 10)) | 0x200U;
        return static_cast<storage_type>(sign | 0x7c00U | payload);
    }

    int32_t half_exp = static_cast<int32_t>(exponent) - exp_bias + 15;
    if (half_exp < -10)
    {
        return static_cast<storage_type>(sign);
    }

    if (half_exp <= 0)
    {
        bits_type const significand = fraction | (bits_type{1} << frac_bits);
        auto const shift = static_cast<uint32_t>(static_cast<int32_t>(frac_bits) - 9 - half_exp);
        bits_type rounded = significand >> shift;
        bits_type const remainder = significand & ((bits_type{1} << shift) - 1);
        bits_type const halfway = bits_type{1} << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U)))
        {
            ++rounded;
        }
        return static_cast<storage_type>(sign | rounded);
    }

    if (half_exp >= 31)
    {
        return static_cast<storage_type>(sign | 0x7c00U);
    }

    constexpr uint32_t shift = frac_bits - 10;
    bits_type rounded = fraction >> shift;
    bits_type const remainder = fraction & ((bits_type{1} << shift) - 1);
    bits_type const halfway = bits_type{1} << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U)))
    {
        ++rounded;
        if (rounded == 0x400U)
        {
            rounded = 0;
            ++half_exp;
        }
    }
    if (half_exp >= 31)
    {
        return static_cast<storage_type>(sign | 0x7c00U);
    }
    return static_cast<storage_type>(sign | (static_cast<uint32_t>(half_exp) << 10) | rounded);
}

constexpr float Float16::decode_fallback(storage_type value_bits)
{
    uint32_t const sign = static_cast<uint32_t>(value_bits & 0x8000U) << 16;
    uint32_t const exponent = (value_bits >> 10) & 0x1fU;
    uint32_t fraction = value_bits & 0x3ffU;
    uint32_t result = sign;

    if (exponent == 0)
    {
        if (fraction != 0)
        {
            int32_t unbiased_exp = -14;
            while ((fraction & 0x400U) == 0)
            {
                fraction <<= 1;
                --unbiased_exp;
            }
            fraction &= 0x3ffU;
            result |= static_cast<uint32_t>(unbiased_exp + 127) << 23;
            result |= fraction << 13;
        }
    }
    else if (exponent == 0x1fU)
    {
        result |= 0x7f800000U;
        if (fraction != 0)
        {
            result |= (fraction << 13) | 0x00400000U;
        }
    }
    else
    {
        result |= (exponent + 112U) << 23;
        result |= fraction << 13;
    }
    return std::bit_cast<float>(result);
}

static_assert(sizeof(Float16) == 2);
static_assert(std::is_trivially_copyable_v<Float16>);

} /* end namespace solvcon */

#undef SC_FLOAT16_NATIVE_TYPE

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
