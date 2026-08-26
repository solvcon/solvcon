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
#include <cfenv>
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
 * Storage uses one sign bit, five exponent bits, and ten fraction bits.
 * Software fallback conversions follow the current rounding mode at runtime.
 * Native conversions require compiler support for dynamic rounding to be
 * enabled. Constant evaluation rounds to nearest with ties to even.
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
     * @return Bit 15 is the sign, bits 14-10 are the exponent, and bits 9-0
     * are the fraction.
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

    /** Return the active floating-point rounding mode. */
#ifdef _MSC_VER
    static int rounding_mode();
#else
    static int rounding_mode() { return std::fegetround(); }
#endif

    /**
     * Decide whether an inexact result uses the next greater magnitude.
     *
     * A greater magnitude is away from zero for either sign. Directed modes
     * use the sign, while round-to-nearest uses the supplied ties-to-even
     * decision.
     *
     * @param mode Active floating-point rounding mode.
     * @param negative Whether the source sign bit is set.
     * @param nearest_away Whether round-to-nearest selects greater magnitude.
     * @return True when rounding selects greater magnitude.
     */
    static constexpr bool should_round_away(int mode, bool negative, bool nearest_away);

    /**
     * Decide rounding from the low significand bits removed by binary16.
     *
     * Source significand = [ retained bits ][ discarded bits ]
     * Halfway            =                   100...0
     *
     * A zero discarded field is exact. Beyond halfway rounds away under
     * ``FE_TONEAREST``. At halfway, an odd retained value rounds away so the
     * resulting least-significant bit is even.
     *
     * @tparam T Unsigned type holding the significand bits.
     * @param mode Active floating-point rounding mode.
     * @param negative Whether the source sign bit is set.
     * @param retained High significand bits kept in the binary16 candidate.
     * @param discarded Low significand bits removed from the source.
     * @param halfway One followed by zeros in the discarded field width.
     * @return True when rounding selects greater magnitude.
     */
    template <typename T>
    static constexpr bool should_round_away(
        int mode, bool negative, T retained, T discarded, T halfway);
    template <typename T>
    static constexpr storage_type encode(T value);
    static constexpr float decode(storage_type value_bits);

    /**
     * Convert an IEEE 754 binary32 or binary64 bit pattern to binary16.
     *
     * binary32 = [ sign:1 ][ exponent:8  ][ fraction:23 ]
     * binary64 = [ sign:1 ][ exponent:11 ][ fraction:52 ]
     * binary16 = [ sign:1 ][ exponent:5  ][ fraction:10 ]
     *
     * @tparam T ``float`` or ``double`` source type.
     * @param value Source value.
     * @return Encoded binary16 bits.
     * @see https://standards.ieee.org/ieee/754/6210/
     */
    template <typename T>
    static constexpr storage_type encode_fallback(T value);
    static constexpr float decode_fallback(storage_type value_bits);

    storage_type m_bits;

}; /* end class Float16 */

constexpr Float16 Float16::from_bits(storage_type bits) { return std::bit_cast<Float16>(bits); }

constexpr bool Float16::should_round_away(int mode, bool negative, bool nearest_away)
{
    switch (mode)
    {
    case FE_UPWARD:
        return !negative;
    case FE_DOWNWARD:
        return negative;
    case FE_TOWARDZERO:
        return false;
    case FE_TONEAREST:
    default:
        return nearest_away;
    }
}

template <typename T>
constexpr bool Float16::should_round_away(
    int mode, bool negative, T retained, T discarded, T halfway)
{
    if (discarded == 0)
    {
        return false;
    }
    bool const nearest_away = discarded > halfway || (discarded == halfway && (retained & 1U));
    return should_round_away(mode, negative, nearest_away);
}

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

    // Constant evaluation cannot query the runtime floating-point environment.
    int mode = FE_TONEAREST;
    if (!std::is_constant_evaluated())
    {
        mode = rounding_mode();
    }

    using bits_type = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint64_t>;
    constexpr uint32_t frac_bits = std::numeric_limits<T>::digits - 1;
    constexpr uint32_t exp_bits = sizeof(T) * 8 - frac_bits - 1;
    constexpr bits_type exp_mask = (bits_type{1} << exp_bits) - 1;
    constexpr bits_type frac_mask = (bits_type{1} << frac_bits) - 1;
    constexpr int32_t exp_bias = (1 << (exp_bits - 1)) - 1;

    auto const value_bits = std::bit_cast<bits_type>(value);
    // Move the source sign to binary16 bit 15; round the remaining magnitude.
    uint32_t const sign = static_cast<uint32_t>(value_bits >> (frac_bits + exp_bits)) << 15;
    bool const negative = sign != 0;
    bits_type const exponent = (value_bits >> frac_bits) & exp_mask;
    bits_type const fraction = value_bits & frac_mask;

    if (exponent == exp_mask)
    {
        // Preserve infinity; truncate NaN payloads and force a quiet binary16 NaN.
        if (fraction == 0)
        {
            return static_cast<storage_type>(sign | 0x7c00U);
        }
        bits_type const payload = (fraction >> (frac_bits - 10)) | 0x200U;
        return static_cast<storage_type>(sign | 0x7c00U | payload);
    }

    // Remove the source bias and apply the binary16 exponent bias of 15.
    int32_t half_exp = static_cast<int32_t>(exponent) - exp_bias + 15;
    if (half_exp < -10)
    {
        // Only zero and the minimum subnormal can represent this magnitude.
        bool const is_nonzero = exponent != 0 || fraction != 0;
        if (is_nonzero && should_round_away(mode, negative, false))
        {
            return static_cast<storage_type>(sign | 1U);
        }
        return static_cast<storage_type>(sign);
    }

    if (half_exp <= 0)
    {
        // Shift the implicit leading bit into the ten-bit subnormal fraction.
        bits_type const significand = fraction | (bits_type{1} << frac_bits);
        auto const shift = static_cast<uint32_t>(static_cast<int32_t>(frac_bits) - 9 - half_exp);
        bits_type retained = significand >> shift;
        bits_type const discarded = significand & ((bits_type{1} << shift) - 1);
        bits_type const halfway = bits_type{1} << (shift - 1);
        if (should_round_away(mode, negative, retained, discarded, halfway))
        {
            ++retained;
        }
        return static_cast<storage_type>(sign | retained);
    }

    if (half_exp >= 31)
    {
        // Directed overflow chooses infinity or the largest finite binary16.
        bool const to_infinity = should_round_away(mode, negative, true);
        return static_cast<storage_type>(sign | (to_infinity ? 0x7c00U : 0x7bffU));
    }

    constexpr uint32_t shift = frac_bits - 10;
    bits_type retained = fraction >> shift;
    bits_type const discarded = fraction & ((bits_type{1} << shift) - 1);
    bits_type const halfway = bits_type{1} << (shift - 1);
    if (should_round_away(mode, negative, retained, discarded, halfway))
    {
        ++retained;
        if (retained == 0x400U)
        {
            retained = 0;
            ++half_exp;
        }
    }
    if (half_exp >= 31)
    {
        return static_cast<storage_type>(sign | 0x7c00U);
    }
    return static_cast<storage_type>(sign | (static_cast<uint32_t>(half_exp) << 10) | retained);
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
