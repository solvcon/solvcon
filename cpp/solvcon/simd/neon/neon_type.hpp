#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#ifdef __aarch64__

#include <arm_neon.h>
#include <cstddef>

namespace solvcon
{

namespace simd
{

namespace neon
{

namespace type
{

namespace detail
{

template <typename T>
struct vector
{
    static constexpr size_t N_lane = 0;
}; /* end struct vector */

template <>
struct vector<uint8_t>
{
    using type = uint8x16_t;
    static constexpr size_t N_lane = 16;
}; /* end struct vector */

template <>
struct vector<uint16_t>
{
    using type = uint16x8_t;
    static constexpr size_t N_lane = 8;
}; /* end struct vector */

template <>
struct vector<uint32_t>
{
    using type = uint32x4_t;
    static constexpr size_t N_lane = 4;
}; /* end struct vector */

template <>
struct vector<uint64_t>
{
    using type = uint64x2_t;
    static constexpr size_t N_lane = 2;
}; /* end struct vector */

template <>
struct vector<int8_t>
{
    using type = int8x16_t;
    static constexpr size_t N_lane = 16;
}; /* end struct vector */

template <>
struct vector<int16_t>
{
    using type = int16x8_t;
    static constexpr size_t N_lane = 8;
}; /* end struct vector */

template <>
struct vector<int32_t>
{
    using type = int32x4_t;
    static constexpr size_t N_lane = 4;
}; /* end struct vector */

template <>
struct vector<int64_t>
{
    using type = int64x2_t;
    static constexpr size_t N_lane = 2;
}; /* end struct vector */

template <>
struct vector<float>
{
    using type = float32x4_t;
    static constexpr size_t N_lane = 4;
}; /* end struct vector */

template <>
struct vector<double>
{
    using type = float64x2_t;
    static constexpr size_t N_lane = 2;
}; /* end struct vector */

} /* end namespace detail */

template <typename T>
using vector_t = typename detail::vector<T>::type;

template <typename T>
inline constexpr size_t vector_lane = detail::vector<T>::N_lane;

template <typename T>
inline constexpr bool has_vectype = detail::vector<T>::N_lane > 0;

} /* end namespace type */

} /* end namespace neon */

} /* end namespace simd */

} /* end namespace solvcon */

#endif /* defined(__aarch64__) */
