#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/simd/simd_generic.hpp>
#include <solvcon/simd/simd_support.hpp>

#include <solvcon/simd/neon/neon.hpp>

namespace solvcon
{

namespace simd
{

// Check if each element from start to end (excluded end) is within the range [min_val, max_val)
template <typename T>
const T * check_between(T const * start, T const * end, T const & min_val, T const & max_val)
{
    switch (detail::detect_simd())
    {
    case detail::SIMD_NEON:
        return neon::check_between<T>(start, end, min_val, max_val);
        break;

    default:
        return generic::check_between<T>(start, end, min_val, max_val);
    }
}

template <typename T>
void add(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    using namespace detail; // FIXME: NOLINT(google-build-using-namespace)
    switch (detect_simd())
    {
    case SIMD_NEON:
        return neon::add<T>(dest, dest_end, src1, src2);
        break;

    default:
        return generic::add<T>(dest, dest_end, src1, src2);
    }
}

template <typename T>
void sub(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    using namespace detail; // FIXME: NOLINT(google-build-using-namespace)
    switch (detect_simd())
    {
    case SIMD_NEON:
        return neon::sub<T>(dest, dest_end, src1, src2);
        break;

    default:
        return generic::sub<T>(dest, dest_end, src1, src2);
    }
}

template <typename T>
void mul(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    using namespace detail; // FIXME: NOLINT(google-build-using-namespace)
    switch (detect_simd())
    {
    case SIMD_NEON:
        return neon::mul<T>(dest, dest_end, src1, src2);
        break;

    default:
        return generic::mul<T>(dest, dest_end, src1, src2);
    }
}

template <typename T>
void div(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    using namespace detail; // FIXME: NOLINT(google-build-using-namespace)
    switch (detect_simd())
    {
    case SIMD_NEON:
        return neon::div<T>(dest, dest_end, src1, src2);
        break;

    default:
        return generic::div<T>(dest, dest_end, src1, src2);
    }
}

template <typename T>
T max(T const * start, T const * end)
{
    return generic::max<T>(start, end);
}

} /* end namespace simd */

} /* end namespace solvcon */
