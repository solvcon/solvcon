#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The interface header file for the time-series kernels.
 *
 * @ingroup group_numerics
 */

#include <algorithm>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <solvcon/buffer/buffer.hpp>

namespace solvcon
{

namespace timeseries
{

namespace detail
{

template <typename T>
void validate_series(SimpleArray<uint64_t> const & times, SimpleArray<T> const & values, char const * op)
{
    solvcon::detail::validate_1d<std::invalid_argument>(times, op, "times", "timeseries");
    solvcon::detail::validate_1d<std::invalid_argument>(values, op, "values", "timeseries");
    if (times.shape(0) != values.shape(0))
    {
        throw std::invalid_argument(std::format(
            "timeseries::{}(): times has {} samples but values has {}", op, times.shape(0), values.shape(0)));
    }

    // TODO: we don't support ghosted time series yet
    if (times.nghost() > 0 || values.nghost() > 0)
    {
        throw std::invalid_argument(std::format("timeseries::{}(): ghosted time series are not supported yet", op));
    }
}

/// Throw `std::invalid_argument` when the timestamps decrease; `strict` also rejects a repeat.
inline void validate_sorted(
    SimpleArray<uint64_t> const & array, char const * op, std::string_view what, bool strict = false)
{
    uint64_t const * const src = array.logical_data();
    ssize_t const step = array.stride(0);
    for (ssize_t i = 1; i < array.shape(0); ++i)
    {
        uint64_t const prev = src[(i - 1) * step], cur = src[i * step];
        if (cur < prev || (strict && cur == prev))
        {
            char const * const order = strict ? "strictly increasing" : "non-decreasing";
            char const * const relation = strict ? "does not exceed" : "is less than";
            // clang-format off
            throw std::invalid_argument(std::format(
                "timeseries::{}(): {} must be {} but element {} = {} {} element {} = {}",
                op, what, order, i, cur, relation, i - 1, prev));
            // clang-format on
        }
    }
}

template <typename R, typename T>
R subtract_exact(T x1, T x0)
{
    static_assert(std::is_floating_point_v<R>, "subtract_exact() must return a floating-point type");

    if constexpr (std::is_integral_v<T>)
    {
        using unsigned_type = std::make_unsigned_t<T>;
        auto const u1 = static_cast<unsigned_type>(x1), u0 = static_cast<unsigned_type>(x0);
        return x1 >= x0
                   ? static_cast<R>(static_cast<unsigned_type>(u1 - u0))
                   : -static_cast<R>(static_cast<unsigned_type>(u0 - u1));
    }
    else
    {
        return x1 - x0;
    }
}

} /* end namespace detail */

/**
 * Merge sorted one-dimensional timestamp arrays into one sorted array that
 * holds every distinct timestamp once. Each input must be non-decreasing; a
 * timestamp repeated within one array or shared between arrays appears once
 * in the result. No input, or only empty input, gives an empty result.
 *
 * @ingroup group_numerics
 */
inline SimpleArray<uint64_t> merge_sorted_unique(small_vector<SimpleArray<uint64_t> const *> const & arrays)
{
    struct Cursor
    {
        uint64_t const * head;
        ssize_t step;
        ssize_t left;
    }; /* end struct Cursor */

    small_vector<Cursor> cursors;
    ssize_t total = 0;
    for (size_t k = 0; k < arrays.size(); ++k)
    {
        SimpleArray<uint64_t> const & array = *arrays[k];
        std::string const what = std::format("array {}", k);
        solvcon::detail::validate_1d<std::invalid_argument>(array, "merge_sorted_unique", what, "timeseries");
        detail::validate_sorted(array, "merge_sorted_unique", what);
        cursors.push_back(Cursor{.head = array.logical_data(), .step = array.stride(0), .left = array.shape(0)});
        total += array.shape(0);
    }

    SimpleCollector<uint64_t> merged;
    merged.reserve(total);
    while (true)
    {
        Cursor * lowest = nullptr;
        for (Cursor & c : cursors)
        {
            if (c.left > 0 && (lowest == nullptr || *c.head < *lowest->head))
            {
                lowest = &c;
            }
        }
        if (lowest == nullptr)
        {
            break;
        }

        uint64_t const value = *lowest->head;
        if (merged.empty() || merged.back() != value)
        {
            merged.push_back(value);
        }

        lowest->head += lowest->step;
        --lowest->left;
    }

    return merged.as_array();
}

/**
 * Keep the last sample of every group of equal timestamps.
 *
 * @tparam T The value type, which may be any type `SimpleArray` holds.
 * @param times The sample timestamps in nanoseconds, non-decreasing.
 * @param values The values sampled at @p times, of the same length.
 * @return The kept timestamps, strictly increasing, and the values kept with them. A series with no repeat comes
 *         back as a copy.
 * @throw std::invalid_argument An array that is not one-dimensional or that carries ghost elements, a length
 *        mismatch, or a decreasing timestamp.
 * @throw std::logic_error The counting pass and the filling pass disagreed, which is a bug in this function.
 *
 * @ingroup group_numerics
 */
template <typename T>
std::pair<SimpleArray<uint64_t>, SimpleArray<T>>
dedup_last(SimpleArray<uint64_t> const & times, SimpleArray<T> const & values)
{
    detail::validate_series(times, values, "dedup_last");
    detail::validate_sorted(times, "dedup_last", "times");

    ssize_t const nsample = times.shape(0);
    uint64_t const * const tsrc = times.logical_data();
    ssize_t const tstep = times.stride(0);

    ssize_t ngroup = nsample > 0 ? 1 : 0;
    for (ssize_t i = 1; i < nsample; ++i)
    {
        if (tsrc[i * tstep] != tsrc[(i - 1) * tstep])
        {
            ++ngroup;
        }
    }

    SimpleArray<uint64_t> otimes(ngroup);
    SimpleArray<T> ovalues(ngroup);
    T const * const vsrc = values.logical_data();
    ssize_t const vstep = values.stride(0);
    ssize_t igroup = 0;
    for (ssize_t i = 0; i < nsample; ++i)
    {
        if (i + 1 == nsample || tsrc[(i + 1) * tstep] != tsrc[i * tstep])
        {
            otimes[igroup] = tsrc[i * tstep];
            ovalues[igroup] = vsrc[i * vstep];
            ++igroup;
        }
    }

    if (igroup != ngroup)
    {
        throw std::logic_error(
            std::format("timeseries::dedup_last(): counted {} groups but filled {}", ngroup, igroup));
    }

    return {std::move(otimes), std::move(ovalues)};
}

/**
 * The element type `deriv()` returns.
 *
 * @tparam T The value type of the input series.
 *
 * @ingroup group_numerics
 */
template <typename T>
using deriv_value_t = std::conditional_t<std::is_floating_point_v<T>, T, double>;

/**
 * Differentiate a series by the backward difference `(x_i - x_{i-1}) / (t_i - t_{i-1})`.
 *
 * Integer values subtract exactly, so a falling unsigned signal gives a negative derivative instead of wrapping.
 * The cast of that difference to the result type rounds a magnitude above 2^53.
 *
 * @tparam T The value type, which must be a real number type.
 * @param times The sample timestamps in nanoseconds, strictly increasing.
 * @param values The values sampled at @p times, of the same length.
 * @return `times[1:]` and one derivative per remaining sample; the first sample has no predecessor. Fewer than two
 *         samples give empty arrays.
 * @throw std::invalid_argument An array that is not one-dimensional or that carries ghost elements, a length
 *        mismatch, or a decreasing or repeated timestamp. A repeat has no time step to divide by, and `dedup_last()` collapses
 *        one beforehand.
 *
 * @ingroup group_numerics
 */
template <typename T>
std::pair<SimpleArray<uint64_t>, SimpleArray<deriv_value_t<T>>>
deriv(SimpleArray<uint64_t> const & times, SimpleArray<T> const & values)
{
    static_assert(std::is_arithmetic_v<T> && !solvcon::detail::is_bool_v<T>, "deriv() requires a real number type");
    using result_type = deriv_value_t<T>;

    detail::validate_series(times, values, "deriv");
    detail::validate_sorted(times, "deriv", "times", /*strict*/ true);

    ssize_t const nsample = times.shape(0);
    ssize_t const nout = std::max<ssize_t>(nsample - 1, 0);
    SimpleArray<uint64_t> otimes(nout);
    SimpleArray<result_type> oderiv(nout);

    uint64_t const * const tsrc = times.logical_data();
    ssize_t const tstep = times.stride(0);
    T const * const vsrc = values.logical_data();
    ssize_t const vstep = values.stride(0);
    for (ssize_t i = 1; i < nsample; ++i)
    {
        uint64_t const cur = tsrc[i * tstep], prev = tsrc[(i - 1) * tstep];
        otimes[i - 1] = cur;
        auto const dvalue = detail::subtract_exact<result_type>(vsrc[i * vstep], vsrc[(i - 1) * vstep]);
        oderiv[i - 1] = dvalue / static_cast<result_type>(cur - prev);
    }

    return {std::move(otimes), std::move(oderiv)};
}

} /* end namespace timeseries */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
