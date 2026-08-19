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

#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

#include <solvcon/buffer/buffer.hpp>

namespace solvcon
{

namespace timeseries
{

namespace detail
{

/// Reject an array whose timestamps decrease.
inline void validate_sorted(SimpleArray<uint64_t> const & array, char const * op, std::string_view what)
{
    uint64_t const * const src = array.logical_data();
    ssize_t const step = array.stride(0);
    for (ssize_t i = 1; i < array.shape(0); ++i)
    {
        uint64_t const prev = src[(i - 1) * step], cur = src[i * step];
        if (cur < prev)
        {
            // clang-format off
            throw std::invalid_argument(std::format(
                "timeseries::{}(): {} must be non-decreasing but element {} = {} is less than element {} = {}",
                op, what, i, cur, i - 1, prev));
            // clang-format on
        }
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

} /* end namespace timeseries */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
