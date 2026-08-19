#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h> // Must be the first include.

#include <solvcon/python/common.hpp>
#include <solvcon/timeseries/timeseries.hpp>

namespace solvcon
{

namespace python
{

namespace detail
{

template <typename... Ts>
struct type_list
{
}; /* end struct type_list */

// clang-format off
/// The value types `deriv()` takes, common ones first because pybind11 tries overloads in order.
using timeseries_real_types = type_list<
    double, float, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t, uint16_t, uint8_t>;

/// The value types `dedup_last()` takes: every value type `SimpleArray` holds.
using timeseries_value_types = type_list<
    double, float, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t, uint16_t, uint8_t,
    bool, Complex<double>, Complex<float>>;
// clang-format on

/// Call `def.template operator()<T>()` for every `T` in the list.
template <typename... Ts, typename F>
void for_each_type(type_list<Ts...>, F const & def)
{
    (def.template operator()<Ts>(), ...);
}

} /* end namespace detail */

void initialize_timeseries(pybind11::module & mod);
void wrap_timeseries(pybind11::module & mod);

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
