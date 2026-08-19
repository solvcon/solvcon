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

void initialize_timeseries(pybind11::module & mod);
void wrap_timeseries(pybind11::module & mod);

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
