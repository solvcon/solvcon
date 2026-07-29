#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/python/python.hpp> // Must be the first include.

namespace solvcon
{

namespace python
{

void initialize_pilot(pybind11::module & mod);
void wrap_pilot(pybind11::module & mod);
void wrap_plot(pybind11::module & mod);

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
