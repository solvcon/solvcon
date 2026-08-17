/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/timeseries/pymod/timeseries_pymod.hpp>

namespace solvcon
{

namespace python
{

struct timeseries_pymod_tag;

template <>
OneTimeInitializer<timeseries_pymod_tag> & OneTimeInitializer<timeseries_pymod_tag>::me()
{
    static OneTimeInitializer<timeseries_pymod_tag> instance;
    return instance;
}

void initialize_timeseries(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_timeseries(mod);
    };

    OneTimeInitializer<timeseries_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
