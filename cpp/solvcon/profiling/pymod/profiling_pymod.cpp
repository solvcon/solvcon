/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/profiling/pymod/profiling_pymod.hpp> // Must be the first include.
#include <solvcon/buffer/pymod/buffer_pymod.hpp>

namespace solvcon
{

namespace python
{

struct profiling_pymod_tag;

template <>
OneTimeInitializer<profiling_pymod_tag> & OneTimeInitializer<profiling_pymod_tag>::me()
{
    static OneTimeInitializer<profiling_pymod_tag> instance;
    return instance;
}

void initialize_profiling(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_profile(mod);
    };

    OneTimeInitializer<profiling_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
