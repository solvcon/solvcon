/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/onedim/pymod/onedim_pymod.hpp> // Must be the first include.
#include <pybind11/stl.h>

#include <solvcon/solvcon.hpp>
#include <solvcon/python/common.hpp>

namespace solvcon
{

namespace python
{

struct onedim_pymod_tag
{
}; /* end struct onedim_pymod_tag */

template <>
OneTimeInitializer<onedim_pymod_tag> & OneTimeInitializer<onedim_pymod_tag>::me()
{
    static OneTimeInitializer<onedim_pymod_tag> instance;
    return instance;
}

void initialize_onedim(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_onedim(mod);
    };

    OneTimeInitializer<onedim_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
