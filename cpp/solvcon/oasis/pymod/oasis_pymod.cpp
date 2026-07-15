/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/oasis/pymod/oasis_pymod.hpp> // Must be the first include.

#include <solvcon/solvcon.hpp>
#include <solvcon/python/common.hpp>

namespace solvcon
{

namespace python
{

struct oasis_pymod_tag
{
}; /* end struct oasis_pymod_tag */

template <>
OneTimeInitializer<oasis_pymod_tag> & OneTimeInitializer<oasis_pymod_tag>::me()
{
    static OneTimeInitializer<oasis_pymod_tag> instance;
    return instance;
}

void initialize_oasis(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_oasis_device(mod);
    };

    OneTimeInitializer<oasis_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
