/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/mcap/pymod/mcap_pymod.hpp>

namespace solvcon
{

namespace python
{

struct mcap_pymod_tag;

template <>
OneTimeInitializer<mcap_pymod_tag> & OneTimeInitializer<mcap_pymod_tag>::me()
{
    static OneTimeInitializer<mcap_pymod_tag> instance;
    return instance;
}

void initialize_mcap(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_McapReader(mod);
    };

    OneTimeInitializer<mcap_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
