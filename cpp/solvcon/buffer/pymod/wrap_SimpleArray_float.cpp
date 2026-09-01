/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/pymod/wrap_SimpleArray.hpp> // Must be the first include.

namespace solvcon
{

namespace python
{

void wrap_SimpleArray_float(pybind11::module & mod)
{
    WrapSimpleArray<Float16>::commit(mod, "SimpleArrayFloat16", "SimpleArrayFloat16");
    WrapSimpleArray<float>::commit(mod, "SimpleArrayFloat32", "SimpleArrayFloat32");
    WrapSimpleArray<double>::commit(mod, "SimpleArrayFloat64", "SimpleArrayFloat64");

    WrapSimpleCollector<Float16>::commit(mod, "SimpleCollectorFloat16", "SimpleCollectorFloat16");
    WrapSimpleCollector<float>::commit(mod, "SimpleCollectorFloat32", "SimpleCollectorFloat32");
    WrapSimpleCollector<double>::commit(mod, "SimpleCollectorFloat64", "SimpleCollectorFloat64");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
