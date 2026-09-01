/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/pymod/wrap_SimpleArray.hpp> // Must be the first include.

namespace solvcon
{

namespace python
{

void wrap_SimpleArray(pybind11::module & mod)
{
    pybind11::register_exception<MatmulKernelUnavailable>(mod, "MatmulKernelUnavailable", PyExc_ValueError);

    wrap_SimpleArray_bool(mod);
    wrap_SimpleArray_int(mod);
    wrap_SimpleArray_uint(mod);
    wrap_SimpleArray_float(mod);
    wrap_SimpleArray_complex(mod);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
