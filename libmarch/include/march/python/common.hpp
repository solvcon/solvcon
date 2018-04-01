#pragma once

/*
 * Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, std::unique_ptr<T>);
PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

#ifdef __GNUG__
#  define MARCH_PYTHON_WRAPPER_VISIBILITY __attribute__((visibility("hidden")))
#else
#  define MARCH_PYTHON_WRAPPER_VISIBILITY
#endif

namespace march {

namespace python {

namespace py = pybind11;

} /* end namespace python */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
