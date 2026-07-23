/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/linalg/pymod/linalg_pymod.hpp>

namespace solvcon
{

namespace python
{

template <typename T>
auto lu_det_for_python(SimpleArray<T> const & a)
{
    if constexpr (is_complex_v<T>)
    {
        // Complex<T> has no pybind11 caster.
        return lu_det(a).to_std_complex();
    }
    else
    {
        return lu_det(a);
    }
}

// Attach .solve() and .inv() methods to an already-registered SimpleArray
// class.  The buffer module (where SimpleArray is defined) intentionally has
// no dependency on linalg/, so LU-based methods are injected here from the
// linalg module after the class has been registered.
template <typename T>
void add_simple_array_lu_methods(pybind11::module & mod, char const * pyname)
{
    namespace py = pybind11;
    py::object cls = mod.attr(pyname); // NOLINT(misc-const-correctness)
    cls.attr("solve") = py::cpp_function(
        [](SimpleArray<T> const & self, SimpleArray<T> const & b)
        { return lu_solve(self, b); },
        py::arg("b"),
        py::is_method(cls),
        "Solve linear system self @ x = b, returns x");
    cls.attr("inv") = py::cpp_function(
        [](SimpleArray<T> const & self)
        { return lu_inv(self); },
        py::is_method(cls),
        "Compute matrix inverse");
    cls.attr("det") = py::cpp_function(
        [](SimpleArray<T> const & self)
        { return lu_det_for_python(self); },
        py::is_method(cls),
        "Compute matrix determinant");
}

void wrap_factorization(pybind11::module & mod)
{
    // clang-format off
#define MM_DECL_LADEF(T)                                                     \
    mod.def(                                                                 \
        "llt_factorization",                                                 \
        [](SimpleArray<T> const & a)                                         \
        { return llt_factorization(a); },                                    \
        pybind11::arg("a"));                                                 \
    mod.def(                                                                 \
        "llt_solve",                                                         \
        [](SimpleArray<T> const & a, SimpleArray<T> const & b)               \
        { return llt_solve(a, b); },                                         \
        pybind11::arg("a"),                                                  \
        pybind11::arg("b"));                                                 \
    mod.def(                                                                 \
        "lu_factorization",                                                  \
        [](SimpleArray<T> const & a)                                         \
        {                                                                    \
            auto [lu, piv] = lu_factorization(a);                            \
            return pybind11::make_tuple(lu, piv);                            \
        },                                                                   \
        pybind11::arg("a"));                                                 \
    mod.def(                                                                 \
        "lu_solve",                                                          \
        [](SimpleArray<T> const & a, SimpleArray<T> const & b)               \
        { return lu_solve(a, b); },                                          \
        pybind11::arg("a"),                                                  \
        pybind11::arg("b"));                                                 \
    mod.def(                                                                 \
        "lu_inv",                                                            \
        [](SimpleArray<T> const & a)                                         \
        { return lu_inv(a); },                                               \
        pybind11::arg("a"));                                                 \
    mod.def(                                                                 \
        "lu_det",                                                            \
        [](SimpleArray<T> const & a)                                         \
        { return lu_det_for_python(a); },                                    \
        pybind11::arg("a"));
    // clang-format on

    MM_DECL_LADEF(float)
    MM_DECL_LADEF(double)
    MM_DECL_LADEF(Complex<float>)
    MM_DECL_LADEF(Complex<double>)

#undef MM_DECL_LADEF

    // Integer arrays are deliberately excluded: LU decomposition involves
    // division, which would silently truncate under integer arithmetic.
    add_simple_array_lu_methods<float>(mod, "SimpleArrayFloat32");
    add_simple_array_lu_methods<double>(mod, "SimpleArrayFloat64");
    add_simple_array_lu_methods<Complex<float>>(mod, "SimpleArrayComplex64");
    add_simple_array_lu_methods<Complex<double>>(mod, "SimpleArrayComplex128");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
