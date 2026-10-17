/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/math/pymod/math_pymod.hpp>

#include <solvcon/solvcon.hpp>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>

namespace solvcon
{

namespace python
{
template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapComplex
    : public WrapBase<WrapComplex<T>, solvcon::Complex<T>, std::shared_ptr<solvcon::Complex<T>>>
{
    using base_type = WrapBase<WrapComplex<T>, solvcon::Complex<T>, std::shared_ptr<solvcon::Complex<T>>>;
    using wrapped_type = typename base_type::wrapped_type;
    using value_type = T;

    friend base_type;

    WrapComplex(pybind11::module & mod, char const * pyname, char const * pydoc)
        : WrapBase<WrapComplex<value_type>, solvcon::Complex<value_type>, std::shared_ptr<solvcon::Complex<value_type>>>(mod, pyname, pydoc)
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)
        PYBIND11_NUMPY_DTYPE(wrapped_type, real_v, imag_v);

        (*this)
            .def(
                py::init(
                    []()
                    { return std::make_shared<wrapped_type>(wrapped_type{0.0, 0.0}); }))
            .def(
                py::init(
                    [](const value_type & real, const value_type & imag)
                    { return std::make_shared<wrapped_type>(wrapped_type{real, imag}); }),
                py::arg("real"),
                py::arg("imag"))
            .def(
                py::init(
                    [](const wrapped_type & other)
                    { return std::make_shared<wrapped_type>(other); }),
                py::arg("complex"))
            .def(
                py::init(
                    [](const std::complex<value_type> & c)
                    { return std::make_shared<wrapped_type>(wrapped_type{c.real(), c.imag()}); }),
                py::arg("np_complex"))
            .def(py::self + py::self) // NOLINT(misc-redundant-expression)
            .def(py::self + value_type()) // NOLINT(misc-redundant-expression)
            .def(value_type() + py::self) // NOLINT(misc-redundant-expression)
            .def(py::self - py::self) // NOLINT(misc-redundant-expression)
            .def(py::self - value_type()) // NOLINT(misc-redundant-expression)
            .def(value_type() - py::self) // NOLINT(misc-redundant-expression)
            .def(py::self * py::self) // NOLINT(misc-redundant-expression)
            .def(py::self * value_type()) // NOLINT(misc-redundant-expression)
            .def(value_type() * py::self) // NOLINT(misc-redundant-expression)
            .def(py::self / py::self) // NOLINT(misc-redundant-expression)
            .def(py::self / value_type()) // NOLINT(misc-redundant-expression)
            .def(value_type() / py::self) // NOLINT(misc-redundant-expression)
            .def(py::self += py::self) // NOLINT(misc-redundant-expression)
            .def(py::self += value_type()) // NOLINT(misc-redundant-expression)
            .def(py::self -= py::self) // NOLINT(misc-redundant-expression)
            .def(py::self -= value_type()) // NOLINT(misc-redundant-expression)
            .def(py::self *= py::self) // NOLINT(misc-redundant-expression)
            .def(py::self *= value_type()) // NOLINT(misc-redundant-expression)
            .def(py::self /= py::self) // NOLINT(misc-redundant-expression)
            .def(py::self /= value_type()) // NOLINT(misc-redundant-expression)
            .def(py::self == py::self) // NOLINT(misc-redundant-expression)
            .def(py::self != py::self) // NOLINT(misc-redundant-expression)
            .def(py::self < py::self) // NOLINT(misc-redundant-expression)
            .def(py::self > py::self) // NOLINT(misc-redundant-expression)
            .def_readonly("real", &wrapped_type::real_v)
            .def_readonly("imag", &wrapped_type::imag_v)
            .def("norm", &wrapped_type::norm)
            .def("conj", &wrapped_type::conj)
            .def("__complex__",
                 [](const wrapped_type & self)
                 {
                     return self.to_std_complex();
                 })
            .def("dtype",
                 []()
                 { return py::dtype(std::is_same_v<value_type, float> ? "complex64" : "complex128"); });
    }

}; /* end class WrapComplex */

void wrap_Complex(pybind11::module & mod)
{
    WrapComplex<float>::commit(mod, "complex64", "complex64");
    WrapComplex<double>::commit(mod, "complex128", "complex128");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
