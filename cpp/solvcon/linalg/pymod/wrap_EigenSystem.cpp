/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <memory>

#include <solvcon/linalg/pymod/linalg_pymod.hpp>

namespace solvcon
{

namespace python
{

#ifdef SC_HAS_VENDOR_LAPACK

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapEigenSystem
    : public WrapBase<WrapEigenSystem<T>, EigenSystem<T>, std::shared_ptr<EigenSystem<T>>>
{

    using base_type = WrapBase<WrapEigenSystem<T>, EigenSystem<T>, std::shared_ptr<EigenSystem<T>>>;
    using wrapped_type = typename base_type::wrapped_type;
    using array_type = SimpleArray<T>;

    friend base_type;

    WrapEigenSystem(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapEigenSystem */

template <typename T>
WrapEigenSystem<T>::WrapEigenSystem(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](array_type const & a, bool do_vl, bool do_vr)
                {
                    return wrapped_type::construct(a, do_vl, do_vr);
                }),
            py::arg("a"),
            py::arg("do_vl") = true,
            py::arg("do_vr") = true,
            // Keep the input array's Python wrapper alive while the
            // EigenSystem lives, so m_matrix's C++ reference stays
            // valid for the lifetime of this instance.
            py::keep_alive<1, 2>());

    // Eigenvalues are exposed as real/imaginary parts for every element type;
    // complex eigenvalues are split into wr/wi.
    (*this)
        .def_property_readonly("wr", &wrapped_type::wr)
        .def_property_readonly("wi", &wrapped_type::wi);

    (*this)
        .def("run", &wrapped_type::run)
        .def_property_readonly(
            "matrix",
            &wrapped_type::matrix,
            pybind11::return_value_policy::reference_internal)
        .def_property_readonly(
            "vl",
            [](wrapped_type const & self) -> array_type const &
            {
                return self.vl();
            })
        .def_property_readonly(
            "vr",
            [](wrapped_type const & self) -> array_type const &
            {
                return self.vr();
            })
        .def(
            "get_vl",
            &wrapped_type::vl,
            py::arg("suppress_exception") = false,
            "Left eigenvectors; suppress_exception=True returns empty "
            "instead of raising when do_vl=False.")
        .def(
            "get_vr",
            &wrapped_type::vr,
            py::arg("suppress_exception") = false,
            "Right eigenvectors; suppress_exception=True returns empty "
            "instead of raising when do_vr=False.")
        .def_property_readonly("do_vl", &wrapped_type::do_vl)
        .def_property_readonly("do_vr", &wrapped_type::do_vr)
        .def_property_readonly("done", &wrapped_type::done);
}

// Type-erased wrapper: constructs from a SimpleArrayPlex and dispatches on its
// runtime element type.  Unlike WrapEigenSystem<T>, it exposes wr/wi (real) and
// w (complex) together; the inapplicable ones raise at runtime.
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapEigenSystemPlex
    : public WrapBase<WrapEigenSystemPlex, EigenSystemPlex>
{

    using base_type = WrapBase<WrapEigenSystemPlex, EigenSystemPlex>;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

    WrapEigenSystemPlex(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapEigenSystemPlex */

WrapEigenSystemPlex::WrapEigenSystemPlex(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](SimpleArrayPlex const & a, bool do_vl, bool do_vr)
                {
                    return std::make_unique<wrapped_type>(a, do_vl, do_vr);
                }),
            py::arg("a"),
            py::arg("do_vl") = true,
            py::arg("do_vr") = true,
            // Keep the plex's Python wrapper alive while the EigenSystemPlex
            // lives; the dispatched solver references the array it owns.
            py::keep_alive<1, 2>());

    (*this)
        .def("run", &wrapped_type::run)
        .def_property_readonly(
            "matrix",
            &wrapped_type::matrix,
            pybind11::return_value_policy::reference_internal)
        .def_property_readonly("wr", &wrapped_type::wr)
        .def_property_readonly("wi", &wrapped_type::wi)
        .def_property_readonly(
            "vl",
            [](wrapped_type const & self)
            { return self.vl(); })
        .def_property_readonly(
            "vr",
            [](wrapped_type const & self)
            { return self.vr(); })
        .def(
            "get_vl",
            &wrapped_type::vl,
            py::arg("suppress_exception") = false,
            "Left eigenvectors; suppress_exception=True returns empty "
            "instead of raising when do_vl=False.")
        .def(
            "get_vr",
            &wrapped_type::vr,
            py::arg("suppress_exception") = false,
            "Right eigenvectors; suppress_exception=True returns empty "
            "instead of raising when do_vr=False.")
        .def_property_readonly("do_vl", &wrapped_type::do_vl)
        .def_property_readonly("do_vr", &wrapped_type::do_vr)
        .def_property_readonly("done", &wrapped_type::done);
}

#endif /* SC_HAS_VENDOR_LAPACK */

void wrap_EigenSystem(pybind11::module & mod)
{
#ifdef SC_HAS_VENDOR_LAPACK
    WrapEigenSystem<float>::commit(mod, "EigenSystemFloat32", "Eigen problem solver (float32)");
    WrapEigenSystem<double>::commit(mod, "EigenSystemFloat64", "Eigen problem solver (float64)");
    WrapEigenSystem<Complex<float>>::commit(mod, "EigenSystemComplex64", "Eigen problem solver (complex64)");
    WrapEigenSystem<Complex<double>>::commit(mod, "EigenSystemComplex128", "Eigen problem solver (complex128)");
    // EigenSystem is the type-erased, general entry point that infers the
    // element type from a SimpleArrayPlex (C++ class: EigenSystemPlex).
    WrapEigenSystemPlex::commit(mod, "EigenSystem", "Type-erased eigen problem solver");
#else // SC_HAS_VENDOR_LAPACK
    mod.attr("EigenSystemFloat32") = pybind11::none();
    mod.attr("EigenSystemFloat64") = pybind11::none();
    mod.attr("EigenSystemComplex64") = pybind11::none();
    mod.attr("EigenSystemComplex128") = pybind11::none();
    mod.attr("EigenSystem") = pybind11::none();
#endif // SC_HAS_VENDOR_LAPACK
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
