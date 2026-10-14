/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/mesh/pymod/mesh_pymod.hpp> // Must be the first include.
#include <solvcon/mesh/DelaunayBuilder.hpp>

namespace solvcon
{

namespace python
{

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapDelaunayBuilder
    : public WrapBase<WrapDelaunayBuilder, DelaunayBuilder>
{

public:

    using base_type = WrapBase<WrapDelaunayBuilder, DelaunayBuilder>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapDelaunayBuilder(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapDelaunayBuilder */

WrapDelaunayBuilder::WrapDelaunayBuilder(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(py::init<>())
        .def(py::init<std::shared_ptr<PointPad<double>>>(), py::arg("pad"))
        //
        ;

    (*this)
        .def_property("relative_tolerance", &wrapped_type::relative_tolerance, &wrapped_type::set_relative_tolerance)
        .def("__len__", [](wrapped_type const & self)
             { return self.size(); })
        .def("add", &wrapped_type::add, py::arg("ia"), py::arg("ib"), py::arg("ic"), py::arg("rr"), py::arg("ux"), py::arg("uy"), py::arg("uz") = 0.0)
        .def(
            "populate_from_points",
            &wrapped_type::populate_from_points,
            py::arg("pad"),
            py::arg("first"),
            py::arg("last"))
        .def("compute_triangle_nodes", &wrapped_type::compute_triangle_nodes, py::arg("pad"), py::arg("nnode"))
        .def_property_readonly("pad", &wrapped_type::pad)
        .def("triangulate", &wrapped_type::triangulate)
        .def("calc_ear_centroids", &wrapped_type::calc_ear_centroids, py::arg("tris"))
        //
        ;
}

void wrap_DelaunayBuilder(pybind11::module & mod)
{
    WrapDelaunayBuilder::commit(mod, "DelaunayBuilder", "DelaunayBuilder");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
