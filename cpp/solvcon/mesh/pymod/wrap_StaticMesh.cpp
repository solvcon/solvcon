/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/mesh/pymod/mesh_pymod.hpp> // Must be the first include.
#include <solvcon/solvcon.hpp>

namespace solvcon
{

namespace python
{

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapStaticMeshBc
    : public WrapBase<WrapStaticMeshBc, StaticMeshBc, std::shared_ptr<StaticMeshBc>>
{

public:

    using base_type = WrapBase<WrapStaticMeshBc, StaticMeshBc, std::shared_ptr<StaticMeshBc>>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapStaticMeshBc(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapStaticMeshBc */

WrapStaticMeshBc::WrapStaticMeshBc(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    (*this)
        .def_property("name", &wrapped_type::name, &wrapped_type::set_name)
        .def_property_readonly("nbound", &wrapped_type::nbound)
        .expose_SimpleArray("facn", [](wrapped_type & self) -> decltype(auto)
                            { return self.facn(); });

    this->cls().attr("NONAME") = wrapped_type::NONAME();
}

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapStaticMesh
    : public WrapBase<WrapStaticMesh, StaticMesh, std::shared_ptr<StaticMesh>>
{

public:

    using base_type = WrapBase<WrapStaticMesh, StaticMesh, std::shared_ptr<StaticMesh>>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapStaticMesh(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapStaticMesh */

WrapStaticMesh::WrapStaticMesh(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    using int_type = typename wrapped_type::int_type;
    using uint_type = typename wrapped_type::uint_type;

    (*this)
        .def_timed(
            py::init(
                [](uint8_t ndim, uint_type nnode, uint_type nface, uint_type ncell)
                { return wrapped_type::construct(ndim, nnode, nface, ncell); }),
            py::arg("ndim"),
            py::arg("nnode"),
            py::arg("nface") = 0,
            py::arg("ncell") = 0)
        //
        ;

#define SC_DECL_STATIC(NAME) \
    .def_property_readonly_static(#NAME, [](py::object const &) { return wrapped_type::NAME; })

    // clang-format off
        (*this)
            SC_DECL_STATIC(FCMND)
            SC_DECL_STATIC(CLMND)
            SC_DECL_STATIC(CLMFC)
            SC_DECL_STATIC(FCREL)
            SC_DECL_STATIC(BFREL)
        ;
    // clang-format on

#undef SC_DECL_STATIC

    (*this)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_property_readonly("nnode", &wrapped_type::nnode)
        .def_property_readonly("nface", &wrapped_type::nface)
        .def_property_readonly("ncell", &wrapped_type::ncell)
        .def_property_readonly("nbound", &wrapped_type::nbound)
        .def_property_readonly("ngstnode", &wrapped_type::ngstnode)
        .def_property_readonly("ngstface", &wrapped_type::ngstface)
        .def_property_readonly("ngstcell", &wrapped_type::ngstcell)
        .def_property_readonly("nedge", &wrapped_type::nedge)
        .def_property_readonly("nbcs", &wrapped_type::nbcs);

    (*this)
        .def_timed("build_interior", &wrapped_type::build_interior, py::arg("do_metric") = true, py::arg("build_edge") = true)
        .def_timed("build_boundary", &wrapped_type::build_boundary)
        .def_timed("build_ghost", &wrapped_type::build_ghost)
        .def_timed("build_edge", &wrapped_type::build_edge);

    (*this)
        .def(
            "add_bc",
            [](wrapped_type & self, std::string const & name, std::vector<int_type> const & faces)
            { return self.add_bc(name, faces); },
            py::arg("name"),
            py::arg("faces"))
        .def(
            "bc",
            [](wrapped_type & self, size_t ibc)
            { return self.bc(ibc); },
            py::arg("ibc"))
        .def(
            "bc",
            [](wrapped_type & self, std::string const & name)
            {
                auto bnd = self.find_bc(name);
                if (!bnd)
                {
                    throw py::key_error(name);
                }
                return bnd;
            },
            py::arg("name"))
        .def_property_readonly("bcs", [](wrapped_type & self)
                               { return self.bcs(); });

#define SC_DECL_ARRAY(NAME) \
    .expose_SimpleArray(#NAME, [](wrapped_type & self) -> decltype(auto) { return self.NAME(); })

    // clang-format off
        (*this)
            SC_DECL_ARRAY(ndcrd)
            SC_DECL_ARRAY(fccnd)
            SC_DECL_ARRAY(fcnml)
            SC_DECL_ARRAY(fcara)
            SC_DECL_ARRAY(clcnd)
            SC_DECL_ARRAY(clvol)
            SC_DECL_ARRAY(fctpn)
            SC_DECL_ARRAY(cltpn)
            SC_DECL_ARRAY(clgrp)
            SC_DECL_ARRAY(fcnds)
            SC_DECL_ARRAY(fccls)
            SC_DECL_ARRAY(clnds)
            SC_DECL_ARRAY(clfcs)
            SC_DECL_ARRAY(ednds)
            SC_DECL_ARRAY(bndfcs)
        ;
    // clang-format on

#undef SC_DECL_ARRAY

    this->cls().attr("NONCELLTYPE") = static_cast<uint8_t>(CellType::NONCELLTYPE);
    this->cls().attr("POINT") = static_cast<uint8_t>(CellType::POINT);
    this->cls().attr("LINE") = static_cast<uint8_t>(CellType::LINE);
    this->cls().attr("QUADRILATERAL") = static_cast<uint8_t>(CellType::QUADRILATERAL);
    this->cls().attr("TRIANGLE") = static_cast<uint8_t>(CellType::TRIANGLE);
    this->cls().attr("HEXAHEDRON") = static_cast<uint8_t>(CellType::HEXAHEDRON);
    this->cls().attr("TETRAHEDRON") = static_cast<uint8_t>(CellType::TETRAHEDRON);
    this->cls().attr("PRISM") = static_cast<uint8_t>(CellType::PRISM);
    this->cls().attr("PYRAMID") = static_cast<uint8_t>(CellType::PYRAMID);
}

void wrap_StaticMesh(pybind11::module & mod)
{
    WrapStaticMeshBc::commit(mod, "StaticMeshBc", "StaticMeshBc");
    WrapStaticMesh::commit(mod, "StaticMesh", "StaticMesh");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
