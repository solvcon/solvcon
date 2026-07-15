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

template <typename Wrapper, typename GT>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapStaticGridBase
    : public WrapBase<Wrapper, GT>
{

public:

    using base_type = WrapBase<Wrapper, GT>;
    using wrapped_type = typename base_type::wrapped_type;

    using serial_type = typename wrapped_type::serial_type;
    using real_type = typename wrapped_type::real_type;

    friend typename base_type::root_base_type;

protected:

    WrapStaticGridBase(pybind11::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def_property_readonly_static(
                "NDIM",
                [](py::object const &)
                { return wrapped_type::NDIM; })
            //
            ;
    }

}; /* end class WrapStaticGridBase */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapStaticGrid1d
    : public WrapStaticGridBase<WrapStaticGrid1d, StaticGrid1d>
{

public:

    friend root_base_type;

    using base_type = WrapStaticGridBase<WrapStaticGrid1d, StaticGrid1d>;

protected:

    WrapStaticGrid1d(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapStaticGrid1d */

WrapStaticGrid1d::WrapStaticGrid1d(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def_timed(
            py::init(
                [](serial_type nx)
                { return std::make_unique<StaticGrid1d>(nx); }),
            py::arg("nx"))
        .def(
            "__len__",
            [](wrapped_type const & self)
            { return self.size(); })
        .def(
            "__getitem__",
            [](wrapped_type const & self, size_t it)
            { return self.at(it); })
        .def(
            "__setitem__",
            [](wrapped_type & self, size_t it, wrapped_type::real_type val)
            { self.at(it) = val; })
        .def_property_readonly(
            "nx",
            [](wrapped_type const & self)
            { return self.nx(); })
        .expose_SimpleArray(
            "coord",
            [](wrapped_type & self) -> decltype(auto)
            { return self.coord(); })
        .def_timed("fill", &wrapped_type::fill, py::arg("value"))
        //
        ;
}

// clang-format off
#define MM_DECL_StaticGridMD(NDIM) \
class \
SOLVCON_PYTHON_WRAPPER_VISIBILITY \
WrapStaticGrid ## NDIM ## d \
  : public WrapStaticGridBase< WrapStaticGrid ## NDIM ## d, StaticGrid ## NDIM ## d > \
{ \
\
public: \
\
      friend root_base_type; \
\
    using base_type = WrapStaticGridBase< WrapStaticGrid ## NDIM ## d, StaticGrid ## NDIM ## d >; \
\
protected: \
\
    explicit WrapStaticGrid ## NDIM ## d(pybind11::module & mod, char const * pyname, char const * pydoc) \
      : base_type(mod, pyname, pydoc) \
    {} \
\
} /* end class WrapStaticGrid##NDIM##d */
// clang-format on

MM_DECL_StaticGridMD(2);
MM_DECL_StaticGridMD(3);

#undef MM_DECL_StaticGridMD

void wrap_StaticGrid(pybind11::module & mod)
{
    WrapStaticGrid1d::commit(mod, "StaticGrid1d", "StaticGrid1d");
    WrapStaticGrid2d::commit(mod, "StaticGrid2d", "StaticGrid2d");
    WrapStaticGrid3d::commit(mod, "StaticGrid3d", "StaticGrid3d");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
