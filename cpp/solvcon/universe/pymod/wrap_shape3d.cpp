/*
 * Copyright (c) 2023, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/universe/pymod/universe_pymod.hpp> // Must be the first include.

namespace solvcon
{

namespace python
{

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapBoundBox3d
    : public WrapBase<WrapBoundBox3d<T>, BoundBox3d<T>>
{
public:
    using base_type = WrapBase<WrapBoundBox3d<T>, BoundBox3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;
    using value_type = T;

    friend typename base_type::root_base_type;

protected:
    WrapBoundBox3d(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapBoundBox3d & wrap_management();
    WrapBoundBox3d & wrap_geometry();
}; /* end class WrapBoundBox3d */

template <typename T>
WrapBoundBox3d<T>::WrapBoundBox3d(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .wrap_management()
        .wrap_geometry()
        //
        ;
}

template <typename T>
WrapBoundBox3d<T> & WrapBoundBox3d<T>::wrap_management()
{
    namespace py = pybind11;

    (*this)
        .def(py::init<value_type, value_type, value_type, value_type, value_type, value_type>(),
             py::arg("min_x"),
             py::arg("min_y"),
             py::arg("min_z"),
             py::arg("max_x"),
             py::arg("max_y"),
             py::arg("max_z"))
        //
        ;

    return *this;
}

template <typename T>
WrapBoundBox3d<T> & WrapBoundBox3d<T>::wrap_geometry()
{
    namespace py = pybind11;

    // Bounding value.
    (*this)
        .def_property_readonly("min_x", &wrapped_type::min_x)
        .def_property_readonly("min_y", &wrapped_type::min_y)
        .def_property_readonly("min_z", &wrapped_type::min_z)
        .def_property_readonly("max_x", &wrapped_type::max_x)
        .def_property_readonly("max_y", &wrapped_type::max_y)
        .def_property_readonly("max_z", &wrapped_type::max_z)
        //
        ;

    // Calculation.
    (*this)
        .def("calc_area", &wrapped_type::calc_area)
        .def("overlap", &wrapped_type::overlap, py::arg("other"))
        .def("contain", &wrapped_type::contain, py::arg("other"))
        .def("expand", &wrapped_type::expand, py::arg("other"))
        //
        ;

    return *this;
}

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapOrientedBoundBox3d
    : public WrapBase<WrapOrientedBoundBox3d<T>, OrientedBoundBox3d<T>>
{
public:
    using base_type = WrapBase<WrapOrientedBoundBox3d<T>, OrientedBoundBox3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;
    using value_type = T;
    using point_type = typename wrapped_type::point_type;
    using bbox_type = typename wrapped_type::bbox_type;

    friend typename base_type::root_base_type;

protected:
    WrapOrientedBoundBox3d(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapOrientedBoundBox3d & wrap_management();
    WrapOrientedBoundBox3d & wrap_geometry();
}; /* end class WrapOrientedBoundBox3d */

template <typename T>
WrapOrientedBoundBox3d<T>::WrapOrientedBoundBox3d(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .wrap_management()
        .wrap_geometry()
        //
        ;
}

template <typename T>
WrapOrientedBoundBox3d<T> & WrapOrientedBoundBox3d<T>::wrap_management()
{
    namespace py = pybind11;

    (*this)
        .def(py::init<bbox_type const &>(), py::arg("aabb"))
        .def(py::init<point_type const &, value_type, value_type, value_type, value_type>(),
             py::arg("center"),
             py::arg("ux"),
             py::arg("uy"),
             py::arg("half_width"),
             py::arg("half_height"))
        //
        ;

    return *this;
}

template <typename T>
WrapOrientedBoundBox3d<T> & WrapOrientedBoundBox3d<T>::wrap_geometry()
{
    namespace py = pybind11;

    // Shape of the box.
    (*this)
        .def_property_readonly("center", &wrapped_type::center)
        .def_property_readonly("ux", &wrapped_type::ux)
        .def_property_readonly("uy", &wrapped_type::uy)
        .def_property_readonly("half_width", &wrapped_type::half_width)
        .def_property_readonly("half_height", &wrapped_type::half_height)
        .def_property_readonly("width", &wrapped_type::width)
        .def_property_readonly("height", &wrapped_type::height)
        .def("corner", &wrapped_type::corner, py::arg("i"))
        .def("corners", &wrapped_type::corners)
        .def("aabb", &wrapped_type::aabb)
        //
        ;

    // Motion.
    (*this)
        .def("translate", &wrapped_type::translate, py::arg("dx"), py::arg("dy"))
        .def("rotate", &wrapped_type::rotate, py::arg("angle"), py::arg("cx"), py::arg("cy"))
        //
        ;

    return *this;
}

void wrap_shape3d(pybind11::module & mod)
{
    WrapBoundBox3d<float>::commit(mod, "BoundBox3dFp32", "BoundBox3dFp32");
    WrapBoundBox3d<double>::commit(mod, "BoundBox3dFp64", "BoundBox3dFp64");
    WrapOrientedBoundBox3d<float>::commit(mod, "OrientedBoundBox3dFp32", "OrientedBoundBox3dFp32");
    WrapOrientedBoundBox3d<double>::commit(mod, "OrientedBoundBox3dFp64", "OrientedBoundBox3dFp64");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: