/*
 * Copyright (c) 2023, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/universe/pymod/universe_pymod.hpp> // Must be the first include.
#include <pybind11/operators.h>

namespace solvcon
{

namespace python
{

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapWorld
    : public WrapBase<WrapWorld<T>, World<T>, std::shared_ptr<World<T>>>
{

public:

    using base_type = WrapBase<WrapWorld<T>, World<T>, std::shared_ptr<World<T>>>;
    using wrapped_type = typename base_type::wrapped_type;

    using value_type = typename wrapped_type::value_type;
    using point_type = typename wrapped_type::point_type;
    using segment_type = typename wrapped_type::segment_type;
    using segment_pad_type = typename wrapped_type::segment_pad_type;
    using bezier_type = typename wrapped_type::bezier_type;
    using curve_pad_type = typename wrapped_type::curve_pad_type;

    friend typename base_type::root_base_type;

protected:

    WrapWorld(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapWorld & wrap_management();
    WrapWorld & wrap_point();
    WrapWorld & wrap_segment();
    WrapWorld & wrap_bezier();
    WrapWorld & wrap_shape();
}; /* end class WrapWorld */

template <typename T>
WrapWorld<T>::WrapWorld(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .wrap_management()
        .wrap_point()
        .wrap_segment()
        .wrap_bezier()
        .wrap_shape()
        //
        ;
}

template <typename T>
WrapWorld<T> & WrapWorld<T>::wrap_management()
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                []()
                { return wrapped_type::construct(); }))
        //
        ;

    (*this)
        .def_property_readonly("nshape", &wrapped_type::nshape)
        .def("undo", &wrapped_type::undo)
        .def("redo", &wrapped_type::redo)
        .def_property_readonly("can_undo", &wrapped_type::can_undo)
        .def_property_readonly("can_redo", &wrapped_type::can_redo)
        .def("begin_operation", &wrapped_type::begin_operation)
        .def("end_operation", &wrapped_type::end_operation)
        .def(
            "remove_shape",
            &wrapped_type::remove_shape,
            py::arg("shape_id"))
        .def(
            "shape_type_of",
            [](wrapped_type const & self, int32_t shape_id)
            {
                return shape_type_name(self.shape_type_of(shape_id));
            },
            py::arg("shape_id"))
        .def(
            "describe_state",
            [](wrapped_type const & self, std::string const & level)
            {
                return self.describe_state(describe_level_from_string(level));
            },
            py::arg("level") = "basic")
        .def("clear", &wrapped_type::clear)
        .def(
            "translate_shape",
            &wrapped_type::translate_shape,
            py::arg("shape_id"),
            py::arg("dx"),
            py::arg("dy"))
        .def(
            "rotate_shape",
            &wrapped_type::rotate_shape,
            py::arg("shape_id"),
            py::arg("angle"),
            py::arg("cx"),
            py::arg("cy"))
        .def(
            "shape_is_live",
            &wrapped_type::shape_is_live,
            py::arg("shape_id"))
        .def(
            "shape_bbox",
            [](wrapped_type const & self, int32_t shape_id)
            {
                auto const bb = self.shape_bbox(shape_id);
                return std::vector<value_type>(bb.begin(), bb.end());
            },
            py::arg("shape_id"))
        .def(
            "shape_handle",
            [](wrapped_type const & self, int32_t shape_id)
            {
                auto const h = self.shape_handle(shape_id);
                return std::vector<value_type>(h.begin(), h.end());
            },
            py::arg("shape_id"))
        .def(
            "shape_obb",
            [](wrapped_type const & self, int32_t shape_id)
            {
                auto const obb = self.shape_obb(shape_id);
                return std::vector<value_type>(obb.begin(), obb.end());
            },
            py::arg("shape_id"))
        .def(
            "pick_shape",
            &wrapped_type::pick_shape,
            py::arg("x"),
            py::arg("y"),
            py::arg("tol"))
        .def(
            "query_visible",
            &wrapped_type::query_visible,
            py::arg("min_x"),
            py::arg("min_y"),
            py::arg("max_x"),
            py::arg("max_y"))
        //
        ;

    return *this;
}

template <typename T>
WrapWorld<T> & WrapWorld<T>::wrap_point()
{
    namespace py = pybind11;

    // Point.
    (*this)
        .def(
            "add_point",
            [](wrapped_type & self, point_type const & point)
            {
                self.add_point(point);
                return self.point_at(self.npoint() - 1);
            },
            py::arg("point"))
        .def(
            "add_point",
            [](wrapped_type & self, value_type x, value_type y, value_type z)
            {
                self.add_point(x, y, z);
                return self.point_at(self.npoint() - 1);
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"))
        .def_property_readonly("npoint", &wrapped_type::npoint)
        .def("point", &wrapped_type::point_at)
        .def_property_readonly("points", &wrapped_type::points)
        //
        ;

    return *this;
}

template <typename T>
WrapWorld<T> & WrapWorld<T>::wrap_segment()
{
    namespace py = pybind11;

    // Segment.
    (*this)
        .def(
            "add_segment",
            [](wrapped_type & self, segment_type const & edge)
            {
                self.add_segment(edge);
                return self.segment_at(self.nsegment() - 1);
            },
            py::arg("s"))
        .def(
            "add_segment",
            [](wrapped_type & self, point_type const & p0, point_type const & p1)
            {
                self.add_segment(p0, p1);
                return self.segment_at(self.nsegment() - 1);
            },
            py::arg("p0"),
            py::arg("p1"))
        .def(
            "add_segments",
            [](wrapped_type & self, segment_pad_type const & segment_pad)
            {
                for (size_t i = 0; i < segment_pad.size(); ++i)
                {
                    self.add_segment(segment_pad.get(i));
                }
            },
            py::arg("pad"))
        .def_property_readonly("nsegment", &wrapped_type::nsegment)
        .def("segment", &wrapped_type::segment_at)
        .def_property_readonly("segments", &wrapped_type::segments)
        //
        ;

    return *this;
}

template <typename T>
WrapWorld<T> & WrapWorld<T>::wrap_bezier()
{
    namespace py = pybind11;

    // Bezier curves
    (*this)
        .def(
            "add_bezier",
            [](wrapped_type & self, bezier_type const & bezier)
            {
                self.add_bezier(bezier);
                return self.bezier_at(self.nbezier() - 1);
            },
            py::arg("b"))
        .def(
            "add_bezier",
            [](wrapped_type & self, point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
            {
                self.add_bezier(p0, p1, p2, p3);
                return self.bezier_at(self.nbezier() - 1);
            },
            py::arg("p0"),
            py::arg("p1"),
            py::arg("p2"),
            py::arg("p3"))
        .def(
            "add_beziers",
            [](wrapped_type & self, curve_pad_type const & curve_pad)
            {
                for (size_t i = 0; i < curve_pad.size(); ++i)
                {
                    self.add_bezier(curve_pad.get(i));
                }
            },
            py::arg("pad"))
        .def_property_readonly("nbezier", &wrapped_type::nbezier)
        .def(
            "bezier",
            [](wrapped_type & self, size_t i)
            {
                return self.bezier_at(i);
            })
        .def_property_readonly("curves", &wrapped_type::curves)
        //
        ;

    return *this;
}

template <typename T>
WrapWorld<T> & WrapWorld<T>::wrap_shape()
{
    namespace py = pybind11;

    (*this)
        .def(
            "add_triangle",
            [](wrapped_type & self, value_type x0, value_type y0, value_type x1, value_type y1, value_type x2, value_type y2)
            {
                return self.add_triangle(x0, y0, x1, y1, x2, y2);
            },
            py::arg("x0"),
            py::arg("y0"),
            py::arg("x1"),
            py::arg("y1"),
            py::arg("x2"),
            py::arg("y2"))
        .def(
            "add_line",
            [](wrapped_type & self, value_type x0, value_type y0, value_type x1, value_type y1)
            {
                return self.add_line(x0, y0, x1, y1);
            },
            py::arg("x0"),
            py::arg("y0"),
            py::arg("x1"),
            py::arg("y1"))
        .def(
            "add_rectangle",
            [](wrapped_type & self, value_type x_min, value_type y_min, value_type x_max, value_type y_max)
            {
                return self.add_rectangle(x_min, y_min, x_max, y_max);
            },
            py::arg("x_min"),
            py::arg("y_min"),
            py::arg("x_max"),
            py::arg("y_max"))
        .def(
            "add_square",
            [](wrapped_type & self, value_type x_min, value_type y_min, value_type size)
            {
                return self.add_square(x_min, y_min, size);
            },
            py::arg("x_min"),
            py::arg("y_min"),
            py::arg("size"))
        .def(
            "add_ellipse",
            [](wrapped_type & self, value_type cx, value_type cy, value_type rx, value_type ry)
            {
                return self.add_ellipse(cx, cy, rx, ry);
            },
            py::arg("cx"),
            py::arg("cy"),
            py::arg("rx"),
            py::arg("ry"))
        .def(
            "add_circle",
            [](wrapped_type & self, value_type cx, value_type cy, value_type r)
            {
                return self.add_circle(cx, cy, r);
            },
            py::arg("cx"),
            py::arg("cy"),
            py::arg("r"))
        .def(
            "add_bezier_shape",
            [](wrapped_type & self, point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
            {
                return self.add_bezier_shape(p0, p1, p2, p3);
            },
            py::arg("p0"),
            py::arg("p1"),
            py::arg("p2"),
            py::arg("p3"))
        .def(
            "add_bezier_shape",
            [](wrapped_type & self, bezier_type const & bezier)
            {
                return self.add_bezier_shape(bezier);
            },
            py::arg("b"))
        //
        ;

    return *this;
}

void wrap_World(pybind11::module & mod)
{
    WrapWorld<float>::commit(mod, "WorldFp32", "WorldFp32");
    WrapWorld<double>::commit(mod, "WorldFp64", "WorldFp64");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
