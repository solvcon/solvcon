/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/universe/pymod/universe_pymod.hpp>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace solvcon
{

namespace python
{

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapTriangle3d
    : public WrapBase<WrapTriangle3d<T>, Triangle3d<T>>
{

public:

    using base_type = WrapBase<WrapTriangle3d<T>, Triangle3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;

    using point_type = typename wrapped_type::point_type;
    using value_type = typename wrapped_type::value_type;

    friend typename base_type::root_base_type;

protected:

    WrapTriangle3d(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapTriangle3d & wrap_management();
    WrapTriangle3d & wrap_operator();
    WrapTriangle3d & wrap_accessor();
    WrapTriangle3d & wrap_geometry();
}; /* end class WrapTriangle3d */

template <typename T>
WrapTriangle3d<T>::WrapTriangle3d(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .wrap_management()
        .wrap_operator()
        .wrap_accessor()
        .wrap_geometry()
        //
        ;
}

template <typename T>
WrapTriangle3d<T> & WrapTriangle3d<T>::wrap_management()
{
    namespace py = pybind11;

    (*this)
        .def(py::init<point_type const &, point_type const &, point_type const &>(),
             py::arg("p0"),
             py::arg("p1"),
             py::arg("p2"))
        .def(
            "__repr__",
            [](wrapped_type const & self)
            {
                static char const * ttypename = std::is_same_v<T, double> ? "Triangle3dFp64" : "Triangle3dFp32";
                static char const * ptypename = std::is_same_v<T, double> ? "Point3dFp64" : "Point3dFp32";
                return std::format("{}({}({}), {}({}), {}({}))",
                                   ttypename,
                                   ptypename,
                                   self.p0().value_string(),
                                   ptypename,
                                   self.p1().value_string(),
                                   ptypename,
                                   self.p2().value_string());
            })
        .def_alias("__repr__", "__str__")
        //
        ;

    return *this;
}

template <typename T>
WrapTriangle3d<T> & WrapTriangle3d<T>::wrap_operator()
{
    namespace py = pybind11;

    (*this)
        .def(py::self == py::self) // NOLINT(misc-redundant-expression)
        .def(py::self != py::self) // NOLINT(misc-redundant-expression)
        //
        ;

    return *this;
}

template <typename T>
WrapTriangle3d<T> & WrapTriangle3d<T>::wrap_accessor()
{
    namespace py = pybind11;

    (*this)
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
            [](wrapped_type & self, size_t it, point_type const & vec)
            { self.at(it) = vec; })
        //
        ;

#define DECL_WRAP(NAME)                               \
    .def_property(                                    \
        #NAME,                                        \
        [](wrapped_type const & self)                 \
        { return self.NAME(); },                      \
        [](wrapped_type & self, point_type const & v) \
        { self.set_##NAME(v); })
    // clang-format off
    (*this)
        DECL_WRAP(p0)
        DECL_WRAP(p1)
        DECL_WRAP(p2);
    // clang-format on
#undef DECL_WRAP

#define DECL_WRAP(NAME)                                              \
    .def_property(                                                   \
        #NAME,                                                       \
        [](wrapped_type const & self)                                \
        { return self.NAME(); },                                     \
        [](wrapped_type & self, typename wrapped_type::value_type v) \
        { self.set_##NAME(v); })
    // clang-format off
    (*this)
        DECL_WRAP(x0)
        DECL_WRAP(y0)
        DECL_WRAP(z0)
        DECL_WRAP(x1)
        DECL_WRAP(y1)
        DECL_WRAP(z1)
        DECL_WRAP(x2)
        DECL_WRAP(y2)
        DECL_WRAP(z2);
    // clang-format on
#undef DECL_WRAP

    return *this;
}

template <typename T>
WrapTriangle3d<T> & WrapTriangle3d<T>::wrap_geometry()
{
    namespace py = pybind11;

    (*this)
        .def(
            "mirror",
            [](wrapped_type & self, std::string const & axis)
            {
                if (axis == "x" || axis == "X")
                {
                    self.mirror(Axis::X);
                }
                else if (axis == "y" || axis == "Y")
                {
                    self.mirror(Axis::Y);
                }
                else if (axis == "z" || axis == "Z")
                {
                    self.mirror(Axis::Z);
                }
                else
                {
                    throw std::invalid_argument("Triangle3d::mirror: axis must be 'x', 'y', or 'z'");
                }
            },
            py::arg("axis"))
        //
        ;

    return *this;
}

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapTrianglePad
    : public WrapBase<WrapTrianglePad<T>, TrianglePad<T>, std::shared_ptr<TrianglePad<T>>>
{

public:

    using base_type = WrapBase<WrapTrianglePad<T>, TrianglePad<T>, std::shared_ptr<TrianglePad<T>>>;
    using wrapped_type = typename base_type::wrapped_type;

    using value_type = typename base_type::wrapped_type::value_type;
    using point_type = typename base_type::wrapped_type::point_type;
    using triangle_type = typename base_type::wrapped_type::triangle_type;
    using point_pad_type = typename base_type::wrapped_type::point_pad_type;

    friend typename base_type::root_base_type;

protected:

    WrapTrianglePad(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapTrianglePad & wrap_management();
    WrapTrianglePad & wrap_accessor_triangle();
    WrapTrianglePad & wrap_accessor_point();
    WrapTrianglePad & wrap_geometry();
}; /* end class WrapTrianglePad */

template <typename T>
WrapTrianglePad<T>::WrapTrianglePad(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .wrap_management()
        .wrap_accessor_triangle()
        .wrap_accessor_point()
        .wrap_geometry()
        //
        ;
}

template <typename T>
WrapTrianglePad<T> & WrapTrianglePad<T>::wrap_management()
{
    namespace py = pybind11;

    // Constructors
    (*this)
        .def(
            py::init(
                [](uint8_t ndim)
                { return wrapped_type::construct(ndim); }),
            py::arg("ndim"))
        .def(
            py::init(
                [](uint8_t ndim, size_t nelem)
                { return wrapped_type::construct(ndim, nelem); }),
            py::arg("ndim"),
            py::arg("nelem"))
        .def(
            py::init(
                [](
                    SimpleArray<T> & x0,
                    SimpleArray<T> & y0,
                    SimpleArray<T> & x1,
                    SimpleArray<T> & y1,
                    SimpleArray<T> & x2,
                    SimpleArray<T> & y2,
                    bool clone)
                { return wrapped_type::construct(x0, y0, x1, y1, x2, y2, clone); }),
            py::arg("x0"),
            py::arg("y0"),
            py::arg("x1"),
            py::arg("y1"),
            py::arg("x2"),
            py::arg("y2"),
            py::arg("clone"))
        .def(
            py::init(
                [](
                    SimpleArray<T> & x0,
                    SimpleArray<T> & y0,
                    SimpleArray<T> & z0,
                    SimpleArray<T> & x1,
                    SimpleArray<T> & y1,
                    SimpleArray<T> & z1,
                    SimpleArray<T> & x2,
                    SimpleArray<T> & y2,
                    SimpleArray<T> & z2,
                    bool clone)
                { return wrapped_type::construct(x0, y0, z0, x1, y1, z1, x2, y2, z2, clone); }),
            py::arg("x0"),
            py::arg("y0"),
            py::arg("z0"),
            py::arg("x1"),
            py::arg("y1"),
            py::arg("z1"),
            py::arg("x2"),
            py::arg("y2"),
            py::arg("z2"),
            py::arg("clone"))
        //
        ;

    (*this)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_timed("clone", &wrapped_type::clone)
        .def_timed("pack_array", &wrapped_type::pack_array)
        .def_timed("expand", &wrapped_type::expand, py::arg("length"))
        //
        ;

    return *this;
}

template <typename T>
WrapTrianglePad<T> & WrapTrianglePad<T>::wrap_accessor_triangle()
{
    namespace py = pybind11;

    // Python dunder accessor.
    (*this)
        .def("__len__", &wrapped_type::size)
        .def("__getitem__",
             [](wrapped_type const & self, size_t it)
             {
                 return self.get_at(it);
             })
        //
        ;

    // Append Triangle3d.
    (*this)
        .def(
            "append",
            [](wrapped_type & self, triangle_type const & t)
            {
                self.append(t);
            },
            py::arg("triangle"))
        .def(
            "append",
            [](wrapped_type & self, point_type const & p0, point_type const & p1, point_type const & p2)
            {
                self.append(p0, p1, p2);
            },
            py::arg("p0"),
            py::arg("p1"),
            py::arg("p2"))
        .def(
            "append",
            [](wrapped_type & self, value_type x0, value_type y0, value_type x1, value_type y1, value_type x2, value_type y2)
            {
                self.append(x0, y0, x1, y1, x2, y2);
            },
            py::arg("x0"),
            py::arg("y0"),
            py::arg("x1"),
            py::arg("y1"),
            py::arg("x2"),
            py::arg("y2"))
        .def(
            "append",
            [](wrapped_type & self, value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1, value_type x2, value_type y2, value_type z2)
            {
                self.append(x0, y0, z0, x1, y1, z1, x2, y2, z2);
            },
            py::arg("x0"),
            py::arg("y0"),
            py::arg("z0"),
            py::arg("x1"),
            py::arg("y1"),
            py::arg("z1"),
            py::arg("x2"),
            py::arg("y2"),
            py::arg("z2"))
        //
        ;

    // Extend Triangle3d
    (*this)
        .def(
            "extend_with",
            [](wrapped_type & self, wrapped_type const & other)
            {
                self.extend_with(other);
            },
            py::arg("triangles"))
        //
        ;

    // Additional getter and setter for Triangle3d.
    (*this)
        .def("get_at",
             [](wrapped_type const & self, size_t it)
             {
                 return self.get_at(it);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, triangle_type const & t)
             {
                 self.set_at(it, t);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, point_type const & p0, point_type const & p1, point_type const & p2)
             {
                 self.set_at(it, p0, p1, p2);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, value_type x0, value_type y0, value_type x1, value_type y1, value_type x2, value_type y2)
             {
                 self.set_at(it, x0, y0, x1, y1, x2, y2);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1, value_type x2, value_type y2, value_type z2)
             {
                 self.set_at(it, x0, y0, z0, x1, y1, z1, x2, y2, z2);
             })
        //
        ;

    return *this;
}

template <typename T>
WrapTrianglePad<T> & WrapTrianglePad<T>::wrap_accessor_point()
{
    namespace py = pybind11;

    // x, y, z, point element accessors.
#define DECL_WRAP(NAME)                         \
    .def(                                       \
        #NAME,                                  \
        [](wrapped_type const & self, size_t i) \
        { return self.NAME(i); })
    // clang-format off
    (*this)
        DECL_WRAP(x0_at)
        DECL_WRAP(y0_at)
        DECL_WRAP(z0_at)
        DECL_WRAP(x1_at)
        DECL_WRAP(y1_at)
        DECL_WRAP(z1_at)
        DECL_WRAP(x2_at)
        DECL_WRAP(y2_at)
        DECL_WRAP(z2_at)
        DECL_WRAP(p0_at)
        DECL_WRAP(p1_at)
        DECL_WRAP(p2_at);
    // clang-format on
#undef DECL_WRAP

    // x, y, z, point array/batch accessors.
#define DECL_WRAP(NAME)         \
    .def_property_readonly(     \
        #NAME,                  \
        [](wrapped_type & self) \
        { return self.NAME(); })
    // clang-format off
    (*this)
        DECL_WRAP(x0)
        DECL_WRAP(y0)
        DECL_WRAP(z0)
        DECL_WRAP(x1)
        DECL_WRAP(y1)
        DECL_WRAP(z1)
        DECL_WRAP(x2)
        DECL_WRAP(y2)
        DECL_WRAP(z2)
        DECL_WRAP(p0)
        DECL_WRAP(p1)
        DECL_WRAP(p2);
    // clang-format on
#undef DECL_WRAP

    return *this;
}

template <typename T>
WrapTrianglePad<T> & WrapTrianglePad<T>::wrap_geometry()
{
    namespace py = pybind11;

    (*this)
        .def(
            "mirror",
            [](wrapped_type & self, std::string const & axis)
            {
                if (axis == "x" || axis == "X")
                {
                    self.mirror(Axis::X);
                }
                else if (axis == "y" || axis == "Y")
                {
                    self.mirror(Axis::Y);
                }
                else if (axis == "z" || axis == "Z")
                {
                    self.mirror(Axis::Z);
                }
                else
                {
                    throw std::invalid_argument("TrianglePad::mirror: axis must be 'x', 'y', or 'z'");
                }
            },
            py::arg("axis"))
        //
        ;

    return *this;
}

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapPolygon
    : public WrapBase<WrapPolygon<T>, Polygon3d<T>>
{
public:
    using base_type = WrapBase<WrapPolygon<T>, Polygon3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;
    using value_type = typename wrapped_type::value_type;
    using point_type = typename wrapped_type::point_type;
    using segment_type = typename wrapped_type::segment_type;
    using polygon_pad_type = typename wrapped_type::polygon_pad_type;

    friend typename base_type::root_base_type;

protected:
    WrapPolygon(pybind11::module & mod, char const * pyname, char const * pydoc);
}; /* end class WrapPolygon */

template <typename T>
WrapPolygon<T>::WrapPolygon(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def_property_readonly("polygon_id", &wrapped_type::polygon_id)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_property_readonly("nnode", &wrapped_type::nnode)
        .def("get_node", &wrapped_type::node, py::arg("index"))
        .def("get_edge", &wrapped_type::edge, py::arg("index"))
        .def("compute_signed_area", &wrapped_type::compute_signed_area)
        .def("is_counter_clockwise", &wrapped_type::is_counter_clockwise)
        .def("calc_bound_box", &wrapped_type::calc_bound_box)
        .def(py::self == py::self) // NOLINT(misc-redundant-expression)
        .def(py::self != py::self) // NOLINT(misc-redundant-expression)
        .def("is_same", &wrapped_type::is, py::arg("other"))
        .def("is_not_same", &wrapped_type::is_not, py::arg("other"));
}

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapPolygonPad
    : public WrapBase<WrapPolygonPad<T>, PolygonPad<T>, std::shared_ptr<PolygonPad<T>>>
{
public:
    using base_type = WrapBase<WrapPolygonPad<T>, PolygonPad<T>, std::shared_ptr<PolygonPad<T>>>;
    using wrapped_type = typename base_type::wrapped_type;
    using value_type = typename wrapped_type::value_type;
    using point_type = typename wrapped_type::point_type;
    using segment_pad_type = typename wrapped_type::segment_pad_type;
    using curve_pad_type = typename wrapped_type::curve_pad_type;
    using polygon_type = typename wrapped_type::polygon_type;

    friend typename base_type::root_base_type;

protected:
    WrapPolygonPad(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapPolygonPad & wrap_management();
    WrapPolygonPad & wrap_accessor();
    WrapPolygonPad & wrap_search();
}; /* end class WrapPolygonPad */

template <typename T>
WrapPolygonPad<T>::WrapPolygonPad(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .wrap_management()
        .wrap_accessor()
        .wrap_search()
        //
        ;
}

template <typename T>
WrapPolygonPad<T> & WrapPolygonPad<T>::wrap_management()
{
    namespace py = pybind11;

    // Constructors.
    (*this)
        .def(
            py::init(
                [](uint8_t ndim)
                { return wrapped_type::construct(ndim); }),
            py::arg("ndim"))
        //
        ;

    (*this)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_property_readonly("num_polygons", &wrapped_type::num_polygons)
        .def_property_readonly("num_points", &wrapped_type::num_points)
        //
        ;

    return *this;
}

template <typename T>
WrapPolygonPad<T> & WrapPolygonPad<T>::wrap_accessor()
{
    namespace py = pybind11;

    (*this)
        .def(
            "add_polygon",
            [](wrapped_type & self, std::vector<point_type> const & nodes)
            {
                return self.add_polygon(nodes);
            },
            py::arg("nodes"))
        .def(
            "add_polygon_from_segments",
            &wrapped_type::add_polygon_from_segments,
            py::arg("segments"))
        .def(
            "add_polygon_from_curves",
            &wrapped_type::add_polygon_from_curves,
            py::arg("curves"),
            py::arg("sample_length"))
        .def(
            "add_polygon_from_segments_and_curves",
            &wrapped_type::add_polygon_from_segments_and_curves,
            py::arg("segments"),
            py::arg("curves"),
            py::arg("sample_length"))
        .def("get_polygon", &wrapped_type::get_polygon, py::arg("polygon_id"))
        //
        ;

    return *this;
}

template <typename T>
WrapPolygonPad<T> & WrapPolygonPad<T>::wrap_search()
{
    namespace py = pybind11;

    (*this)
        .def(
            "search_segments",
            [](wrapped_type const & self, BoundBox3d<T> const & box)
            {
                std::vector<Segment3d<T>> results;
                self.search_segments(box, results);
                return results;
            },
            py::arg("box"))
        .def("rebuild_rtree", &wrapped_type::rebuild_rtree)
        .def(
            "decompose_to_trapezoid",
            [](wrapped_type & self, size_t polygon_id)
            {
                return self.decompose_to_trapezoid(polygon_id);
            },
            py::arg("polygon_id"))
        .def(
            "boolean_union",
            &wrapped_type::boolean_union,
            py::arg("p1"),
            py::arg("p2"))
        .def(
            "boolean_intersection",
            &wrapped_type::boolean_intersection,
            py::arg("p1"),
            py::arg("p2"))
        .def(
            "boolean_difference",
            &wrapped_type::boolean_difference,
            py::arg("p1"),
            py::arg("p2"))
        .def(
            "decomposed_trapezoids",
            [](wrapped_type & self)
            {
                return self.decomposed_trapezoids();
            })
        //
        ;

    return *this;
}

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapTrapezoidPad
    : public WrapBase<WrapTrapezoidPad<T>, TrapezoidPad<T>, std::shared_ptr<TrapezoidPad<T>>>
{
public:
    using base_type = WrapBase<WrapTrapezoidPad<T>, TrapezoidPad<T>, std::shared_ptr<TrapezoidPad<T>>>;
    using wrapped_type = typename base_type::wrapped_type;
    using value_type = typename wrapped_type::value_type;
    using point_type = typename wrapped_type::point_type;

    friend typename base_type::root_base_type;

protected:
    WrapTrapezoidPad(pybind11::module & mod, char const * pyname, char const * pydoc);
}; /* end class WrapTrapezoidPad */

template <typename T>
WrapTrapezoidPad<T>::WrapTrapezoidPad(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](uint8_t ndim)
                {
                    return wrapped_type::construct(ndim);
                }),
            py::arg("ndim"))
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_property_readonly("size", &wrapped_type::size)
        .def("x0", py::overload_cast<size_t>(&wrapped_type::x0, py::const_), py::arg("index"))
        .def("y0", py::overload_cast<size_t>(&wrapped_type::y0, py::const_), py::arg("index"))
        .def("z0", py::overload_cast<size_t>(&wrapped_type::z0, py::const_), py::arg("index"))
        .def("x1", py::overload_cast<size_t>(&wrapped_type::x1, py::const_), py::arg("index"))
        .def("y1", py::overload_cast<size_t>(&wrapped_type::y1, py::const_), py::arg("index"))
        .def("z1", py::overload_cast<size_t>(&wrapped_type::z1, py::const_), py::arg("index"))
        .def("x2", py::overload_cast<size_t>(&wrapped_type::x2, py::const_), py::arg("index"))
        .def("y2", py::overload_cast<size_t>(&wrapped_type::y2, py::const_), py::arg("index"))
        .def("z2", py::overload_cast<size_t>(&wrapped_type::z2, py::const_), py::arg("index"))
        .def("x3", py::overload_cast<size_t>(&wrapped_type::x3, py::const_), py::arg("index"))
        .def("y3", py::overload_cast<size_t>(&wrapped_type::y3, py::const_), py::arg("index"))
        .def("z3", py::overload_cast<size_t>(&wrapped_type::z3, py::const_), py::arg("index"));
}

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapTrapezoidalDecomposer
    : public WrapBase<WrapTrapezoidalDecomposer<T>, TrapezoidalDecomposer<T>>
{
public:
    using base_type = WrapBase<WrapTrapezoidalDecomposer<T>, TrapezoidalDecomposer<T>>;
    using wrapped_type = typename base_type::wrapped_type;
    using value_type = typename wrapped_type::value_type;
    using point_type = typename wrapped_type::point_type;
    using trapezoid_pad_type = typename wrapped_type::trapezoid_pad_type;

    friend typename base_type::root_base_type;

protected:
    WrapTrapezoidalDecomposer(pybind11::module & mod, char const * pyname, char const * pydoc);
}; /* end class WrapTrapezoidalDecomposer */

template <typename T>
WrapTrapezoidalDecomposer<T>::WrapTrapezoidalDecomposer(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(py::init<uint8_t>(), py::arg("ndim"))
        .def(
            "decompose",
            [](wrapped_type & self, size_t polygon_id, std::vector<point_type> const & points)
            {
                return self.decompose(polygon_id, points);
            },
            py::arg("polygon_id"),
            py::arg("points"))
        .def("num_trapezoids", &wrapped_type::num_trapezoids, py::arg("polygon_id"))
        .def("trapezoids", py::overload_cast<>(&wrapped_type::trapezoids))
        .def("clear", &wrapped_type::clear);
}

void wrap_shape2d(pybind11::module & mod)
{
    WrapTriangle3d<float>::commit(mod, "Triangle3dFp32", "Triangle3dFp32");
    WrapTriangle3d<double>::commit(mod, "Triangle3dFp64", "Triangle3dFp64");
    WrapTrianglePad<float>::commit(mod, "TrianglePadFp32", "TrianglePadFp32");
    WrapTrianglePad<double>::commit(mod, "TrianglePadFp64", "TrianglePadFp64");

    WrapPolygon<float>::commit(mod, "Polygon3dFp32", "Polygon3dFp32");
    WrapPolygon<double>::commit(mod, "Polygon3dFp64", "Polygon3dFp64");

    WrapPolygonPad<float>::commit(mod, "PolygonPadFp32", "PolygonPadFp32");
    WrapPolygonPad<double>::commit(mod, "PolygonPadFp64", "PolygonPadFp64");

    WrapTrapezoidPad<float>::commit(mod, "TrapezoidPadFp32", "TrapezoidPadFp32");
    WrapTrapezoidPad<double>::commit(mod, "TrapezoidPadFp64", "TrapezoidPadFp64");

    WrapTrapezoidalDecomposer<float>::commit(mod, "TrapezoidalDecomposerFp32", "TrapezoidalDecomposerFp32");
    WrapTrapezoidalDecomposer<double>::commit(mod, "TrapezoidalDecomposerFp64", "TrapezoidalDecomposerFp64");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
