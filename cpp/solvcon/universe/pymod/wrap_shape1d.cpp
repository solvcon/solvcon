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

/// Map the lower-case axis name to the Axis enum; anything else throws.
static Axis parse_axis(std::string const & axis, char const * what)
{
    if (axis == "x")
    {
        return Axis::X;
    }
    if (axis == "y")
    {
        return Axis::Y;
    }
    if (axis == "z")
    {
        return Axis::Z;
    }
    throw std::invalid_argument(std::format("{}: axis must be 'x', 'y', or 'z'", what));
}

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapSegment3d
    : public WrapBase<WrapSegment3d<T>, Segment3d<T>>
{

public:

    using base_type = WrapBase<WrapSegment3d<T>, Segment3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;

    using point_type = typename wrapped_type::point_type;
    using value_type = typename wrapped_type::value_type;

    friend typename base_type::root_base_type;

protected:

    WrapSegment3d(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapSegment3d & wrap_management();
    WrapSegment3d & wrap_operator();
    WrapSegment3d & wrap_accessor();
    WrapSegment3d & wrap_geometry();
}; /* end class WrapSegment3d */

template <typename T>
WrapSegment3d<T>::WrapSegment3d(pybind11::module & mod, const char * pyname, const char * pydoc)
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
WrapSegment3d<T> & WrapSegment3d<T>::wrap_management()
{
    namespace py = pybind11;

    // Constructors.
    (*this)
        .def(py::init<point_type const &, point_type const &>(),
             py::arg("p0"),
             py::arg("p1"))
        .def(py::init<value_type, value_type, value_type, value_type>(),
             py::arg("x0"),
             py::arg("y0"),
             py::arg("x1"),
             py::arg("y1"))
        .def(py::init<value_type, value_type, value_type, value_type, value_type, value_type>(),
             py::arg("x0"),
             py::arg("y0"),
             py::arg("z0"),
             py::arg("x1"),
             py::arg("y1"),
             py::arg("z1"))
        //
        ;

    // Python string representations.
    (*this)
        .def(
            "__repr__",
            [](wrapped_type const & self)
            {
                // Hard-code the Python type names in the static variables before finding a systematic way.
                static char const * stypename = std::is_same_v<T, double> ? "Segment3dFp64" : "Segment3dFp32";
                static char const * ptypename = std::is_same_v<T, double> ? "Point3dFp64" : "Point3dFp32";
                return std::format("{}({}({}), {}({}))",
                                   stypename,
                                   ptypename,
                                   self.p0().value_string(),
                                   ptypename,
                                   self.p1().value_string());
            })
        .def_alias("__repr__", "__str__")
        //
        ;

    return *this;
}

template <typename T>
WrapSegment3d<T> & WrapSegment3d<T>::wrap_operator()
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
WrapSegment3d<T> & WrapSegment3d<T>::wrap_accessor()
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

    // p0 (tail) and p1 (head) accessors.
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
        ;
#undef DECL_WRAP
    // clang-format on

    // x, y, z accessors.
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
        ;
#undef DECL_WRAP
    // clang-format on

    return *this;
}

template <typename T>
WrapSegment3d<T> & WrapSegment3d<T>::wrap_geometry()
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
                    throw std::invalid_argument("Segment3d::mirror: axis must be 'x', 'y', or 'z'");
                }
            },
            py::arg("axis"))
        .def("calc_length2", &wrapped_type::calc_length2)
        .def("calc_length", &wrapped_type::calc_length)
        .def("direction", &wrapped_type::direction)
        .def(
            "normal_by_axis",
            [](wrapped_type const & self, std::string const & axis)
            { return self.normal_by_axis(parse_axis(axis, "Segment3d::normal_by_axis")); },
            py::arg("axis") = "z")
        .def(
            "offset_by_axis",
            [](wrapped_type const & self, point_type const & point, std::string const & axis)
            { return self.offset_by_axis(point, parse_axis(axis, "Segment3d::offset_by_axis")); },
            py::arg("point"),
            py::arg("axis") = "z")
        .def(
            "point_along_axis",
            [](wrapped_type const & self, value_type value, std::string const & axis)
            { return self.point_along_axis(value, parse_axis(axis, "Segment3d::point_along_axis")); },
            py::arg("value"),
            py::arg("axis") = "z")
        //
        ;

    return *this;
}

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapSegmentPad
    : public WrapBase<WrapSegmentPad<T>, SegmentPad<T>, std::shared_ptr<SegmentPad<T>>>
{

public:

    using base_type = WrapBase<WrapSegmentPad<T>, SegmentPad<T>, std::shared_ptr<SegmentPad<T>>>;
    using wrapped_type = typename base_type::wrapped_type;

    using value_type = typename base_type::wrapped_type::value_type;
    using point_type = typename base_type::wrapped_type::point_type;
    using segment_type = typename base_type::wrapped_type::segment_type;
    using point_pad_type = typename base_type::wrapped_type::point_pad_type;

    friend typename base_type::root_base_type;

protected:

    WrapSegmentPad(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapSegmentPad & wrap_management();
    WrapSegmentPad & wrap_accessor_segment();
    WrapSegmentPad & wrap_accessor_point();
    WrapSegmentPad & wrap_geometry();
}; /* end class WrapSegmentPad */

template <typename T>
WrapSegmentPad<T>::WrapSegmentPad(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .wrap_management()
        .wrap_accessor_segment()
        .wrap_accessor_point()
        .wrap_geometry()
        //
        ;
}

template <typename T>
WrapSegmentPad<T> & WrapSegmentPad<T>::wrap_management()
{
    namespace py = pybind11;

    // Constructors.
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
                    bool clone)
                { return wrapped_type::construct(x0, y0, x1, y1, clone); }),
            py::arg("x0"),
            py::arg("y0"),
            py::arg("x1"),
            py::arg("y1"),
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
                    bool clone)
                { return wrapped_type::construct(x0, y0, z0, x1, y1, z1, clone); }),
            py::arg("x0"),
            py::arg("y0"),
            py::arg("z0"),
            py::arg("x1"),
            py::arg("y1"),
            py::arg("z1"),
            py::arg("clone"))
        //
        ;

    (*this)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_timed("pack_array", &wrapped_type::pack_array)
        .def_timed("expand", &wrapped_type::expand, py::arg("length"))
        //
        ;

    return *this;
}

template <typename T>
WrapSegmentPad<T> & WrapSegmentPad<T>::wrap_accessor_segment()
{
    namespace py = pybind11;

    // Python dunder accessor.
    (*this)
        .def("__len__", &wrapped_type::size)
        .def(
            "__getitem__",
            [](wrapped_type const & self, ssize_t it)
            {
                return self.get_at(normalize_pad_index(it, self.size(), "SegmentPad"));
            },
            "Return the segment at the index. A negative index counts from the end of the pad.")
        .def(
            "__getitem__",
            [](wrapped_type const & self, py::slice const & key)
            {
                return copy_pad_slice(self, key);
            },
            "Return a new pad of the same ndim holding a copy of the selected segments. "
            "A write to the result does not reach the pad it was sliced from.")
        //
        ;

    // Append Segment3d.
    (*this)
        .def(
            "append",
            [](wrapped_type & self, segment_type const & s)
            {
                self.append(s);
            },
            py::arg("s"))
        .def(
            "append",
            [](wrapped_type & self, point_type const & p0, point_type const & p1)
            {
                self.append(p0, p1);
            },
            py::arg("p0"),
            py::arg("p1"))
        .def(
            "append",
            [](wrapped_type & self, value_type x0, value_type y0, value_type x1, value_type y1)
            {
                self.append(x0, y0, x1, y1);
            },
            py::arg("x0"),
            py::arg("y0"),
            py::arg("x1"),
            py::arg("y1"))
        .def(
            "append",
            [](wrapped_type & self, value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1)
            {
                self.append(x0, y0, z0, x1, y1, z1);
            },
            py::arg("x0"),
            py::arg("y0"),
            py::arg("z0"),
            py::arg("x1"),
            py::arg("y1"),
            py::arg("z1"))
        //
        ;

    // Extend Segment3d.
    (*this)
        .def(
            "extend_with",
            [](wrapped_type & self, wrapped_type const & other)
            {
                self.extend_with(other);
            },
            py::arg("segments"))
        //
        ;

    // Additional getter and setter for Segment3d.
    (*this)
        .def("get_at",
             [](wrapped_type const & self, size_t it)
             {
                 return self.get_at(it);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, segment_type const & s)
             {
                 self.set_at(it, s);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, point_type const & p0, point_type const & p1)
             {
                 self.set_at(it, p0, p1);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, value_type x0, value_type y0, value_type x1, value_type y1)
             {
                 self.set_at(it, x0, y0, x1, y1);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1)
             {
                 self.set_at(it, x0, y0, z0, x1, y1, z1);
             })
        //
        ;

    return *this;
}

template <typename T>
WrapSegmentPad<T> & WrapSegmentPad<T>::wrap_accessor_point()
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
        DECL_WRAP(p0_at)
        DECL_WRAP(p1_at)
        ;
#undef DECL_WRAP
    // clang-format on

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
        DECL_WRAP(p0)
        DECL_WRAP(p1)
        ;
#undef DECL_WRAP
    // clang-format on

    return *this;
}

template <typename T>
WrapSegmentPad<T> & WrapSegmentPad<T>::wrap_geometry()
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
                    throw std::invalid_argument("SegmentPad::mirror: axis must be 'x', 'y', or 'z'");
                }
            },
            py::arg("axis"))
        .def(
            "offset_by_axis",
            [](wrapped_type const & self,
               size_t index,
               SimpleArray<T> const & xs,
               SimpleArray<T> const & ys,
               std::string const & axis)
            { return self.offset_by_axis(index, xs, ys, parse_axis(axis, "SegmentPad::offset_by_axis")); },
            py::arg("index"),
            py::arg("xs"),
            py::arg("ys"),
            py::arg("axis") = "z")
        //
        ;

    return *this;
}

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapBezier3d
    : public WrapBase<WrapBezier3d<T>, Bezier3d<T>>
{

public:

    using base_type = WrapBase<WrapBezier3d<T>, Bezier3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;

    using point_type = typename wrapped_type::point_type;

    friend typename base_type::root_base_type;

protected:

    WrapBezier3d(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapBezier3d & wrap_management();
    WrapBezier3d & wrap_operator();
    WrapBezier3d & wrap_accessor();
    WrapBezier3d & wrap_geometry();
}; /* end class WrapBezier3d */

template <typename T>
WrapBezier3d<T>::WrapBezier3d(pybind11::module & mod, const char * pyname, const char * pydoc)
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
WrapBezier3d<T> & WrapBezier3d<T>::wrap_management()
{
    namespace py = pybind11;

    (*this)
        .def(py::init<point_type const &, point_type const &, point_type const &, point_type const &>(),
             py::arg("p0"),
             py::arg("p1"),
             py::arg("p2"),
             py::arg("p3"))
        .def(
            "__repr__",
            [](wrapped_type const & self)
            {
                static char const * btypename = std::is_same_v<T, double> ? "Bezier3dFp64" : "Bezier3dFp32";
                static char const * ptypename = std::is_same_v<T, double> ? "Point3dFp64" : "Point3dFp32";
                return std::format("{}({}({}), {}({}), {}({}), {}({}))",
                                   btypename,
                                   ptypename,
                                   self.p0().value_string(),
                                   ptypename,
                                   self.p1().value_string(),
                                   ptypename,
                                   self.p2().value_string(),
                                   ptypename,
                                   self.p3().value_string());
            })
        .def_alias("__repr__", "__str__")
        //
        ;

    return *this;
}

template <typename T>
WrapBezier3d<T> & WrapBezier3d<T>::wrap_operator()
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
WrapBezier3d<T> & WrapBezier3d<T>::wrap_accessor()
{
    namespace py = pybind11;

    (*this)
        .def("__len__",
             [](wrapped_type const &)
             { return 4; })
        .def(
            "__getitem__",
            [](wrapped_type const & self, size_t it)
            {
                point_type ret;
                switch (it)
                {
                case 0: ret = self.p0(); break;
                case 1: ret = self.p1(); break;
                case 2: ret = self.p2(); break;
                case 3: ret = self.p3(); break;
                default: throw std::out_of_range("Bezier3d: (control) i 4 >= size 4"); break;
                }
                return ret;
            })
        .def(
            "__setitem__",
            [](wrapped_type & self, size_t it, point_type const & p)
            {
                switch (it)
                {
                case 0: self.p0() = p; break;
                case 1: self.p1() = p; break;
                case 2: self.p2() = p; break;
                case 3: self.p3() = p; break;
                default: throw py::stop_iteration(); break;
                }
            })
        //
        ;

    return *this;
}

template <typename T>
WrapBezier3d<T> & WrapBezier3d<T>::wrap_geometry()
{
    namespace py = pybind11;

    (*this)
        .def("sample", &wrapped_type::sample, py::arg("nlocus"))
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
                    throw std::invalid_argument("Bezier3d::mirror: axis must be 'x', 'y', or 'z'");
                }
            },
            py::arg("axis"))
        //
        ;

    return *this;
}

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapCurvePad
    : public WrapBase<WrapCurvePad<T>, CurvePad<T>, std::shared_ptr<CurvePad<T>>>
{

public:

    using base_type = WrapBase<WrapCurvePad<T>, CurvePad<T>, std::shared_ptr<CurvePad<T>>>;
    using wrapped_type = typename base_type::wrapped_type;

    using value_type = typename base_type::wrapped_type::value_type;
    using point_type = typename base_type::wrapped_type::point_type;
    using segment_type = typename base_type::wrapped_type::segment_type;
    using bezier_type = typename base_type::wrapped_type::bezier_type;
    using point_pad_type = typename base_type::wrapped_type::point_pad_type;

    friend typename base_type::root_base_type;

protected:

    WrapCurvePad(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapCurvePad & wrap_management();
    WrapCurvePad & wrap_accessor_bezier();
    WrapCurvePad & wrap_accessor_point();
    WrapCurvePad & wrap_geometry();
}; /* end class WrapCurvePad */

template <typename T>
WrapCurvePad<T>::WrapCurvePad(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .wrap_management()
        .wrap_accessor_bezier()
        .wrap_accessor_point()
        .wrap_geometry()
        //
        ;
}

template <typename T>
WrapCurvePad<T> & WrapCurvePad<T>::wrap_management()
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
        //
        ;

    (*this)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        //
        ;

    return *this;
}

template <typename T>
WrapCurvePad<T> & WrapCurvePad<T>::wrap_accessor_bezier()
{
    namespace py = pybind11;

    // Python dunder accessor.
    (*this)
        .def("__len__", &wrapped_type::size)
        .def(
            "__getitem__",
            [](wrapped_type const & self, ssize_t it)
            {
                return self.get_at(normalize_pad_index(it, self.size(), "CurvePad"));
            },
            "Return the curve at the index. A negative index counts from the end of the pad.")
        .def(
            "__getitem__",
            [](wrapped_type const & self, py::slice const & key)
            {
                return copy_pad_slice(self, key);
            },
            "Return a new pad of the same ndim holding a copy of the selected curves. "
            "A write to the result does not reach the pad it was sliced from.")
        .def(
            "__setitem__",
            [](wrapped_type & self, ssize_t it, bezier_type const & c)
            {
                self.set_at(normalize_pad_index(it, self.size(), "CurvePad"), c);
            },
            "Write the curve at the index. A negative index counts from the end of the pad.")
        //
        ;

    // Append Bezier3d
    (*this)
        .def(
            "append",
            [](wrapped_type & self, bezier_type const & c)
            {
                self.append(c);
            },
            py::arg("c"))
        .def(
            "append",
            [](wrapped_type & self, point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
            {
                self.append(p0, p1, p2, p3);
            },
            py::arg("p0"),
            py::arg("p1"),
            py::arg("p2"),
            py::arg("p3"))
        //
        ;

    // Additional getter and setter for Bezier3d.
    (*this)
        .def("get_at", &wrapped_type::get_at)
        .def("set_at", &wrapped_type::set_at)
        //
        ;

    return *this;
}

template <typename T>
WrapCurvePad<T> & WrapCurvePad<T>::wrap_accessor_point()
{
    namespace py = pybind11;

    // x, y, z, point element accessors.
#define DECL_WRAP(NAME) \
    .def(#NAME, [](wrapped_type const & self, size_t i) { return self.NAME(i); })
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
        DECL_WRAP(x3_at)
        DECL_WRAP(y3_at)
        DECL_WRAP(z3_at)
        DECL_WRAP(p0_at)
        DECL_WRAP(p1_at)
        DECL_WRAP(p2_at)
        DECL_WRAP(p3_at)
        ;
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
        DECL_WRAP(x3)
        DECL_WRAP(y3)
        DECL_WRAP(z3)
        DECL_WRAP(p0)
        DECL_WRAP(p1)
        DECL_WRAP(p2)
        DECL_WRAP(p3)
        ;
    // clang-format on
#undef DECL_WRAP

    return *this;
}

template <typename T>
WrapCurvePad<T> & WrapCurvePad<T>::wrap_geometry()
{
    namespace py = pybind11;

    (*this)
        .def("sample", &wrapped_type::sample, py::arg("length"))
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
                    throw std::invalid_argument("CurvePad::mirror: axis must be 'x', 'y', or 'z'");
                }
            },
            py::arg("axis"))
        //
        ;

    return *this;
}

void wrap_shape1d(pybind11::module & mod)
{
    WrapSegment3d<float>::commit(mod, "Segment3dFp32", "Segment3dFp32");
    WrapSegment3d<double>::commit(mod, "Segment3dFp64", "Segment3dFp64");
    WrapSegmentPad<float>::commit(mod, "SegmentPadFp32", "SegmentPadFp32");
    WrapSegmentPad<double>::commit(mod, "SegmentPadFp64", "SegmentPadFp64");
    WrapBezier3d<float>::commit(mod, "Bezier3dFp32", "Bezier3dFp32");
    WrapBezier3d<double>::commit(mod, "Bezier3dFp64", "Bezier3dFp64");
    WrapCurvePad<float>::commit(mod, "CurvePadFp32", "CurvePadFp32");
    WrapCurvePad<double>::commit(mod, "CurvePadFp64", "CurvePadFp64");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: