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
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapPoint3d
    : public WrapBase<WrapPoint3d<T>, Point3d<T>>
{

public:

    using base_type = WrapBase<WrapPoint3d<T>, Point3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;

    using value_type = typename base_type::wrapped_type::value_type;

    friend typename base_type::root_base_type;

protected:

    WrapPoint3d(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapPoint3d & wrap_management();
    WrapPoint3d & wrap_operator();
    WrapPoint3d & wrap_accessor();
    WrapPoint3d & wrap_geometry();
}; /* end class WrapPoint3d */

template <typename T>
WrapPoint3d<T>::WrapPoint3d(pybind11::module & mod, const char * pyname, const char * pydoc)
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
WrapPoint3d<T> & WrapPoint3d<T>::wrap_management()
{
    namespace py = pybind11;

    // Constructors.
    (*this)
        .def(py::init<value_type, value_type>(), py::arg("x"), py::arg("y"))
        .def(py::init<value_type, value_type, value_type>(), py::arg("x"), py::arg("y"), py::arg("z"))
        //
        ;

    // String representations.
    (*this)
        .def(
            "__repr__",
            [](wrapped_type const & self)
            {
                // Hard-code the Python type names in the static variable before finding a systematic way.
                static char const * ptypename = std::is_same_v<T, double> ? "Point3dFp64" : "Point3dFp32";
                return std::format("{}({})", ptypename, self.value_string());
            })
        .def_alias("__repr__", "__str__")
        //
        ;

    return *this;
}

template <typename T>
WrapPoint3d<T> & WrapPoint3d<T>::wrap_operator()
{
    namespace py = pybind11;

    // Wrap for operators.
    (*this)
        .def(py::self == py::self) // NOLINT(misc-redundant-expression)
        .def(py::self != py::self) // NOLINT(misc-redundant-expression)
        .def(py::self += py::self)
        .def(py::self += value_type())
        .def(py::self -= py::self)
        .def(py::self -= value_type())
        .def(py::self *= value_type())
        .def(py::self /= value_type())
        //
        ;

    return *this;
}

template <typename T>
WrapPoint3d<T> & WrapPoint3d<T>::wrap_accessor()
{
    namespace py = pybind11;

    // Python dunder accessors.
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
            [](wrapped_type & self, size_t it, value_type val)
            { self.at(it) = val; })
        .def("fill", &wrapped_type::fill, py::arg("value"))
        //
        ;

    // x, y, z accessors.
#define DECL_WRAP(NAME)                                              \
    .def_property(                                                   \
        #NAME,                                                       \
        [](wrapped_type const & self)                                \
        { return self.NAME(); },                                     \
        [](wrapped_type & self, typename wrapped_type::value_type v) \
        { self.NAME() = v; })

    // clang-format off
   (*this)
       DECL_WRAP(x)
       DECL_WRAP(y)
       DECL_WRAP(z)
       ;
#undef DECL_WRAP
    // clang-format on

    return *this;
}

template <typename T>
WrapPoint3d<T> & WrapPoint3d<T>::wrap_geometry()
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
                    throw std::invalid_argument("Point3d::mirror: axis must be 'x', 'y', or 'z'");
                }
            },
            py::arg("axis"))
        //
        ;

    return *this;
}

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapPointPad
    : public WrapBase<WrapPointPad<T>, PointPad<T>, std::shared_ptr<PointPad<T>>>
{

public:

    using base_type = WrapBase<WrapPointPad<T>, PointPad<T>, std::shared_ptr<PointPad<T>>>;
    using wrapped_type = typename base_type::wrapped_type;
    using wrapper_type = typename base_type::wrapper_type;

    using value_type = typename base_type::wrapped_type::value_type;
    using point_type = typename base_type::wrapped_type::point_type;

    friend typename base_type::root_base_type;

protected:

    WrapPointPad(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapPointPad & wrap_management();
    WrapPointPad & wrap_accessor_point();
    WrapPointPad & wrap_accessor_value();
    WrapPointPad & wrap_geometry();
}; /* end class WrapPointPad */

template <typename T>
WrapPointPad<T>::WrapPointPad(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .wrap_management()
        .wrap_accessor_point()
        .wrap_accessor_value()
        .wrap_geometry()
        //
        ;
}

template <typename T>
WrapPointPad<T> & WrapPointPad<T>::wrap_management()
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
                [](uint8_t ndim, size_t nelem, size_t alignment)
                { return wrapped_type::construct(ndim, nelem, alignment); }),
            py::arg("ndim"),
            py::arg("nelem"),
            py::arg("alignment"))
        .def(
            py::init(
                [](SimpleArray<T> & x, SimpleArray<T> & y, bool clone)
                { return wrapped_type::construct(x, y, clone); }),
            py::arg("x"),
            py::arg("y"),
            py::arg("clone"))
        .def(
            py::init(
                [](SimpleArray<T> & x, SimpleArray<T> & y, bool clone, size_t alignment)
                { return wrapped_type::construct(x, y, clone, alignment); }),
            py::arg("x"),
            py::arg("y"),
            py::arg("clone"),
            py::arg("alignment"))
        .def(
            py::init(
                [](SimpleArray<T> & x, SimpleArray<T> & y, SimpleArray<T> & z, bool clone)
                { return wrapped_type::construct(x, y, z, clone); }),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("clone"))
        .def(
            py::init(
                [](SimpleArray<T> & x, SimpleArray<T> & y, SimpleArray<T> & z, bool clone, size_t alignment)
                { return wrapped_type::construct(x, y, z, clone, alignment); }),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("clone"),
            py::arg("alignment"))
        //
        ;

    // Container management.
    (*this)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_property_readonly("alignment", &wrapped_type::alignment)
        .def_timed("pack_array", &wrapped_type::pack_array)
        .def_timed("expand", &wrapped_type::expand, py::arg("length"))
        //
        ;

    return *this;
}

template <typename T>
WrapPointPad<T> & WrapPointPad<T>::wrap_accessor_point()
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

    // Append Point3d
    (*this)
        .def(
            "append",
            [](wrapped_type & self, value_type x, value_type y)
            {
                self.append(x, y);
            },
            py::arg("x"),
            py::arg("y"))
        .def(
            "append",
            [](wrapped_type & self, value_type x, value_type y, value_type z)
            {
                self.append(x, y, z);
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"))
        //
        ;

    (*this)
        .def("get_at",
             [](wrapped_type const & self, size_t it)
             {
                 return self.get_at(it);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, point_type const & v)
             {
                 self.set_at(it, v);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, value_type x, value_type y)
             {
                 self.set_at(it, x, y);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, value_type x, value_type y, value_type z)
             {
                 self.set_at(it, x, y, z);
             })
        //
        ;

    return *this;
}

template <typename T>
WrapPointPad<T> & WrapPointPad<T>::wrap_accessor_value()
{
    namespace py = pybind11;

    // x, y, z element accessors.
#define DECL_WRAP(NAME)                         \
    .def(                                       \
        #NAME,                                  \
        [](wrapped_type const & self, size_t i) \
        { return self.NAME(i); })
    // clang-format off
    (*this)
        DECL_WRAP(x_at)
        DECL_WRAP(y_at)
        DECL_WRAP(z_at)
        ;
#undef DECL_WRAP
    // clang-format on

    // x, y, z array/batch accessors.
#define DECL_WRAP(NAME)         \
    .def_property_readonly(     \
        #NAME,                  \
        [](wrapped_type & self) \
        { return self.NAME(); })
    // clang-format off
    (*this)
        DECL_WRAP(x)
        DECL_WRAP(y)
        DECL_WRAP(z)
        ;
#undef DECL_WRAP
    // clang-format on

    return *this;
}

template <typename T>
WrapPointPad<T> & WrapPointPad<T>::wrap_geometry()
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
                    throw std::invalid_argument("PointPad::mirror: axis must be 'x', 'y', or 'z'");
                }
            },
            py::arg("axis"))
        //
        ;

    return *this;
}

void wrap_shape0d(pybind11::module & mod)
{
    namespace py = pybind11;

    mod
        .def("calc_bernstein_polynomial", calc_bernstein_polynomial, py::arg("t"), py::arg("i"), py::arg("n"))
        .def("interpolate_bernstein", interpolate_bernstein, py::arg("t"), py::arg("values"), py::arg("n"))
        //
        ;

    WrapPoint3d<float>::commit(mod, "Point3dFp32", "Point3dFp32");
    WrapPoint3d<double>::commit(mod, "Point3dFp64", "Point3dFp64");
    WrapPointPad<float>::commit(mod, "PointPadFp32", "PointPadFp32");
    WrapPointPad<double>::commit(mod, "PointPadFp64", "PointPadFp64");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: