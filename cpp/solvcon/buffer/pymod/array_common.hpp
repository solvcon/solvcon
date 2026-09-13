#pragma once

/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h> // Must be the first include.

#include <solvcon/buffer/SimpleArray.hpp>
#include <solvcon/buffer/pymod/TypeBroadcast.hpp>
#include <solvcon/math/math.hpp>

// We faced an issue where the template specialization for the caster of
// SimpleArray<T> doesn't function correctly on both macOS and Windows.
// While the root cause of the problem remains unclear, a workaround is
// available by including the caster header in this file, impacting
// wrap_SimpleArray.cpp.
// See more details in the issue: https://github.com/solvcon/solvcon/issues/283
#include <solvcon/buffer/pymod/SimpleArrayCaster.hpp>

#include <algorithm>
#include <cstring>
#include <limits>

namespace solvcon
{
namespace python
{
namespace detail
{

inline pybind11::dtype float16_dtype() { return pybind11::dtype("float16"); }
inline std::string float16_format() { return "e"; }

inline bool try_load_real(pybind11::handle src, double & value)
{
    if (!src || !PyNumber_Check(src.ptr()) ||
        pybind11::isinstance<Complex<float>>(src) ||
        pybind11::isinstance<Complex<double>>(src))
    {
        return false;
    }

    if (!PyFloat_Check(src.ptr()) && !PyLong_Check(src.ptr()) &&
        pybind11::module_::import("numpy").attr("iscomplexobj")(src).cast<bool>())
    {
        return false;
    }

    value = PyFloat_AsDouble(src.ptr());
    if (PyErr_Occurred())
    {
        PyErr_Clear();
        return false;
    }
    return true;
}

inline bool try_load_exact_float16(pybind11::handle src, Float16 & value)
{
    if (!src || !PyObject_CheckBuffer(src.ptr()) ||
        !pybind11::type::of(src).is(float16_dtype().attr("type")))
    {
        return false;
    }

    auto const info = pybind11::reinterpret_borrow<pybind11::buffer>(src).request();
    if (info.ndim != 0 || info.itemsize != sizeof(Float16) || info.format != float16_format())
    {
        return false;
    }

    Float16::storage_type bits;
    std::memcpy(&bits, info.ptr, sizeof(bits));
    value = Float16::from_bits(bits);
    return true;
}

inline bool try_load_float16_scalar(pybind11::handle src, bool convert, Float16 & value)
{
    if (!src)
    {
        return false;
    }
    if (try_load_exact_float16(src, value))
    {
        return true;
    }
    if (!convert && !PyFloat_Check(src.ptr()))
    {
        return false;
    }

    double real;
    if (!try_load_real(src, real))
    {
        return false;
    }
    value = Float16(real);
    return true;
}

} /* end namespace detail */
} /* end namespace python */
} /* end namespace solvcon */

namespace pybind11
{

namespace detail
{

template <>
struct type_caster<solvcon::Float16>
{
public:
    bool load(pybind11::handle src, bool convert);

    static pybind11::handle cast(solvcon::Float16 src, pybind11::return_value_policy, pybind11::handle)
    {
        return PyFloat_FromDouble(static_cast<float>(src));
    }

    PYBIND11_TYPE_CASTER(solvcon::Float16, const_name("float"));
}; /* end struct type_caster */

inline bool type_caster<solvcon::Float16>::load(pybind11::handle src, bool convert)
{
    return solvcon::python::detail::try_load_float16_scalar(src, convert, value);
}

template <>
struct npy_format_descriptor<solvcon::Float16>
{
    static constexpr auto name = const_name("numpy.float16");
    static pybind11::dtype dtype() { return solvcon::python::detail::float16_dtype(); }
    static std::string format() { return solvcon::python::detail::float16_format(); }
}; /* end struct npy_format_descriptor */

template <>
struct npy_format_descriptor<solvcon::Complex<double>>
{
    static constexpr auto name = const_name("complex128");
    static constexpr int value = npy_api::NPY_CDOUBLE_;

    static pybind11::dtype dtype()
    {
        return pybind11::dtype("complex128");
    }

    // The format string is used by numpy to correctly interpret the memory layout
    // of Complex<T> when converting between c++ and python.
    static std::string format()
    {
        return "=Zd";
    }

    static void register_dtype(any_container<field_descriptor> fields)
    {
        register_structured_dtype(std::move(fields),
                                  typeid(std::remove_cv_t<solvcon::Complex<double>>),
                                  sizeof(solvcon::Complex<double>),
                                  &direct_converter);
    }

private:
    static PyObject * dtype_ptr()
    {
        // NOLINTNEXTLINE(misc-const-correctness)
        static PyObject * ptr = get_numpy_internals().get_type_info<solvcon::Complex<double>>(true)->dtype_ptr;
        return ptr;
    }

    static bool direct_converter(PyObject * obj, void *& storage)
    {
        auto & api = npy_api::get();
        if (!PyObject_TypeCheck(obj, api.PyVoidArrType_Type_))
        {
            return false;
        }
        if (auto descr = reinterpret_steal<object>(api.PyArray_DescrFromScalar_(obj)))
        {
            if (api.PyArray_EquivTypes_(dtype_ptr(), descr.ptr()))
            {
                storage = (reinterpret_cast<PyVoidScalarObject_Proxy *>(obj))->obval; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                return true;
            }
        }
        return false;
    }
}; /* end struct npy_format_descriptor */

template <>
struct npy_format_descriptor<solvcon::Complex<float>>
{
    static constexpr auto name = const_name("complex64");
    static constexpr int value = npy_api::NPY_CFLOAT_;

    static pybind11::dtype dtype()
    {
        return pybind11::dtype("complex64");
    }

    static std::string format()
    {
        return "=Zf";
    }

    static void register_dtype(any_container<field_descriptor> fields)
    {
        register_structured_dtype(std::move(fields),
                                  typeid(std::remove_cv_t<solvcon::Complex<float>>),
                                  sizeof(solvcon::Complex<float>),
                                  &direct_converter);
    }

private:
    static PyObject * dtype_ptr()
    {
        // NOLINTNEXTLINE(misc-const-correctness)
        static PyObject * ptr = get_numpy_internals().get_type_info<solvcon::Complex<double>>(true)->dtype_ptr;
        return ptr;
    }

    static bool direct_converter(PyObject * obj, void *& storage)
    {
        auto & api = npy_api::get();
        if (!PyObject_TypeCheck(obj, api.PyVoidArrType_Type_))
        {
            return false;
        }
        if (auto descr = reinterpret_steal<object>(api.PyArray_DescrFromScalar_(obj)))
        {
            if (api.PyArray_EquivTypes_(dtype_ptr(), descr.ptr()))
            {
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                storage = (reinterpret_cast<PyVoidScalarObject_Proxy *>(obj))->obval;
                return true;
            }
        }
        return false;
    }
}; /* end struct npy_format_descriptor */

} /* end namespace detail */

} /* end namespace pybind11 */

namespace solvcon
{

namespace python
{

inline solvcon::detail::shape_type make_shape(pybind11::object const & shape_in)
{
    solvcon::detail::shape_type shape;
    try
    {
        shape.push_back(shape_in.cast<ssize_t>());
    }
    catch (const pybind11::cast_error &)
    {
        shape = shape_in.cast<std::vector<ssize_t>>();
    }
    return shape;
}

/// Helper class for array property in Python.
template <typename T>
class ArrayPropertyHelper
{
public:
    using shape_type = solvcon::detail::shape_type;
    using slices_type = typename TypeBroadcast<T>::slices_type;

    static SimpleArray<T> getitem(SimpleArray<T> & array, pybind11::object const & key);
    static void setitem(SimpleArray<T> & array, pybind11::object const & key, pybind11::object const & value);

    static pybind11::buffer_info get_buffer_info(SimpleArray<T> & array)
    {
        std::vector<pybind11::ssize_t> stride;
        auto const itemsize = static_cast<pybind11::ssize_t>(sizeof(T));
        for (ssize_t const i : array.stride())
        {
            stride.push_back(static_cast<pybind11::ssize_t>(i) * itemsize);
        }

        // Special handling for Complex types
        std::string format;
        if constexpr (is_complex_v<T>)
        {
            if constexpr (std::is_same_v<T, Complex<double>>)
            {
                format = pybind11::format_descriptor<Complex<double>>::format();
            }
            else
            {
                format = pybind11::format_descriptor<Complex<float>>::format();
            }
        }
        else
        {
            format = pybind11::format_descriptor<T>::format();
        }

        return pybind11::buffer_info(
            array.logical_data(), /* Pointer to buffer */
            sizeof(T), /* Size of one scalar */
            format, /* Python struct-style format descriptor */
            array.ndim(), /* Number of dimensions */
            std::vector<pybind11::ssize_t>(array.shape().begin(), array.shape().end()), /* Buffer dimensions */
            stride /* Strides (in bytes) for each index */
        );
    }

private:
    static slices_type parse_slices(SimpleArray<T> const & array, pybind11::object const & key);
    static void copy_key(shape_type & slice, pybind11::handle key, SimpleArray<T> const & arr, size_t axis);

    static bool is_sequence(pybind11::object const & py_value)
    {
        return pybind11::isinstance<pybind11::list>(py_value) ||
               pybind11::isinstance<pybind11::array>(py_value) ||
               pybind11::isinstance<pybind11::tuple>(py_value);
    }

    static bool is_scalar(pybind11::object const & py_value)
    {
        if (is_sequence(py_value))
        {
            return false;
        }

        bool const is_number = PyNumber_Check(py_value.ptr());

        if constexpr (std::is_same_v<T, Complex<float>> || std::is_same_v<T, Complex<double>>)
        {
            return is_number || pybind11::isinstance<T>(py_value);
        }
        else
        {
            return is_number;
        }
    }

    template <typename U>
    static Complex<U> cast_complex_scalar(
        pybind11::object const & py_value)
    {
        pybind11::object const complex_class =
            pybind11::module_::import("builtins").attr("complex");
        return complex_class(py_value).cast<std::complex<U>>();
    }

    static T cast_scalar(pybind11::object const & py_value)
    {
        if constexpr (std::is_same_v<T, Complex<float>>)
        {
            return cast_complex_scalar<float>(py_value);
        }
        else if constexpr (std::is_same_v<T, Complex<double>>)
        {
            return cast_complex_scalar<double>(py_value);
        }
        else
        {
            return py_value.cast<T>();
        }
    }

    static slices_type make_default_slices(SimpleArray<T> const & arr)
    {
        slices_type slices;
        auto const & shape = arr.shape();
        slices.reserve(shape.size());
        for (ssize_t const dim : shape)
        {
            shape_type default_slice(4);
            default_slice[0] = 0; // start
            default_slice[1] = dim; // stop
            default_slice[2] = 1; // step
            default_slice[3] = dim; // length
            slices.push_back(std::move(default_slice));
        }
        return slices;
    }

    static pybind11::object shift_slice_bound(pybind11::handle bound, ssize_t offset);

    static void copy_slice(shape_type & slice_out,
                           pybind11::slice const & slice_in,
                           ssize_t length,
                           ssize_t offset)
    {
        pybind11::slice normalized_slice = slice_in;
        if (offset != 0)
        {
            pybind11::object const start = shift_slice_bound(slice_in.attr("start"), offset);
            pybind11::object const stop = shift_slice_bound(slice_in.attr("stop"), offset);
            normalized_slice = pybind11::slice(start, stop, slice_in.attr("step"));
        }

        pybind11::ssize_t start = 0;
        pybind11::ssize_t stop = 0;
        pybind11::ssize_t step = 0;
        pybind11::ssize_t slicelength = 0;
        if (!normalized_slice.compute(length, &start, &stop, &step, &slicelength))
        {
            throw pybind11::error_already_set();
        }

        slice_out[0] = start;
        slice_out[1] = stop;
        slice_out[2] = step;
        slice_out[3] = slicelength;
    }

    static void slice_syntax_check(pybind11::tuple const & tuple, ssize_t ndim)
    {
        namespace py = pybind11;

        ssize_t ellipsis_cnt = 0;
        ssize_t index_cnt = 0;

        for (auto it = tuple.begin(); it != tuple.end(); it++)
        {
            if (py::isinstance<py::ellipsis>(*it))
            {
                ellipsis_cnt += 1;
            }
            else if (py::isinstance<py::slice>(*it) || PyIndex_Check((*it).ptr()))
            {
                index_cnt += 1;
            }
            else
            {
                throw std::runtime_error("unsupported operation.");
            }
        }

        if (index_cnt > ndim)
        {
            throw std::runtime_error("syntax error. dimensions mismatches");
        }

        if (ellipsis_cnt > 1)
        {
            throw std::runtime_error("syntax error. no more than one ellipsis.");
        }
    }

    static void process_slices(pybind11::tuple const & tuple, slices_type & slices, SimpleArray<T> const & arr)
    {
        namespace py = pybind11;

        ssize_t const ndim = arr.ndim();
        slice_syntax_check(tuple, ndim);

        // copy slices from the front until an ellipsis
        bool ellipsis_flag = false;
        for (auto it = tuple.begin(); it != tuple.end(); it++)
        {
            if (py::isinstance<py::ellipsis>(*it))
            {
                // stop here and iterator the tuple from back later
                ellipsis_flag = true;
                break;
            }

            ssize_t const axis = it - tuple.begin();
            copy_key(slices[axis], *it, arr, axis);
        }

        // copy slices from the back until an ellipsis
        if (ellipsis_flag)
        {
            ssize_t const tuple_size = tuple.size();
            for (ssize_t offset = 0; offset < tuple_size; ++offset)
            {
                auto it = tuple.end() - offset - 1;

                if (py::isinstance<py::ellipsis>(*it))
                {
                    break;
                }
                ssize_t const axis = ndim - offset - 1;
                copy_key(slices[axis], *it, arr, axis);
            }
        }
    }

    static void assign_slice(SimpleArray<T> & arr_out, slices_type const & slices, pybind11::array const & arr_in)
    {
        TypeBroadcast<T>::check_shape(arr_out, slices, arr_in);
        TypeBroadcast<T>::broadcast(arr_out, slices, arr_in);
    }
}; /* end class ArrayPropertyHelper */

template <typename T>
void ArrayPropertyHelper<T>::copy_key(shape_type & slice, pybind11::handle key, SimpleArray<T> const & arr, size_t axis)
{
    namespace py = pybind11;
    ssize_t const ghost = axis == 0 ? arr.nghost() : 0;
    ssize_t const length = arr.shape(axis);
    if (py::isinstance<py::slice>(key))
    {
        copy_slice(slice, py::reinterpret_borrow<py::slice>(key), length, ghost);
        return;
    }

    py::object const shifted = shift_slice_bound(key, ghost);
    ssize_t index = PyNumber_AsSsize_t(shifted.ptr(), PyExc_IndexError);
    if (index == -1 && PyErr_Occurred())
    {
        throw py::error_already_set();
    }
    if (index < -length || index >= length)
    {
        throw py::index_error(std::format("index out of range for axis {}", axis));
    }
    index = index < 0 ? index + length : index;
    // A zero step marks an integer index, which removes this axis from the view.
    slice[0] = index;
    slice[1] = index + 1;
    slice[2] = 0;
    slice[3] = 1;
}

template <typename T>
typename ArrayPropertyHelper<T>::slices_type ArrayPropertyHelper<T>::parse_slices(
    SimpleArray<T> const & array, pybind11::object const & key)
{
    namespace py = pybind11;
    slices_type slices = make_default_slices(array);
    if (py::isinstance<py::tuple>(key))
    {
        process_slices(key.cast<py::tuple>(), slices, array);
    }
    else if (py::isinstance<py::slice>(key))
    {
        if (array.ndim() == 0)
        {
            throw py::index_error("cannot slice a zero-dimensional array");
        }
        copy_slice(slices[0], key.cast<py::slice>(), array.shape(0), array.nghost());
    }
    else if (!py::isinstance<py::ellipsis>(key))
    {
        throw py::type_error("expected a slice, a tuple of slices, or an ellipsis");
    }
    return slices;
}

template <typename T>
void ArrayPropertyHelper<T>::setitem(SimpleArray<T> & array, pybind11::object const & key, pybind11::object const & value)
{
    namespace py = pybind11;
    if (is_scalar(value))
    {
        if (py::isinstance<py::int_>(key))
        {
            array.at(key.cast<ssize_t>()) = cast_scalar(value);
            return;
        }
        if (py::isinstance<py::tuple>(key))
        {
            array.at(key.cast<std::vector<ssize_t>>()) = cast_scalar(value);
            return;
        }
    }
    if (is_sequence(value) &&
        (py::isinstance<py::slice>(key) || py::isinstance<py::tuple>(key) || py::isinstance<py::ellipsis>(key)))
    {
        SimpleArray<T> view = getitem(array, key);
        assign_slice(view, make_default_slices(view), value.cast<py::array>());
        return;
    }
    throw std::runtime_error("unsupported operation.");
}

template <typename T>
SimpleArray<T> ArrayPropertyHelper<T>::getitem(SimpleArray<T> & array, pybind11::object const & key)
{
    slices_type const slices = parse_slices(array, key);
    if (array.ndim() == 0 && !array.logical_data())
    {
        throw pybind11::index_error("cannot slice an array without storage");
    }

    shape_type shape, stride;
    constexpr auto itemsize = static_cast<ssize_t>(sizeof(T));
    for (size_t axis = 0; axis < slices.size(); ++axis)
    {
        shape_type const & slice = slices[axis];
        ssize_t const source_stride = array.stride(axis);
        ssize_t const step = slice[2];
        if (step == 0)
        {
            continue;
        }
        ssize_t const max = std::numeric_limits<ssize_t>::max();
        ssize_t const min = std::numeric_limits<ssize_t>::min();
        if ((source_stride > 0 && (step > max / source_stride || step < min / source_stride)) ||
            (source_stride < -1 && (step > min / source_stride || step < max / source_stride)) ||
            (source_stride == -1 && step == min))
        {
            throw std::overflow_error("slice stride exceeds the supported range");
        }
        ssize_t const view_stride = source_stride * step;
        if (view_stride > max / itemsize || view_stride < min / itemsize)
        {
            throw std::overflow_error("slice byte stride exceeds the supported range");
        }
        shape.push_back(slice[3]);
        stride.push_back(view_stride);
    }

    bool const empty = std::find(shape.begin(), shape.end(), 0) != shape.end();
    ssize_t offset = array.logical_data() ? array.logical_data() - array.data() : 0;
    if (!empty)
    {
        for (size_t axis = 0; axis < slices.size(); ++axis)
        {
            offset += slices[axis][0] * array.stride(axis);
        }
    }
    return SimpleArray<T>(shape, stride, array.buffer().shared_from_this(), offset * itemsize);
}

template <typename T>
pybind11::object ArrayPropertyHelper<T>::shift_slice_bound(
    pybind11::handle bound, ssize_t offset)
{
    if (bound.is_none())
    {
        return pybind11::none();
    }

    PyObject * index = PyNumber_Index(bound.ptr());
    if (index == nullptr)
    {
        throw pybind11::error_already_set();
    }
    return pybind11::reinterpret_steal<pybind11::object>(index) + pybind11::int_(offset);
}

} /* end namespace python */
} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
