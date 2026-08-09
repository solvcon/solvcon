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

namespace pybind11
{

namespace detail
{

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

    static bool direct_converter(PyObject * obj, void *& value)
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
                value = (reinterpret_cast<PyVoidScalarObject_Proxy *>(obj))->obval; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
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

    static bool direct_converter(PyObject * obj, void *& value)
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
                value = (reinterpret_cast<PyVoidScalarObject_Proxy *>(obj))->obval;
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

/**
 * Parser of the Python subscript keys of an array.
 *
 * One instance binds to one array and serves both `__getitem__` and
 * `__setitem__`, so the two operators read the same key grammar: a scalar
 * key addresses one element, while a slice, an ellipsis, or a tuple
 * containing either selects a region.
 */
template <typename T>
class ArrayPropertyHelper
{
public:
    using shape_type = solvcon::detail::shape_type;
    using slice_type = solvcon::detail::slice_type;

    explicit ArrayPropertyHelper(SimpleArray<T> & arr)
        : m_arr(arr)
    {
    }

    /// Report whether the key selects a region rather than a single element.
    static bool is_region_key(pybind11::object const & key);

    pybind11::object getitem(pybind11::object const & key) const;

    void setitem(pybind11::object const & key, pybind11::object const & value);

    /// Build the view that a region key selects.
    SimpleArray<T> slice(pybind11::object const & key) const { return m_arr.slice(slices_from_key(key)); }

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

    static bool is_index(pybind11::handle key) { return 0 != PyIndex_Check(key.ptr()); }

    static bool is_index_tuple(pybind11::object const & key);

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

    slice_type make_default_slices() const;

    slice_type slices_from_key(pybind11::object const & key) const;

    static pybind11::object shift_slice_bound(pybind11::handle bound, ssize_t offset);

    static void copy_slice(AxisSlice & slice_out,
                           pybind11::slice const & slice_in,
                           ssize_t length,
                           ssize_t offset);

    void slice_syntax_check(pybind11::tuple const & tuple) const;

    void process_slices(pybind11::tuple const & tuple, slice_type & slices) const;

    void broadcast_array_using_slice(slice_type const & slices, pybind11::array const & arr_in);

    SimpleArray<T> & m_arr;
}; /* end class ArrayPropertyHelper */

template <typename T>
bool ArrayPropertyHelper<T>::is_region_key(pybind11::object const & key)
{
    namespace py = pybind11;

    return py::isinstance<py::slice>(key) ||
           py::isinstance<py::ellipsis>(key) ||
           (py::isinstance<py::tuple>(key) && !is_index_tuple(key));
}

template <typename T>
bool ArrayPropertyHelper<T>::is_index_tuple(pybind11::object const & key)
{
    const pybind11::tuple tuple_in = key;
    for (auto it = tuple_in.begin(); it != tuple_in.end(); it++)
    {
        if (!is_index(*it))
        {
            return false;
        }
    }
    return true;
}

template <typename T>
pybind11::object ArrayPropertyHelper<T>::getitem(pybind11::object const & key) const
{
    namespace py = pybind11;

    // sarr[K]
    if (is_index(key))
    {
        return py::cast(m_arr.at(key.cast<ssize_t>()), py::return_value_policy::copy);
    }
    // sarr[K1, K2, K3] and sarr[[K1, K2, K3]]
    if (py::isinstance<py::list>(key) || (py::isinstance<py::tuple>(key) && is_index_tuple(key)))
    {
        return py::cast(m_arr.at(key.cast<std::vector<ssize_t>>()), py::return_value_policy::copy);
    }
    // sarr[slice], sarr[ellipsis], and sarr[slice, slice, ellipsis]
    if (is_region_key(key))
    {
        return py::cast(slice(key));
    }
    throw std::runtime_error("unsupported operation.");
}

template <typename T>
void ArrayPropertyHelper<T>::setitem(pybind11::object const & key, pybind11::object const & value)
{
    namespace py = pybind11;

    if (is_scalar(value))
    {
        // sarr[K] = V
        if (py::isinstance<py::int_>(key))
        {
            m_arr.at(key.cast<ssize_t>()) = cast_scalar(value);
            return;
        }
        // sarr[K1, K2, K3] = V
        if (py::isinstance<py::tuple>(key))
        {
            m_arr.at(key.cast<std::vector<ssize_t>>()) = cast_scalar(value);
            return;
        }
    }
    // sarr[slice] = ndarr, sarr[ellipsis] = ndarr, and
    // sarr[slice, slice, ellipsis] = ndarr
    else if (is_sequence(value) && is_region_key(key))
    {
        broadcast_array_using_slice(slices_from_key(key), value.cast<py::array>());
        return;
    }
    throw std::runtime_error("unsupported operation.");
}

template <typename T>
typename ArrayPropertyHelper<T>::slice_type ArrayPropertyHelper<T>::make_default_slices() const
{
    auto const & shape = m_arr.shape();
    slice_type slices(shape.size());
    for (size_t axis = 0; axis < shape.size(); ++axis)
    {
        slices[axis] = AxisSlice{.start = 0, .step = 1, .length = shape[axis]};
    }
    return slices;
}

template <typename T>
typename ArrayPropertyHelper<T>::slice_type
ArrayPropertyHelper<T>::slices_from_key(pybind11::object const & key) const
{
    namespace py = pybind11;

    slice_type slices = make_default_slices();
    if (py::isinstance<py::slice>(key))
    {
        if (0 == m_arr.ndim())
        {
            throw std::runtime_error("SimpleArray: cannot slice a zero-dimensional array");
        }
        copy_slice(slices[0], key.cast<py::slice>(), m_arr.shape(0), m_arr.nghost());
    }
    else if (py::isinstance<py::tuple>(key))
    {
        const py::tuple tuple_in = key;
        process_slices(tuple_in, slices);
    }
    return slices;
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

template <typename T>
void ArrayPropertyHelper<T>::copy_slice(AxisSlice & slice_out,
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

    slice_out.start = start;
    slice_out.step = step;
    slice_out.length = slicelength;
}

template <typename T>
void ArrayPropertyHelper<T>::broadcast_array_using_slice(slice_type const & slices, pybind11::array const & arr_in)
{
    TypeBroadcast<T>::check_shape(m_arr, slices, arr_in);
    TypeBroadcast<T>::broadcast(m_arr, slices, arr_in);
}

template <typename T>
void ArrayPropertyHelper<T>::slice_syntax_check(pybind11::tuple const & tuple) const
{
    namespace py = pybind11;

    ssize_t ellipsis_cnt = 0;
    ssize_t slice_cnt = 0;

    for (auto it = tuple.begin(); it != tuple.end(); it++)
    {
        if (py::isinstance<py::ellipsis>(*it))
        {
            ellipsis_cnt += 1;
        }
        else if (py::isinstance<py::slice>(*it))
        {
            slice_cnt += 1;
        }
        else
        {
            throw std::runtime_error("unsupported operation.");
        }
    }

    if (slice_cnt > m_arr.ndim())
    {
        throw std::runtime_error("syntax error. dimensions mismatches");
    }

    if (ellipsis_cnt > 1)
    {
        throw std::runtime_error("syntax error. no more than one ellipsis.");
    }
}

template <typename T>
void ArrayPropertyHelper<T>::process_slices(pybind11::tuple const & tuple, slice_type & slices) const
{
    namespace py = pybind11;

    ssize_t const ndim = m_arr.ndim();
    slice_syntax_check(tuple);

    // copy slices from the front until an ellipsis
    bool ellipsis_flag = false;
    for (auto it = tuple.begin(); it != tuple.end(); it++)
    {
        if (py::isinstance<py::ellipsis>(*it))
        {
            // stop here and iterate the tuple from the back later
            ellipsis_flag = true;
            break;
        }

        ssize_t const axis = it - tuple.begin();
        auto & slice_out = slices[axis];
        const auto slice_in = (*it).cast<py::slice>();

        ssize_t const bound_offset = axis == 0 ? m_arr.nghost() : 0;
        copy_slice(slice_out, slice_in, m_arr.shape(axis), bound_offset);
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
            auto & slice_out = slices[axis];
            const auto slice_in = (*it).cast<py::slice>();

            ssize_t const bound_offset = axis == 0 ? m_arr.nghost() : 0;
            copy_slice(slice_out, slice_in, m_arr.shape(axis), bound_offset);
        }
    }
}

} /* end namespace python */
} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
