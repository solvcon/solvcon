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

#include <cstring>

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

    static void broadcast_array_using_ellipsis(SimpleArray<T> & arr_out, pybind11::array const & arr_in)
    {
        auto slices = make_default_slices(arr_out);
        broadcast_array_using_slice(arr_out, slices, arr_in);
    }

    // FIXME: NOLINTNEXTLINE(readability-function-cognitive-complexity)
    static void setitem_parser(SimpleArray<T> & arr_out, pybind11::args const & args)
    {
        namespace py = pybind11;

        if (args.size() == 2)
        {
            const py::object & py_key = args[0];
            const py::object & py_value = args[1];

            const bool is_sequence_value = is_sequence(py_value);
            const bool is_scalar_value = is_scalar(py_value);

            // sarr[K] = V
            if (py::isinstance<py::int_>(py_key) && is_scalar_value)
            {
                const auto key = py_key.cast<ssize_t>();
                arr_out.at(key) = cast_scalar(py_value);
                return;
            }
            // sarr[K1, K2, K3] = V
            if (py::isinstance<py::tuple>(py_key) && is_scalar_value)
            {
                const auto key = py_key.cast<std::vector<ssize_t>>();
                arr_out.at(key) = cast_scalar(py_value);
                return;
            }

            // multi-dimension with slice and ellipsis
            // sarr[slice, slice, ellipsis] = ndarr
            if (py::isinstance<py::tuple>(py_key) && is_sequence_value)
            {
                const py::tuple tuple_in = py_key;
                const py::array arr_in = py_value;

                auto slices = make_default_slices(arr_out);
                process_slices(tuple_in, slices, arr_out);

                broadcast_array_using_slice(arr_out, slices, arr_in);
                return;
            }
            // one-dimension with slice
            // sarr[slice] = ndarr
            if (py::isinstance<py::slice>(py_key) && is_sequence_value)
            {
                const auto slice_in = py_key.cast<py::slice>();
                const auto arr_in = py_value.cast<py::array>();

                auto slices = make_default_slices(arr_out);
                copy_slice(slices[0], slice_in, arr_out.shape(0), arr_out.nghost());

                broadcast_array_using_slice(arr_out, slices, arr_in);
                return;
            }
            // sarr[ellipsis] = ndarr
            if (py::isinstance<py::ellipsis>(py_key) && is_sequence_value)
            {
                const auto arr_in = py_value.cast<py::array>();

                broadcast_array_using_ellipsis(arr_out, arr_in);
                return;
            }
        }
        throw std::runtime_error("unsupported operation.");
    }

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

    static std::vector<shape_type> make_default_slices(SimpleArray<T> const & arr)
    {
        std::vector<shape_type> slices;
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

        if (slice_cnt > ndim)
        {
            throw std::runtime_error("syntax error. dimensions mismatches");
        }

        if (ellipsis_cnt > 1)
        {
            throw std::runtime_error("syntax error. no more than one ellipsis.");
        }
    }

    static void process_slices(pybind11::tuple const & tuple,
                               std::vector<shape_type> & slices,
                               SimpleArray<T> const & arr)
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
            auto & slice_out = slices[axis];
            const auto slice_in = (*it).cast<py::slice>();

            ssize_t const bound_offset = axis == 0 ? arr.nghost() : 0;
            copy_slice(slice_out, slice_in, arr.shape(axis), bound_offset);
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

                ssize_t const bound_offset = axis == 0 ? arr.nghost() : 0;
                copy_slice(slice_out, slice_in, arr.shape(axis), bound_offset);
            }
        }
    }

    static void broadcast_array_using_slice(SimpleArray<T> & arr_out,
                                            std::vector<shape_type> const & slices,
                                            pybind11::array const & arr_in)
    {
        TypeBroadcast<T>::check_shape(arr_out, slices, arr_in);
        TypeBroadcast<T>::broadcast(arr_out, slices, arr_in);
    }
}; /* end class ArrayPropertyHelper */

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

/**
 * Build an array viewing the memory of a numpy array.
 *
 * The result shares the buffer with @p arr_in and keeps it alive, so it
 * carries the per-axis strides in elements and the offset a negative stride
 * puts the first element at.
 *
 * @param[in] arr_in Source array, whose dtype must be @p T.
 * @return An array over the same memory.
 */
template <typename T>
SimpleArray<T> make_array_from_numpy(pybind11::array & arr_in)
{
    namespace py = pybind11;

    using value_type = typename SimpleArray<T>::value_type;
    using array_order_type = typename SimpleArray<T>::ArrayOrder;

    if (!dtype_is_type<T>(arr_in))
    {
        throw std::runtime_error("dtype mismatch");
    }

    solvcon::detail::shape_type shape;
    solvcon::detail::shape_type stride;
    constexpr auto itemsize = static_cast<ssize_t>(sizeof(value_type));
    ssize_t byte_span_begin = 0;
    ssize_t byte_span_end = 0;
    bool has_element = true;
    for (ssize_t i = 0; i < arr_in.ndim(); ++i)
    {
        shape.push_back(arr_in.shape(i));
        ssize_t const byte_stride = arr_in.strides(i);
        if (byte_stride % itemsize != 0)
        {
            throw std::runtime_error(
                std::format("NumPy byte stride {} in dimension {} is not divisible by item size {}",
                            byte_stride,
                            i,
                            itemsize));
        }
        stride.push_back(byte_stride / itemsize);
        if (shape[i] == 0)
        {
            has_element = false;
            continue;
        }
        ssize_t const axis_byte_offset = (shape[i] - 1) * byte_stride;
        if (axis_byte_offset < 0)
        {
            byte_span_begin += axis_byte_offset;
        }
        else
        {
            byte_span_end += axis_byte_offset;
        }
    }
    if (!has_element)
    {
        byte_span_begin = 0;
        byte_span_end = 0;
    }

    array_order_type array_order = array_order_type::Unspecified;
    if ((arr_in.flags() & py::array::c_style) == py::array::c_style)
    {
        array_order |= array_order_type::CType;
    }
    if ((arr_in.flags() & py::array::f_style) == py::array::f_style)
    {
        array_order |= array_order_type::FType;
    }

    py::array owner = arr_in;
    /*
     * In the following document, it introduces the base object in ndarray.
     * https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.base.html
     * The `array.base` is base object if memory is from some other object.
     * If object owns its memory, base is None.
     */
    while (true)
    {
        const py::object b = owner.attr("base");
        if (b.is_none() || !py::isinstance<py::array>(b))
        {
            break;
        }
        auto next = b.cast<py::array>();
        /*
         * Prevent the infinite loop.
         * For example, the following code will create a loop:
         * nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
         * nparr = nparr[::2, ::2, ::2]
         */
        if (next.ptr() == owner.ptr())
        {
            break;
        }
        owner = next;
    }

    char * view_ptr = static_cast<char *>(arr_in.mutable_data());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    if (reinterpret_cast<std::uintptr_t>(view_ptr) % alignof(value_type) != 0)
    {
        throw std::runtime_error(
            std::format("NumPy data pointer is not aligned for item alignment {}", alignof(value_type)));
    }
    char * storage_ptr = view_ptr + byte_span_begin;
    const size_t storage_nbytes = has_element
                                      ? static_cast<size_t>(byte_span_end - byte_span_begin + itemsize)
                                      : 0;
    const auto data_offset = static_cast<size_t>(-byte_span_begin);
    auto remover = std::make_unique<ConcreteBufferNdarrayRemover>(owner);
    const auto buffer = ConcreteBuffer::construct(storage_nbytes, storage_ptr, std::move(remover));
    return SimpleArray<T>(shape, stride, buffer, data_offset, array_order);
}

} /* end namespace python */
} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
