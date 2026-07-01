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
};

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
};

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
    using sshape_type = solvcon::detail::sshape_type;

    static void broadcast_array_using_ellipsis(SimpleArray<T> & arr_out, pybind11::array const & arr_in)
    {
        auto slices = make_default_slices(arr_out);

        TypeBroadcast<T>::check_shape(arr_out, slices, arr_in);

        const ssize_t nghost = arr_out.nghost();
        if (0 != nghost)
        {
            arr_out.set_nghost(0);
        }

        TypeBroadcast<T>::broadcast(arr_out, slices, arr_in);

        if (0 != nghost)
        {
            arr_out.set_nghost(nghost);
        }
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
                process_slices(tuple_in, slices, arr_out.ndim());

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
                copy_slice(slices[0], slice_in);

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
            array.data(), /* Pointer to buffer */
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

    static std::vector<sshape_type> make_default_slices(SimpleArray<T> const & arr)
    {
        std::vector<sshape_type> slices;
        auto const & shape = arr.shape();
        slices.reserve(shape.size());
        for (ssize_t const dim : shape)
        {
            sshape_type default_slice(3);
            default_slice[0] = 0; // start
            default_slice[1] = dim; // stop
            default_slice[2] = 1; // step
            slices.push_back(std::move(default_slice));
        }
        return slices;
    }

    static void copy_slice(sshape_type & slice_out, pybind11::slice const & slice_in)
    {
        auto start = std::string(pybind11::str(slice_in.attr("start")));
        auto stop = std::string(pybind11::str(slice_in.attr("stop")));
        auto step = std::string(pybind11::str(slice_in.attr("step")));

        slice_out[0] = start == "None" ? slice_out[0] : std::stoi(start);
        slice_out[1] = stop == "None" ? slice_out[1] : std::stoi(stop);
        slice_out[2] = step == "None" ? slice_out[2] : std::stoi(step);
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

        if (ellipsis_cnt + slice_cnt > ndim)
        {
            throw std::runtime_error("syntax error. dimensions mismatches");
        }

        if (ellipsis_cnt > 1)
        {
            throw std::runtime_error("syntax error. no more than one ellipsis.");
        }
    }

    static void process_slices(pybind11::tuple const & tuple,
                               std::vector<sshape_type> & slices,
                               ssize_t ndim)
    {
        namespace py = pybind11;

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

            auto & slice_out = slices[it - tuple.begin()];
            const auto slice_in = (*it).cast<py::slice>();

            copy_slice(slice_out, slice_in);
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
                auto & slice_out = slices[ndim - offset - 1];
                const auto slice_in = (*it).cast<py::slice>();

                copy_slice(slice_out, slice_in);
            }
        }
    }

    static void broadcast_array_using_slice(SimpleArray<T> & arr_out,
                                            std::vector<sshape_type> const & slices,
                                            pybind11::array const & arr_in)
    {
        TypeBroadcast<T>::check_shape(arr_out, slices, arr_in);

        const ssize_t nghost = arr_out.nghost();
        if (0 != nghost)
        {
            arr_out.set_nghost(0);
        }

        TypeBroadcast<T>::broadcast(arr_out, slices, arr_in);

        if (0 != nghost)
        {
            arr_out.set_nghost(nghost);
        }
    }
};

} /* end namespace python */
} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
