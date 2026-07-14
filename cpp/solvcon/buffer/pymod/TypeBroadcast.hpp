#pragma once

/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/SimpleArray.hpp>
#include <solvcon/math/math.hpp>
#include <solvcon/python/common.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h> // Must be the first include.

namespace solvcon
{
namespace python
{

template <typename T /* original type */, typename D /* for destination type */>
struct TypeBroadcastImpl
{
    using shape_type = solvcon::detail::shape_type;
    using sshape_type = solvcon::detail::sshape_type;

    // NOLINTNEXTLINE(misc-no-recursion)
    static void copy_idx(SimpleArray<T> & arr_out,
                         std::vector<sshape_type> const & slices,
                         pybind11::array_t<D> const * arr_in,
                         shape_type const & left_shape,
                         sshape_type sidx,
                         ssize_t dim)
    {
        using out_type = typename std::remove_reference_t<decltype(arr_out[0])>;

        if (dim < 0)
        {
            return;
        }

        auto const axis = static_cast<size_t>(dim);
        ssize_t const length = left_shape[axis];
        for (ssize_t i = 0; i < length; ++i)
        {
            sidx[axis] = i;

            ssize_t offset_in = 0;
            pybind11::ssize_t const ndim_in = arr_in->ndim();
            for (pybind11::ssize_t py_axis = 0; py_axis < ndim_in; ++py_axis)
            {
                auto const axis_in = static_cast<size_t>(py_axis);
                offset_in += arr_in->strides(py_axis) / arr_in->itemsize() * sidx[axis_in];
            }
            const D * ptr_in = arr_in->data() + offset_in;

            ssize_t offset_out = 0;
            ssize_t const ndim = arr_out.ndim();
            for (ssize_t it = 0; it < ndim; ++it)
            {
                auto const axis_index = static_cast<size_t>(it);
                ssize_t const step = slices[axis_index][2];
                offset_out += arr_out.stride(it) * sidx[axis_index] * step;
            }

            constexpr bool valid_conversion = (!is_complex_v<T> && !is_complex_v<D>) || (is_complex_v<T> && is_complex_v<D> && std::is_same_v<T, D>);

            if constexpr (valid_conversion)
            {
                auto * ptr_out = arr_out.data() + offset_out;
                // FIXME: NOLINTNEXTLINE(bugprone-signed-char-misuse,cert-str34-c)
                *ptr_out = static_cast<out_type>(*ptr_in);
            }
            else
            {
                throw std::runtime_error("Cannot convert between complex and non-complex types");
            }

            // recursion here
            copy_idx(arr_out, slices, arr_in, left_shape, sidx, dim - 1);
        }
    }

    static void broadcast(SimpleArray<T> & arr_out,
                          std::vector<sshape_type> const & slices,
                          pybind11::array const & arr_in)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto * arr_new = reinterpret_cast<pybind11::array_t<D> const *>(&arr_in);

        ssize_t const ndim = arr_out.ndim();
        shape_type left_shape(ndim);
        for (ssize_t axis = 0; axis < ndim; ++axis)
        {
            sshape_type const & slice = slices[axis];
            if ((slice[1] - slice[0]) % slice[2] == 0)
            {
                left_shape[axis] = (slice[1] - slice[0]) / slice[2];
            }
            else
            {
                left_shape[axis] = (slice[1] - slice[0]) / slice[2] + 1;
            }
        }

        sshape_type const sidx_init(ndim, 0);

        copy_idx(arr_out, slices, arr_new, left_shape, sidx_init, ndim - 1);
    }
}; /* end struct TypeBroadcastImpl */

template <typename T>
struct TypeBroadcast
{
    using shape_type = solvcon::detail::shape_type;
    using sshape_type = solvcon::detail::sshape_type;

    static void check_shape(SimpleArray<T> const & arr_out,
                            std::vector<sshape_type> const & slices,
                            pybind11::array const & arr_in)
    {
        pybind11::ssize_t const right_ndim = arr_in.ndim();
        shape_type right_shape(right_ndim);
        for (pybind11::ssize_t axis = 0; axis < right_ndim; ++axis)
        {
            right_shape[axis] = arr_in.shape(axis);
        }

        ssize_t const ndim = arr_out.ndim();
        shape_type left_shape(ndim);
        // TODO: range check
        for (ssize_t axis = 0; axis < ndim; ++axis)
        {
            sshape_type const & slice = slices[axis];
            if ((slice[1] - slice[0]) % slice[2] == 0)
            {
                left_shape[axis] = (slice[1] - slice[0]) / slice[2];
            }
            else
            {
                left_shape[axis] = (slice[1] - slice[0]) / slice[2] + 1;
            }
        }

        if (arr_out.ndim() != arr_in.ndim())
        {
            throw_shape_error(left_shape, right_shape);
        }

        for (ssize_t i = 0; i < ndim; ++i)
        {
            if (left_shape[i] != right_shape[i])
            {
                throw_shape_error(left_shape, right_shape);
            }
        }
    }

    static void broadcast(SimpleArray<T> & arr_out,
                          std::vector<sshape_type> const & slices,
                          pybind11::array const & arr_in)
    {
        if (dtype_is_type<bool>(arr_in))
        {
            TypeBroadcastImpl<T, bool>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<int8_t>(arr_in))
        {
            TypeBroadcastImpl<T, int8_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<int16_t>(arr_in))
        {
            TypeBroadcastImpl<T, int16_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<int32_t>(arr_in))
        {
            TypeBroadcastImpl<T, int32_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<int64_t>(arr_in))
        {
            TypeBroadcastImpl<T, int64_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<uint8_t>(arr_in))
        {
            TypeBroadcastImpl<T, uint8_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<uint16_t>(arr_in))
        {
            TypeBroadcastImpl<T, uint16_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<uint32_t>(arr_in))
        {
            TypeBroadcastImpl<T, uint32_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<uint64_t>(arr_in))
        {
            TypeBroadcastImpl<T, uint64_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<float>(arr_in))
        {
            TypeBroadcastImpl<T, float>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<double>(arr_in))
        {
            TypeBroadcastImpl<T, double>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<Complex<float>>(arr_in))
        {
            TypeBroadcastImpl<T, Complex<float>>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<Complex<double>>(arr_in))
        {
            TypeBroadcastImpl<T, Complex<double>>::broadcast(arr_out, slices, arr_in);
        }
        else
        {
            throw std::runtime_error("input array data type not support!");
        }
    }

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    static void throw_shape_error(shape_type const & left_shape, shape_type const & right_shape)
    {

        std::ostringstream msg;
        msg << "Broadcast input array from shape(";
        for (size_t i = 0; i < right_shape.size(); ++i)
        {
            msg << right_shape[i];
            if (i != right_shape.size() - 1)
            {
                msg << ", ";
            }
        }
        msg << ") into shape(";
        for (size_t i = 0; i < left_shape.size(); ++i)
        {
            msg << left_shape[i];
            if (i != left_shape.size() - 1)
            {
                msg << ", ";
            }
        }
        msg << ")";

        throw std::runtime_error(msg.str());
    }
}; /* end struct TypeBroadCast */

} /* end namespace python */
} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
