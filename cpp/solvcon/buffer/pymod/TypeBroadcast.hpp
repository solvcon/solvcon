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

#include <algorithm>
#include <cstring>
#include <functional>
#include <type_traits>

namespace solvcon
{
namespace python
{

namespace detail
{

using shape_type = solvcon::detail::shape_type;

template <typename Function>
void for_each_index(shape_type const & shape, Function const & function)
{
    size_t count = 1;
    for (ssize_t const length : shape)
    {
        count *= static_cast<size_t>(length);
    }

    shape_type index(shape.size(), 0);
    for (size_t step = 0; step < count; ++step)
    {
        function(index);
        size_t axis = 0;
        while (axis < shape.size() && ++index[axis] == shape[axis])
        {
            index[axis++] = 0;
        }
    }
}

inline shape_type shape_from_slices(std::vector<shape_type> const & slices)
{
    shape_type shape(slices.size());
    for (size_t axis = 0; axis < slices.size(); ++axis)
    {
        shape[axis] = slices[axis][3];
    }
    return shape;
}

} /* end namespace detail */

template <typename T, typename D>
struct TypeBroadcastImpl
{
    using shape_type = solvcon::detail::shape_type;
    using slices_type = std::vector<shape_type>;

    static D input_at(char const * data, pybind11::ssize_t const * strides, shape_type const & sidx);
    static T & output_at(SimpleArray<T> & arr_out, slices_type const & slices, shape_type const & sidx);
    static bool may_overlap(SimpleArray<T> const & arr_out, pybind11::array const & arr_in);
    static void broadcast(SimpleArray<T> & arr_out, slices_type const & slices, pybind11::array const & arr_in);
}; /* end struct TypeBroadcastImpl */

template <typename T, typename D>
D TypeBroadcastImpl<T, D>::input_at(char const * data, pybind11::ssize_t const * strides, shape_type const & sidx)
{
    static_assert(std::is_trivially_copyable_v<D>);
    for (size_t axis = 0; axis < sidx.size(); ++axis)
    {
        data += strides[axis] * sidx[axis];
    }
    D value{};
    std::memcpy(&value, data, sizeof(value));
    return value;
}

template <typename T, typename D>
T & TypeBroadcastImpl<T, D>::output_at(
    SimpleArray<T> & arr_out, slices_type const & slices, shape_type const & sidx)
{
    ssize_t offset = 0;
    for (ssize_t axis = 0; axis < arr_out.ndim(); ++axis)
    {
        ssize_t const index = slices[axis][0] + sidx[axis] * slices[axis][2];
        offset += arr_out.stride(axis) * index;
    }
    return arr_out.logical_data()[offset];
}

template <typename T, typename D>
bool TypeBroadcastImpl<T, D>::may_overlap(SimpleArray<T> const & arr_out, pybind11::array const & arr_in)
{
    if (!arr_out || arr_in.size() == 0)
    {
        return false;
    }

    ssize_t byte_begin = 0;
    ssize_t byte_end = sizeof(D);
    for (pybind11::ssize_t axis = 0; axis < arr_in.ndim(); ++axis)
    {
        ssize_t const offset = (arr_in.shape(axis) - 1) * arr_in.strides(axis);
        byte_begin += std::min<ssize_t>(0, offset);
        byte_end += std::max<ssize_t>(0, offset);
    }

    auto const * input = static_cast<char const *>(arr_in.data());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto const * output = reinterpret_cast<char const *>(arr_out.buffer().data());
    std::less<> const address_less;
    return address_less(input + byte_begin, output + arr_out.buffer().nbytes()) &&
           address_less(output, input + byte_end);
}

template <typename T, typename D>
void TypeBroadcastImpl<T, D>::broadcast(
    SimpleArray<T> & arr_out, slices_type const & slices, pybind11::array const & arr_in)
{
    if (arr_in.size() == 0)
    {
        return;
    }

    constexpr bool valid_conversion =
        (!is_complex_v<T> && !is_complex_v<D>) ||
        (is_complex_v<T> && is_complex_v<D> && std::is_same_v<T, D>);

    if constexpr (valid_conversion)
    {
        shape_type const output_shape = detail::shape_from_slices(slices);
        auto const * data = static_cast<char const *>(arr_in.data());
        auto const * strides = arr_in.strides();
        if (may_overlap(arr_out, arr_in))
        {
            SimpleArray<T> staged(arr_in.size());
            size_t staged_index = 0;
            detail::for_each_index(output_shape, [&](shape_type const & sidx)
                                   {
                                       staged[staged_index++] = static_cast<T>(
                                           input_at(data, strides, sidx)); // FIXME: NOLINT(bugprone-signed-char-misuse,cert-str34-c)
                                   });
            staged_index = 0;
            detail::for_each_index(output_shape, [&](shape_type const & sidx)
                                   { output_at(arr_out, slices, sidx) = staged[staged_index++]; });
        }
        else
        {
            detail::for_each_index(output_shape, [&](shape_type const & sidx)
                                   {
                                       output_at(arr_out, slices, sidx) = static_cast<T>(
                                           input_at(data, strides, sidx)); // FIXME: NOLINT(bugprone-signed-char-misuse,cert-str34-c)
                                   });
        }
    }
    else
    {
        throw std::runtime_error("Cannot convert between complex and non-complex types");
    }
}

template <typename T>
struct TypeBroadcast
{
    using shape_type = solvcon::detail::shape_type;
    using slices_type = std::vector<shape_type>;

    static void check_shape(SimpleArray<T> const & arr_out, slices_type const & slices, pybind11::array const & arr_in)
    {
        pybind11::ssize_t const right_ndim = arr_in.ndim();
        shape_type right_shape(right_ndim);
        for (pybind11::ssize_t axis = 0; axis < right_ndim; ++axis)
        {
            right_shape[axis] = arr_in.shape(axis);
        }

        ssize_t const ndim = arr_out.ndim();
        shape_type const left_shape = detail::shape_from_slices(slices);

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

    static void broadcast(SimpleArray<T> & arr_out, slices_type const & slices, pybind11::array const & arr_in)
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
}; /* end struct TypeBroadcast */

} /* end namespace python */
} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
