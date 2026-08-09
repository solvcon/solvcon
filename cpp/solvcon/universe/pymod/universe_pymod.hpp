#pragma once

/*
 * Copyright (c) 2023, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h> // Must be the first include.
#include <pybind11/stl.h>

#include <solvcon/python/common.hpp>
#include <solvcon/universe/universe.hpp>

#include <format>
#include <stdexcept>

namespace solvcon
{

namespace python
{

/**
 * Turn a Python index into a pad offset, counting a negative index from the
 * end of the pad.
 *
 * @param index the index as Python spells it
 * @param size the number of elements in the pad
 * @param name the pad type name the error message carries
 *
 * @throw std::out_of_range if the normalized index falls outside the pad
 */
inline size_t normalize_pad_index(ssize_t index, size_t size, char const * name)
{
    auto const nelem = static_cast<ssize_t>(size);
    ssize_t const it = index < 0 ? index + nelem : index;
    if (it < 0 || it >= nelem)
    {
        throw std::out_of_range(
            std::format("{}: index {} is out of bounds with size {}", name, index, size));
    }
    return static_cast<size_t>(it);
}

/**
 * Copy the elements a Python slice selects into a new pad of the same type
 * and dimensionality.
 *
 * A pad owns one array per axis and carries neither an offset nor a stride,
 * so the result copies the selected elements instead of viewing them. The
 * result is unaligned, because an alignment the slice length does not divide
 * cannot be honored.
 *
 * @param self the pad to read
 * @param key the Python slice to apply
 */
template <typename Pad>
std::shared_ptr<Pad> copy_pad_slice(Pad const & self, pybind11::slice const & key)
{
    ssize_t start = 0, stop = 0, step = 0, slicelength = 0;
    if (!key.compute(static_cast<ssize_t>(self.size()), &start, &stop, &step, &slicelength))
    {
        throw pybind11::error_already_set();
    }

    std::shared_ptr<Pad> ret = Pad::construct(self.ndim(), static_cast<size_t>(slicelength));
    for (ssize_t i = 0; i < slicelength; ++i)
    {
        ret->set_at(static_cast<size_t>(i), self.get_at(static_cast<size_t>(start)));
        start += step;
    }
    return ret;
}

void initialize_universe(pybind11::module & mod);
void wrap_shape0d(pybind11::module & mod);
void wrap_shape1d(pybind11::module & mod);
void wrap_shape2d(pybind11::module & mod);
void wrap_shape3d(pybind11::module & mod);
void wrap_view_transform2d(pybind11::module & mod);
void wrap_World(pybind11::module & mod);

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: