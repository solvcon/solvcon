/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/timeseries/pymod/timeseries_pymod.hpp>

namespace solvcon
{

namespace python
{

void wrap_timeseries(pybind11::module & mod)
{
    namespace py = pybind11;

    mod.def(
        "merge_sorted_unique",
        [](py::args const & args)
        {
            small_vector<SimpleArray<uint64_t> const *> arrays;
            for (py::handle const arg : args)
            {
                if (!py::isinstance<SimpleArray<uint64_t>>(arg))
                {
                    throw py::type_error(std::format(
                        "timeseries::merge_sorted_unique(): every array must be SimpleArrayUint64 but got {}",
                        py::str(py::type::of(arg).attr("__name__")).cast<std::string>()));
                }
                arrays.push_back(&arg.cast<SimpleArray<uint64_t> const &>());
            }
            return timeseries::merge_sorted_unique(arrays);
        },
        "Merge sorted SimpleArrayUint64 timestamp arrays into one sorted array of the distinct timestamps");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
