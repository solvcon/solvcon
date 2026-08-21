/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/timeseries/pymod/timeseries_pymod.hpp>

#include <pybind11/stl.h>

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

    detail::for_each_type(
        detail::timeseries_value_types{},
        [&mod]<typename T>()
        {
            mod.def(
                "dedup_last",
                &timeseries::dedup_last<T>,
                py::arg("times"),
                py::arg("values"),
                "Keep the last sample of every group of equal timestamps; returns (times, values)");
        });

    detail::for_each_type(
        detail::timeseries_real_types{},
        [&mod]<typename T>()
        {
            mod.def(
                "deriv",
                &timeseries::deriv<T>,
                py::arg("times"),
                py::arg("values"),
                "Differentiate a series by the backward difference; returns (times[1:], derivatives)");
            mod.def(
                "movavg",
                &timeseries::movavg<T>,
                py::arg("times"),
                py::arg("values"),
                py::arg("span"),
                "Average a series over the trailing half-open window (t - span, t]; returns (times, means)");
        });

    mod.def(
        "held",
        &timeseries::held,
        py::arg("times"),
        py::arg("values"),
        py::arg("span"),
        "Report whether a boolean series was true over the trailing half-open window (t - span, t]; "
        "the last sample at or before t - span must be true as well; returns (times, answers)");

    mod.def(
        "true_intervals",
        &timeseries::true_intervals,
        py::arg("times"),
        py::arg("values"),
        "Run-length encode the true stretches of a boolean series into rows of (start, end, duration); "
        "a run still open at the last sample ends there");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
