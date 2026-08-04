/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/stl.h> // Must be the first include.

#include <solvcon/pilot/wrap_pilot.hpp> // Must be the first include but give way to above.
#include <solvcon/python/common.hpp>

#include <pybind11/operators.h>

#include <solvcon/buffer/pymod/SimpleArrayCaster.hpp>

#include <solvcon/pilot/plot/plot_style.hpp>
#include <solvcon/pilot/plot/RPlotSeries.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace solvcon
{

namespace python
{

namespace
{

/// A 4-tuple, not the list that an auto-cast std::array<double, 4> would give.
pybind11::object limits_to_python(std::optional<std::array<double, 4>> const & limits)
{
    if (!limits.has_value())
    {
        return pybind11::none();
    }
    std::array<double, 4> const & lim = *limits;
    return pybind11::make_tuple(lim[0], lim[1], lim[2], lim[3]);
}

/**
 * Reject a negative index with IndexError. A std::size_t parameter would raise
 * TypeError instead, and the unsigned wrap-around would turn -1 into a huge
 * in-range-looking index.
 */
std::size_t checked_index(std::int64_t index, std::size_t count, char const * what, char const * noun)
{
    if (index < 0 || static_cast<std::size_t>(index) >= count)
    {
        throw std::out_of_range(std::format("{}: index {} is out of bounds with {} {}", what, index, noun, count));
    }
    return static_cast<std::size_t>(index);
}

} /* end namespace */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapPlotColor
    : public WrapBase<WrapPlotColor, PlotColor>
{

    friend root_base_type;

    WrapPlotColor(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def(py::init<>())
            .def(
                py::init<std::uint8_t, std::uint8_t, std::uint8_t, std::uint8_t>(),
                py::arg("r"),
                py::arg("g"),
                py::arg("b"),
                py::arg("a") = 255)
            //
            ;

        // Read-only, not writable: every function yielding a PlotColor yields
        // a copy, so `plot_cycle_color(0).a = 128` would recolor nothing.
        (*this)
            .def_readonly("r", &wrapped_type::r)
            .def_readonly("g", &wrapped_type::g)
            .def_readonly("b", &wrapped_type::b)
            .def_readonly("a", &wrapped_type::a)
            .def(py::self == py::self) // NOLINT(misc-redundant-expression)
            .def(py::self != py::self) // NOLINT(misc-redundant-expression)
            .def(
                "__hash__",
                [](wrapped_type const & self)
                {
                    return static_cast<py::ssize_t>(
                        (static_cast<std::uint32_t>(self.a) << 24U) | (static_cast<std::uint32_t>(self.b) << 16U) |
                        (static_cast<std::uint32_t>(self.g) << 8U) | static_cast<std::uint32_t>(self.r));
                })
            .def(
                "__repr__",
                [](wrapped_type const & self)
                {
                    return std::format(
                        "PlotColor(r={}, g={}, b={}, a={})",
                        static_cast<int>(self.r),
                        static_cast<int>(self.g),
                        static_cast<int>(self.b),
                        static_cast<int>(self.a));
                })
            //
            ;
    }

}; /* end class WrapPlotColor */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapRPlotSeries
    : public WrapBase<WrapRPlotSeries, RPlotSeries, std::shared_ptr<RPlotSeries>>
{

    friend root_base_type;

    WrapRPlotSeries(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def(py::init<>())
            //
            ;

        (*this)
            .def("set_data", &wrapped_type::set_data, py::arg("x"), py::arg("y"))
            .def("clear_data", &wrapped_type::clear_data)
            .def_property_readonly("size", &wrapped_type::size)
            .def("__len__", &wrapped_type::size)
            .def(
                "x",
                [](wrapped_type const & self, std::int64_t index)
                { return self.x_at(checked_index(index, self.size(), "RPlotSeries::x_at", "size")); },
                py::arg("index"))
            .def(
                "y",
                [](wrapped_type const & self, std::int64_t index)
                { return self.y_at(checked_index(index, self.size(), "RPlotSeries::y_at", "size")); },
                py::arg("index"))
            .def(
                "data_limits",
                [](wrapped_type const & self)
                { return limits_to_python(self.data_limits()); })
            .def_property("label", &wrapped_type::label, &wrapped_type::set_label)
            // A copy, not a reference into the series: `series.color.a = 128`
            // must fail rather than write to a temporary.
            .def_property("color", &wrapped_type::color, &wrapped_type::set_color, py::return_value_policy::copy)
            .def_property_readonly("color_is_set", &wrapped_type::color_is_set)
            .def_property("line_width", &wrapped_type::line_width, &wrapped_type::set_line_width)
            //
            ;
    }

}; /* end class WrapRPlotSeries */

void wrap_plot(pybind11::module & mod)
{
    namespace py = pybind11;

    WrapPlotColor::commit(
        mod,
        "PlotColor",
        "One sRGB color with alpha, a byte per channel. An immutable value: "
        "the channels are read-only, so rebuild the color to change one.");
    WrapRPlotSeries::commit(
        mod,
        "RPlotSeries",
        "One xy data series: a copy of a contiguous SimpleArrayFloat64 pair "
        "plus the style used to stroke it. Samples are read through "
        "size / x / y.");

    mod.def(
        "plot_color_cycle",
        []()
        {
            std::span<PlotColor const> const cycle = plot_color_cycle();
            return std::vector<PlotColor>(cycle.begin(), cycle.end());
        });
    mod.def("plot_cycle_color", &plot_cycle_color, py::arg("index"));
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
