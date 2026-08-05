/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/plot/RPlotSeries.hpp>

#include <cmath>
#include <format>
#include <stdexcept>

namespace solvcon
{

namespace
{

void validate_operand(SimpleArray<double> const & arr, char const * name)
{
    if (arr.ndim() != 1)
    {
        throw std::invalid_argument(
            std::format("RPlotSeries::set_data: {} must be 1-dimensional, but ndim is {}", name, arr.ndim()));
    }
    if (arr.nbody() > 1 && arr.stride(0) != 1)
    {
        throw std::invalid_argument(
            std::format(
                "RPlotSeries::set_data: {} must be contiguous with unit stride, but stride is {}",
                name,
                arr.stride(0)));
    }
    // The collector clones the whole buffer, so the ghost part would become
    // samples that the nbody-based length check never saw.
    if (arr.nghost() != 0)
    {
        throw std::invalid_argument(
            std::format("RPlotSeries::set_data: {} must be ghost-free, but nghost is {}", name, arr.nghost()));
    }
}

} /* end namespace */

void RPlotSeries::set_data(SimpleArray<double> const & x, SimpleArray<double> const & y)
{
    validate_operand(x, "x");
    validate_operand(y, "y");
    if (x.nbody() != y.nbody())
    {
        throw std::invalid_argument(
            std::format(
                "RPlotSeries::set_data: x and y must have the same length, but they are {} and {}",
                x.nbody(),
                y.nbody()));
    }

    m_x = SimpleCollector<double>(x);
    m_y = SimpleCollector<double>(y);
    m_limits_stale = true;
}

void RPlotSeries::clear_data()
{
    m_x = SimpleCollector<double>();
    m_y = SimpleCollector<double>();
    m_limits_stale = true;
}

double RPlotSeries::x_at(std::size_t it) const
{
    if (it >= size())
    {
        throw std::out_of_range(
            std::format("RPlotSeries::x_at: index {} is out of bounds with size {}", it, size()));
    }
    return x()[it];
}

double RPlotSeries::y_at(std::size_t it) const
{
    if (it >= size())
    {
        throw std::out_of_range(
            std::format("RPlotSeries::y_at: index {} is out of bounds with size {}", it, size()));
    }
    return y()[it];
}

std::optional<std::array<double, 4>> RPlotSeries::data_limits() const
{
    if (!m_limits_stale)
    {
        return m_limits;
    }

    std::optional<std::array<double, 4>> limits;
    std::span<double const> const xs = x();
    std::span<double const> const ys = y();
    for (std::size_t it = 0; it < xs.size(); ++it)
    {
        double const xv = xs[it];
        double const yv = ys[it];
        // Skip the whole sample, not just the offending coordinate: a point
        // with a NaN y is not at a known x either.
        if (!std::isfinite(xv) || !std::isfinite(yv))
        {
            continue;
        }
        if (!limits.has_value())
        {
            limits = std::array<double, 4>{xv, xv, yv, yv};
            continue;
        }
        std::array<double, 4> & lim = *limits;
        if (xv < lim[0])
        {
            lim[0] = xv;
        }
        if (xv > lim[1])
        {
            lim[1] = xv;
        }
        if (yv < lim[2])
        {
            lim[2] = yv;
        }
        if (yv > lim[3])
        {
            lim[3] = yv;
        }
    }

    m_limits = limits;
    m_limits_stale = false;
    return m_limits;
}

void RPlotSeries::set_line_width(double width)
{
    if (!std::isfinite(width) || !(width > 0.0))
    {
        throw std::invalid_argument(
            std::format("RPlotSeries::set_line_width: width must be finite and positive, but it is {}", width));
    }
    m_line_width = width;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
