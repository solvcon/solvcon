/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/plot/RPlotModel.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>

namespace solvcon
{

namespace
{

/**
 * The nonsingular guard, then the autoscale margin: a degenerate span is
 * first opened around its value so the margin has a span to scale.
 */
std::pair<double, double> expand_axis(double lo, double hi, double margin)
{
    if (lo == hi)
    {
        double const half = (lo == 0.0) ? 0.5 : std::abs(lo) * 0.05;
        lo -= half;
        hi += half;
    }

    double const pad = (hi - lo) * margin;
    return {lo - pad, hi + pad};
}

void validate_axis_limits(double lo, double hi, char const * axis)
{
    if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi))
    {
        throw std::invalid_argument(
            std::format(
                "RPlotModel::set_view_limits: {} limits must be finite and increasing, but they are {} and {}",
                axis,
                lo,
                hi));
    }
}

} /* end namespace */

std::shared_ptr<RPlotSeries> RPlotModel::add_series(std::shared_ptr<RPlotSeries> const & series)
{
    if (!series)
    {
        throw std::invalid_argument("RPlotModel::add_series: series must not be None");
    }

    if (!series->color_is_set())
    {
        series->set_color(plot_cycle_color(m_cycle_index++));
    }
    m_series.push_back(series);
    return m_series.back();
}

std::shared_ptr<RPlotSeries> const & RPlotModel::series(std::size_t it) const
{
    if (it >= m_series.size())
    {
        throw std::out_of_range(
            std::format("RPlotModel::series: index {} is out of bounds with size {}", it, m_series.size()));
    }
    return m_series[it];
}

std::optional<std::array<double, 4>> RPlotModel::data_limits() const
{
    std::optional<std::array<double, 4>> limits;
    for (std::shared_ptr<RPlotSeries> const & ser : m_series)
    {
        std::optional<std::array<double, 4>> const serlim = ser->data_limits();
        if (!serlim.has_value())
        {
            continue;
        }
        if (!limits.has_value())
        {
            limits = serlim;
            continue;
        }
        std::array<double, 4> & lim = *limits;
        lim[0] = std::min(lim[0], (*serlim)[0]);
        lim[1] = std::max(lim[1], (*serlim)[1]);
        lim[2] = std::min(lim[2], (*serlim)[2]);
        lim[3] = std::max(lim[3], (*serlim)[3]);
    }
    return limits;
}

void RPlotModel::set_margin(double margin)
{
    if (!std::isfinite(margin) || margin < 0.0)
    {
        throw std::invalid_argument(
            std::format("RPlotModel::set_margin: margin must be finite and non-negative, but it is {}", margin));
    }
    m_margin = margin;
}

void RPlotModel::set_view_limits(double xmin, double xmax, double ymin, double ymax)
{
    validate_axis_limits(xmin, xmax, "x");
    validate_axis_limits(ymin, ymax, "y");
    m_view_limits = {xmin, xmax, ymin, ymax};
}

void RPlotModel::autoscale()
{
    std::optional<std::array<double, 4>> const limits = data_limits();
    if (!limits.has_value())
    {
        m_view_limits = {0.0, 1.0, 0.0, 1.0};
        return;
    }

    auto const [xmin, xmax] = expand_axis((*limits)[0], (*limits)[1], m_margin);
    auto const [ymin, ymax] = expand_axis((*limits)[2], (*limits)[3], m_margin);
    m_view_limits = {xmin, xmax, ymin, ymax};
}

ViewTransform2dFp64 RPlotModel::view(double width, double height) const
{
    if (!std::isfinite(width) || !(width > 0.0) || !std::isfinite(height) || !(height > 0.0))
    {
        throw std::invalid_argument(
            std::format(
                "RPlotModel::view: width and height must be finite and positive, but they are {} and {}",
                width,
                height));
    }

    double const zoom = std::min(width / (m_view_limits[1] - m_view_limits[0]),
                                 height / (m_view_limits[3] - m_view_limits[2]));
    double const center_x = 0.5 * (m_view_limits[0] + m_view_limits[1]);
    double const center_y = 0.5 * (m_view_limits[2] + m_view_limits[3]);

    ViewTransform2dFp64 transform;
    transform.set_zoom(zoom);
    transform.set_pan_x(0.5 * width - zoom * center_x);
    transform.set_pan_y(0.5 * height + zoom * center_y);
    return transform;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
