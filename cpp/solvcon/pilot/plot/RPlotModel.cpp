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

std::optional<PlotLimits2d> RPlotModel::data_limits() const
{
    std::optional<PlotLimits2d> limits;
    for (std::shared_ptr<RPlotSeries> const & ser : m_series)
    {
        std::optional<PlotLimits2d> const serlim = ser->data_limits();
        if (!serlim.has_value())
        {
            continue;
        }
        if (!limits.has_value())
        {
            limits = serlim;
            continue;
        }
        limits->merge(*serlim);
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

void RPlotModel::set_view_limits(PlotLimits2d const & limits)
{
    validate_axis_limits(limits.xmin, limits.xmax, "x");
    validate_axis_limits(limits.ymin, limits.ymax, "y");
    m_view_limits = limits;
}

void RPlotModel::autoscale()
{
    std::optional<PlotLimits2d> const limits = data_limits();
    if (!limits.has_value())
    {
        m_view_limits = PlotLimits2d{0.0, 1.0, 0.0, 1.0};
        return;
    }

    auto const [xmin, xmax] = expand_axis(limits->xmin, limits->xmax, m_margin);
    auto const [ymin, ymax] = expand_axis(limits->ymin, limits->ymax, m_margin);
    m_view_limits = PlotLimits2d{xmin, xmax, ymin, ymax};
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

    double const zoom = std::min(width / (m_view_limits.xmax - m_view_limits.xmin),
                                 height / (m_view_limits.ymax - m_view_limits.ymin));
    double const center_x = 0.5 * (m_view_limits.xmin + m_view_limits.xmax);
    double const center_y = 0.5 * (m_view_limits.ymin + m_view_limits.ymax);

    ViewTransform2dFp64 transform;
    transform.set_zoom(zoom);
    transform.set_pan_x(0.5 * width - zoom * center_x);
    transform.set_pan_y(0.5 * height + zoom * center_y);
    return transform;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
