#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The series list of one xy plot: the color cycle a new series draws from,
 * the aggregate data limits, and the view limits that autoscale derives and
 * view() maps onto the screen. Qt-free, so it compiles into the no-GUI test
 * target.
 *
 * @ingroup group_domain
 */

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include <solvcon/universe/ViewTransform2d.hpp>

#include <solvcon/pilot/plot/plot_style.hpp>
#include <solvcon/pilot/plot/PlotLimits2d.hpp>
#include <solvcon/pilot/plot/RPlotSeries.hpp>

namespace solvcon
{

/**
 * The model of naive xy plot: it owns the series list, colors a new
 * series from the matplotlib color cycle, unions the per-series data limits,
 * and autoscales the view limits that view() maps onto the screen.
 */
class RPlotModel
{
public:

    RPlotModel() = default;
    RPlotModel(RPlotModel const &) = default;
    RPlotModel(RPlotModel &&) = default;
    RPlotModel & operator=(RPlotModel const &) = default;
    RPlotModel & operator=(RPlotModel &&) = default;
    ~RPlotModel() = default;

    /**
     * Register a series, coloring it from the cycle when it carries no
     * explicit color, and return it. An explicitly colored series does not
     * consume a cycle slot.
     */
    std::shared_ptr<RPlotSeries> add_series(std::shared_ptr<RPlotSeries> const & series);

    std::shared_ptr<RPlotSeries> add_series() { return add_series(std::make_shared<RPlotSeries>()); }

    std::size_t size() const { return m_series.size(); }

    std::shared_ptr<RPlotSeries> const & series(std::size_t it) const;

    /**
     * The union of every series' data limits; nullopt when no series holds a
     * finite sample.
     */
    std::optional<PlotLimits2d> data_limits() const;

    double margin() const { return m_margin; }

    void set_margin(double margin);

    PlotLimits2d view_limits() const { return m_view_limits; }

    /**
     * Set the view limits and hold them: auto_view() turns false, so a front
     * end stops autoscaling over them.
     */
    void set_view_limits(PlotLimits2d const & limits);

    /**
     * Whether the view follows the data. It is true until the view limits
     * are set, zoomed, or panned, and true again after reset_view().
     */
    bool auto_view() const { return m_auto_view; }

    /**
     * Set the view limits to the data limits opened by the margin, guarding
     * a singular span first so the margin has a span to scale. Without any
     * finite sample the view falls back to the unit square.
     */
    void autoscale();

    /// Follow the data again: autoscale and turn auto_view() back on.
    void reset_view();

    /**
     * Zoom the view limits about an anchor in data coordinates and hold
     * them. A factor above one zooms in, as in ViewTransform2d::zoom_at, and
     * a factor of one leaves its axis untouched. An axis that would lose its
     * resolution or overflow keeps its limits.
     */
    void zoom_view(double factor_x, double factor_y, double anchor_x, double anchor_y);

    /**
     * Move the view limits by (dx, dy) in data units and hold them. An axis
     * that would overflow keeps its limits.
     */
    void pan_view(double dx, double dy);

    /**
     * The transform that maps the view limits onto a width-by-height
     * screen. ViewTransform2d carries a single zoom, so the view box fits
     * inside the screen and is centered; a per-axis stretch is the
     * widget's later concern.
     */
    ViewTransform2dFp64 view(double width, double height) const;

private:

    std::vector<std::shared_ptr<RPlotSeries>> m_series;
    std::size_t m_cycle_index = 0;
    double m_margin = PLOT_DEFAULT_MARGIN;
    PlotLimits2d m_view_limits = {0.0, 1.0, 0.0, 1.0};
    bool m_auto_view = true;
}; /* end class RPlotModel */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
