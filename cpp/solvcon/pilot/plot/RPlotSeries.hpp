#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * One xy data series of the native plot: the samples, the stroke style, and the
 * raw data limits. Qt-free, so it compiles into the no-GUI test target.
 *
 * @ingroup group_domain
 */

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include <solvcon/buffer/SimpleArray.hpp>
#include <solvcon/buffer/SimpleCollector.hpp>

#include <solvcon/pilot/plot/plot_style.hpp>

namespace solvcon
{

class RPlotSeries
{
public:

    RPlotSeries() = default;
    RPlotSeries(RPlotSeries const &) = default;
    RPlotSeries(RPlotSeries &&) = default;
    RPlotSeries & operator=(RPlotSeries const &) = default;
    RPlotSeries & operator=(RPlotSeries &&) = default;
    ~RPlotSeries() = default;

    void set_data(SimpleArray<double> const & x, SimpleArray<double> const & y);

    void clear_data();

    std::size_t size() const { return m_x.size(); }

    std::span<double const> x() const { return std::span<double const>(m_x.data(), size()); }
    std::span<double const> y() const { return std::span<double const>(m_y.data(), size()); }

    double x_at(std::size_t it) const;

    double y_at(std::size_t it) const;

    std::optional<std::array<double, 4>> data_limits() const;

    std::string const & label() const { return m_label; }
    void set_label(std::string label) { m_label = std::move(label); }

    PlotColor color() const { return m_color; }

    void set_color(PlotColor color)
    {
        m_color = color;
        m_color_is_set = true;
    }

    bool color_is_set() const { return m_color_is_set; }

    double line_width() const { return m_line_width; }

    void set_line_width(double width);

private:

    SimpleCollector<double> m_x;
    SimpleCollector<double> m_y;
    std::string m_label;
    PlotColor m_color;
    bool m_color_is_set = false;
    double m_line_width = PLOT_DEFAULT_LINE_WIDTH;

    mutable bool m_limits_stale = true;
    mutable std::optional<std::array<double, 4>> m_limits;
}; /* end class RPlotSeries */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
