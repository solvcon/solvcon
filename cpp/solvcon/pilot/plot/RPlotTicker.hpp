#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Round tick positions and labels for the axes of a native xy plot. Qt-free,
 * so it compiles into the no-GUI test target.
 *
 * @ingroup group_domain
 */

#include <cstddef>
#include <string>
#include <vector>

#include <solvcon/buffer/small_vector.hpp>

namespace solvcon
{

/**
 * Locate 1-2-5 ticks over a linear range, or whole decades over a log range,
 * and format their short labels.
 */
class RPlotTicker
{
public:

    using ticks_type = small_vector<double, 16>;
    using labels_type = std::vector<std::string>;

    explicit RPlotTicker(std::size_t target_count);
    RPlotTicker(RPlotTicker const &) = default;
    RPlotTicker(RPlotTicker &&) = default;
    RPlotTicker & operator=(RPlotTicker const &) = default;
    RPlotTicker & operator=(RPlotTicker &&) = default;
    ~RPlotTicker() = default;

    std::size_t target_count() const { return m_target_count; }
    void set_target_count(std::size_t count);

    ticks_type locate(double lo, double hi) const;
    ticks_type locate_decades(double lo, double hi) const;

    /// Label increasing ticks in one notation, with the digits their spacing needs, so neighbours never read alike.
    labels_type labels(ticks_type const & ticks) const;
    /// Label log-axis ticks: whole exponents as powers of ten, fractional ends by value, neighbours kept distinct.
    labels_type decade_labels(ticks_type const & ticks) const;

private:

    std::size_t m_target_count = 0;
}; /* end class RPlotTicker */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
