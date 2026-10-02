#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Pick the samples of a dense xy series that its pixel columns can show, so the
 * polyline drawn for it is bounded by the plot width instead of the sample
 * count. Qt-free, so it compiles into the no-GUI test target.
 *
 * @ingroup group_domain
 */

#include <cstddef>
#include <cstdint>
#include <limits>

#include <solvcon/buffer/SimpleArray.hpp>

#include <solvcon/pilot/plot/RPlotSeries.hpp>

namespace solvcon
{

/**
 * Reduce a series to the first, lowest, highest, and last sample of each pixel
 * column: the M4 aggregation of Jugel et al., "M4: A Visualization-Oriented
 * Time Series Data Aggregation", PVLDB 7(10), 2014
 * (https://www.vldb.org/pvldb/vol7/p797-jugel.pdf). The line through them keeps
 * the vertical extent of every column and joins neighbouring columns where the
 * whole series does, with at most four points per column.
 */
class RPlotDecimator
{
public:

    using indices_type = SimpleArray<uint64_t>;

    RPlotDecimator() = default;
    RPlotDecimator(RPlotDecimator const &) = default;
    RPlotDecimator(RPlotDecimator &&) = default;
    RPlotDecimator & operator=(RPlotDecimator const &) = default;
    RPlotDecimator & operator=(RPlotDecimator &&) = default;
    ~RPlotDecimator() = default;

    /**
     * Select the samples of a series to draw over evenly split pixel columns.
     *
     * A sample is drawable when both its coordinates are finite and its y is
     * above @p y_floor. Outside [@p xmin, @p xmax] only the drawable sample
     * nearest each end is selected, to carry the line into and out of the
     * view. Every drawable sample is selected when there are no more than four
     * per column, when x decreases anywhere, or when the range places no
     * column.
     *
     * @param series The series to select from.
     * @param xmin The abscissa at the left edge of the first column.
     * @param xmax The abscissa at the right edge of the last column.
     * @param columns The number of pixel columns, at least 1.
     * @param y_floor The ordinate at or below which a sample is not drawable.
     * @return The indices of the selected samples, in increasing order.
     */
    indices_type select(
        RPlotSeries const & series,
        double xmin,
        double xmax,
        std::size_t columns,
        double y_floor = -std::numeric_limits<double>::infinity()) const;

}; /* end class RPlotDecimator */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
