/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/plot/RPlotDecimator.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

#include <solvcon/buffer/SimpleCollector.hpp>

namespace solvcon
{

namespace
{

/// M4 selects at most this many samples per column, so a series no denser has nothing to drop.
constexpr std::size_t M4_SAMPLES_PER_COLUMN = 4;

bool is_drawable(double x, double y, double y_floor) { return std::isfinite(x) && std::isfinite(y) && y > y_floor; }

/// The first, lowest, highest, and last drawable sample of one pixel column, as series indices.
struct ColumnPicks
{
    std::size_t first;
    std::size_t lowest;
    std::size_t highest;
    std::size_t last;
}; /* end struct ColumnPicks */

void widen(ColumnPicks & picks, std::span<double const> ys, std::size_t it)
{
    if (ys[it] < ys[picks.lowest])
    {
        picks.lowest = it;
    }
    if (ys[it] > ys[picks.highest])
    {
        picks.highest = it;
    }
    picks.last = it;
}

void append(SimpleCollector<uint64_t> & selected, ColumnPicks const & picks)
{
    // In sample order, each index once: first <= low <= high <= last.
    std::size_t const low = std::min(picks.lowest, picks.highest);
    std::size_t const high = std::max(picks.lowest, picks.highest);
    selected.push_back(picks.first);
    if (low > picks.first)
    {
        selected.push_back(low);
    }
    if (high > low)
    {
        selected.push_back(high);
    }
    if (picks.last > high)
    {
        selected.push_back(picks.last);
    }
}

SimpleCollector<uint64_t> select_drawable(
    std::span<double const> xs, std::span<double const> ys, double y_floor, std::size_t drawable)
{
    SimpleCollector<uint64_t> selected;
    selected.reserve(drawable);
    for (std::size_t it = 0; it < xs.size(); ++it)
    {
        if (is_drawable(xs[it], ys[it], y_floor))
        {
            selected.push_back(it);
        }
    }
    return selected;
}

/**
 * Select M4 over samples whose drawable abscissae never decrease, so the
 * samples left of the view, in it, and right of it come in that order.
 */
SimpleCollector<uint64_t> select_m4(
    std::span<double const> xs,
    std::span<double const> ys,
    double xmin,
    double xmax,
    std::size_t columns,
    double y_floor)
{
    SimpleCollector<uint64_t> selected;
    selected.reserve(M4_SAMPLES_PER_COLUMN * columns + 2);

    std::size_t it = 0;
    std::optional<std::size_t> entering;
    for (; it < xs.size(); ++it)
    {
        if (is_drawable(xs[it], ys[it], y_floor))
        {
            if (xs[it] >= xmin)
            {
                break;
            }
            entering = it;
        }
    }
    if (entering.has_value())
    {
        selected.push_back(*entering);
    }

    double const scale = static_cast<double>(columns) / (xmax - xmin);
    std::optional<ColumnPicks> picks;
    std::size_t column = 0;
    for (; it < xs.size(); ++it)
    {
        if (!is_drawable(xs[it], ys[it], y_floor))
        {
            continue;
        }
        if (xs[it] > xmax)
        {
            break;
        }
        // The cast floors a non-negative position; xmax itself lands one past
        // the last column.
        std::size_t const at = std::min(static_cast<std::size_t>((xs[it] - xmin) * scale), columns - 1);
        if (picks.has_value() && at == column)
        {
            widen(*picks, ys, it);
            continue;
        }
        if (picks.has_value())
        {
            append(selected, *picks);
        }
        picks = ColumnPicks{it, it, it, it};
        column = at;
    }
    if (picks.has_value())
    {
        append(selected, *picks);
    }

    // The first sample past the view carries the line out of it, and nothing
    // after that can be seen.
    if (it < xs.size())
    {
        selected.push_back(it);
    }
    return selected;
}

} /* end namespace */

RPlotDecimator::indices_type RPlotDecimator::select(
    RPlotSeries const & series,
    double xmin,
    double xmax,
    std::size_t columns,
    double y_floor) const
{
    if (columns < 1)
    {
        throw std::invalid_argument(
            std::format("RPlotDecimator::select: columns must be at least 1, but it is {}", columns));
    }

    std::span<double const> const xs = series.x();
    std::span<double const> const ys = series.y();
    std::size_t drawable = 0;
    bool increasing = true;
    double previous = -std::numeric_limits<double>::infinity();
    for (std::size_t it = 0; it < xs.size(); ++it)
    {
        if (is_drawable(xs[it], ys[it], y_floor))
        {
            ++drawable;
            increasing = increasing && xs[it] >= previous;
            previous = xs[it];
        }
    }

    // A curve that doubles back cannot be cut into columns, and a range that
    // places no column has nothing to cut into. The ceiling division keeps a
    // huge column count from overflowing the density check.
    double const span = xmax - xmin;
    bool const placeable = std::isfinite(span) && span > 0.0 && std::isfinite(static_cast<double>(columns) / span);
    bool const dense = (drawable + M4_SAMPLES_PER_COLUMN - 1) / M4_SAMPLES_PER_COLUMN > columns;
    if (increasing && placeable && dense)
    {
        return select_m4(xs, ys, xmin, xmax, columns, y_floor).as_array();
    }
    return select_drawable(xs, ys, y_floor, drawable).as_array();
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
