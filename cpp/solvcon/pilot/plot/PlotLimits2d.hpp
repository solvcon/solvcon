#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The named rectangular limits shared by the native xy plot data and view.
 *
 * @ingroup group_domain
 */

#include <algorithm>

namespace solvcon
{

/**
 * One xy rectangle, with named bounds rather than positional array entries.
 * The dimension suffix keeps this delivered type honestly 2d: a future 3d
 * plot adds a sibling type instead of changing this one.
 */
struct PlotLimits2d
{
    double xmin = 0.0;
    double xmax = 0.0;
    double ymin = 0.0;
    double ymax = 0.0;

    constexpr PlotLimits2d() = default;

    constexpr PlotLimits2d(double xmin, double xmax, double ymin, double ymax)
        : xmin(xmin)
        , xmax(xmax)
        , ymin(ymin)
        , ymax(ymax)
    {
    }

    void merge(PlotLimits2d const & other);

    constexpr bool operator==(PlotLimits2d const &) const = default;
}; /* end struct PlotLimits2d */

inline void PlotLimits2d::merge(PlotLimits2d const & other)
{
    xmin = std::min(xmin, other.xmin);
    xmax = std::max(xmax, other.xmax);
    ymin = std::min(ymin, other.ymin);
    ymax = std::max(ymax, other.ymax);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
