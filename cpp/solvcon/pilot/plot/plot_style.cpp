/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/plot/plot_style.hpp>

#include <array>

namespace solvcon
{

namespace
{

// The matplotlib "tab10" cycle, reused verbatim so a pilot plot reads like the
// matplotlib one (https://matplotlib.org/stable/users/explain/colors/colors.html).
constexpr std::array<PlotColor, 10> PLOT_COLOR_TABLE = {
    PlotColor(31, 119, 180), // C0 #1f77b4
    PlotColor(255, 127, 14), // C1 #ff7f0e
    PlotColor(44, 160, 44), // C2 #2ca02c
    PlotColor(214, 39, 40), // C3 #d62728
    PlotColor(148, 103, 189), // C4 #9467bd
    PlotColor(140, 86, 75), // C5 #8c564b
    PlotColor(227, 119, 194), // C6 #e377c2
    PlotColor(127, 127, 127), // C7 #7f7f7f
    PlotColor(188, 189, 34), // C8 #bcbd22
    PlotColor(23, 190, 207), // C9 #17becf
};

} /* end namespace */

std::span<PlotColor const> plot_color_cycle()
{
    return std::span<PlotColor const>(PLOT_COLOR_TABLE.data(), PLOT_COLOR_TABLE.size());
}

PlotColor plot_cycle_color(std::size_t index)
{
    return PLOT_COLOR_TABLE[index % PLOT_COLOR_TABLE.size()];
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
