#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Qt-free plot styling vocabulary: the color a curve is stroked with, the
 * default stroke width, and the matplotlib C0-C9 categorical cycle to draw
 * from when no explicit color is given.
 *
 * Nothing here may mention Qt. Converting a PlotColor to a QColor belongs in
 * the Qt front end, at that boundary and nowhere else.
 *
 * @ingroup group_domain
 */

#include <cstddef>
#include <cstdint>
#include <span>

namespace solvcon
{

/// One sRGB color with alpha, a byte per channel.
struct PlotColor
{
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
    std::uint8_t a = 255;

    constexpr PlotColor() = default;

    constexpr PlotColor(std::uint8_t red, std::uint8_t green, std::uint8_t blue, std::uint8_t alpha = 255)
        : r(red)
        , g(green)
        , b(blue)
        , a(alpha)
    {
    }

    constexpr bool operator==(PlotColor const &) const = default;
}; /* end struct PlotColor */

/// Default stroke width in screen pixels; matplotlib's lines.linewidth.
inline constexpr double PLOT_DEFAULT_LINE_WIDTH = 1.5;

/**
 * The matplotlib C0-C9 categorical cycle, in order. Exactly ten entries over
 * storage of static lifetime, so the span outlives any caller.
 */
std::span<PlotColor const> plot_color_cycle();

/// The cycle entry for @p index, wrapping with index % plot_color_cycle().size().
PlotColor plot_cycle_color(std::size_t index);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
