/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/plot/RPlotTicker.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>

namespace solvcon
{

namespace
{

constexpr double TICK_BOUND_SLACK = 1e-9;
constexpr double PLAIN_LABEL_MINIMUM = 1e-3;
constexpr double PLAIN_LABEL_MAXIMUM = 1e5;
constexpr int DECADE_END_DIGITS = 4;
constexpr int LABEL_MAXIMUM_DIGITS = 17;

} /* end namespace */

RPlotTicker::RPlotTicker(std::size_t target_count)
{
    set_target_count(target_count);
}

void RPlotTicker::set_target_count(std::size_t count)
{
    if (count < 1)
    {
        throw std::invalid_argument(
            std::format("RPlotTicker::set_target_count: target count must be at least 1, but it is {}", count));
    }
    m_target_count = count;
}

RPlotTicker::ticks_type RPlotTicker::locate(double lo, double hi) const
{
    double const span = hi - lo;
    if (!std::isfinite(span) || !(span > 0.0))
    {
        return {};
    }

    double const raw_step = span / static_cast<double>(m_target_count);
    if (!std::isfinite(raw_step) || !(raw_step > 0.0))
    {
        return {};
    }

    double const magnitude = std::pow(10.0, std::floor(std::log10(raw_step)));
    if (!std::isfinite(magnitude) || !(magnitude > 0.0))
    {
        return {};
    }

    double step = 0.0;
    for (double multiplier : {1.0, 2.0, 5.0, 10.0})
    {
        double const candidate = multiplier * magnitude;
        if (!std::isfinite(candidate))
        {
            break;
        }
        step = candidate;
        if (raw_step <= step)
        {
            break;
        }
    }
    if (!std::isfinite(step) || !(step > 0.0) || raw_step > step)
    {
        return {};
    }

    double const first = std::ceil(lo / step);
    double const last = std::floor((hi + TICK_BOUND_SLACK * step) / step);
    double const count = last - first + 1.0;
    if (!std::isfinite(first) || !std::isfinite(last) || !std::isfinite(count))
    {
        return {};
    }

    double const maximum_count = static_cast<double>(m_target_count) + 2.0;
    double const maximum_size =
        static_cast<double>(std::numeric_limits<std::size_t>::max());
    if (!(count > 0.0) || count > maximum_count || !(count < maximum_size))
    {
        return {};
    }

    ticks_type ticks;
    std::size_t const tick_count = static_cast<std::size_t>(count);
    for (std::size_t it = 0; it < tick_count; ++it)
    {
        // Rebuild from the index so a small step advances at a large offset.
        double const index = first + static_cast<double>(it);
        double const value = (index == 0.0) ? 0.0 : std::clamp(index * step, lo, hi);
        ticks.push_back(value);
    }
    return ticks;
}

RPlotTicker::ticks_type RPlotTicker::locate_decades(double lo, double hi) const
{
    if (!std::isfinite(lo) || !std::isfinite(hi))
    {
        return {};
    }

    double const first = std::ceil(lo);
    double const last = std::floor(hi);
    if (last < first)
    {
        return {lo, hi};
    }

    double const count = last - first + 1.0;
    double const maximum_size =
        static_cast<double>(std::numeric_limits<std::size_t>::max());
    if (!std::isfinite(count) || !(count > 0.0) || !(count < maximum_size))
    {
        return {};
    }

    ticks_type ticks;
    std::size_t const tick_count = static_cast<std::size_t>(count);
    for (std::size_t it = 0; it < tick_count; ++it)
    {
        ticks.push_back(first + static_cast<double>(it));
    }
    return ticks;
}

/// The smallest gap between neighbouring ticks, or zero when no two ticks differ.
static double smallest_gap(RPlotTicker::ticks_type const & ticks)
{
    double gap = 0.0;
    for (std::size_t it = 1; it < ticks.size(); ++it)
    {
        double const step = ticks[it] - ticks[it - 1];
        if (step > 0.0 && (gap == 0.0 || step < gap))
        {
            gap = step;
        }
    }
    return gap;
}

/// Whether two neighbouring ticks of different values read alike.
static bool labels_collide(RPlotTicker::ticks_type const & ticks, RPlotTicker::labels_type const & texts)
{
    for (std::size_t it = 1; it < texts.size(); ++it)
    {
        if (texts[it] == texts[it - 1] && ticks[it] != ticks[it - 1])
        {
            return true;
        }
    }
    return false;
}

static std::string format_plain(double value, int digits)
{
    std::string text = std::format("{:.{}f}", value, digits);
    // A residue just below zero would otherwise read "-0.0".
    if (text.find_first_not_of("-0.") == std::string::npos)
    {
        return "0";
    }
    return text;
}

RPlotTicker::labels_type RPlotTicker::labels(ticks_type const & ticks) const
{
    double largest = 0.0;
    for (double const value : ticks)
    {
        largest = std::max(largest, std::abs(value));
    }
    // One notation for the whole axis, so a label never switches form beside its neighbour.
    bool const plain = largest == 0.0 || (PLAIN_LABEL_MINIMUM <= largest && largest < PLAIN_LABEL_MAXIMUM);

    double const gap = smallest_gap(ticks);
    if (!(gap > 0.0) || !std::isfinite(gap))
    {
        // A lone tick has no neighbour to tell apart, so it keeps the short form.
        labels_type texts;
        for (double const value : ticks)
        {
            if (value == 0.0)
            {
                texts.emplace_back("0");
            }
            else
            {
                texts.push_back(plain ? std::format("{:.4g}", value) : std::format("{:.0e}", value));
            }
        }
        return texts;
    }

    auto const render = [&ticks, plain](int digits)
    {
        labels_type texts;
        for (double const value : ticks)
        {
            if (value == 0.0)
            {
                texts.emplace_back("0");
            }
            else
            {
                texts.push_back(plain ? format_plain(value, digits) : std::format("{:.{}e}", value, digits));
            }
        }
        return texts;
    };

    // The digits that resolve the spacing, as the 2D canvas grid labels do. The
    // slack keeps a gap that rounding left just short of its power of ten on it.
    int const gap_exponent = static_cast<int>(std::floor(std::log10(gap) + TICK_BOUND_SLACK));
    int const top_exponent = static_cast<int>(std::floor(std::log10(largest)));
    int digits = std::clamp(plain ? -gap_exponent : top_exponent - gap_exponent, 0, LABEL_MAXIMUM_DIGITS);
    labels_type texts = render(digits);
    while (digits < LABEL_MAXIMUM_DIGITS && labels_collide(ticks, texts))
    {
        ++digits;
        texts = render(digits);
    }
    return texts;
}

RPlotTicker::labels_type RPlotTicker::decade_labels(ticks_type const & ticks) const
{
    auto const render = [&ticks](int digits)
    {
        labels_type texts;
        for (double const value : ticks)
        {
            if (std::isfinite(value) && std::floor(value) == value)
            {
                texts.push_back(std::format("1e{:.0f}", value));
            }
            else
            {
                texts.push_back(std::format("{:.{}g}", std::pow(10.0, value), digits));
            }
        }
        return texts;
    };

    int digits = DECADE_END_DIGITS;
    labels_type texts = render(digits);
    while (digits < LABEL_MAXIMUM_DIGITS && labels_collide(ticks, texts))
    {
        ++digits;
        texts = render(digits);
    }
    return texts;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
