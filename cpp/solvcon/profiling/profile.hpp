#pragma once

/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Wall-time timing helpers for the runtime profiler.
 *
 * @ingroup group_core
 */

#include <solvcon/base.hpp>
#include <chrono>

namespace solvcon
{

/**
 * Simple timer for wall time using high-resolution clock.
 *
 * @ingroup group_core
 */
class StopWatch
{

private:

    using clock_type = std::chrono::high_resolution_clock;
    using time_type = std::chrono::time_point<clock_type>;

public:

    /// A singleton.
    static StopWatch & me()
    {
        static StopWatch instance;
        return instance;
    }

    // m_start and m_stop are initialized below; the body is not a trivial
    // default, so it cannot use '= default'.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,modernize-use-equals-default)
    StopWatch()
        : m_start(clock_type::now())
        , m_stop(m_start)
    {
    }

    StopWatch(StopWatch const &) = default;
    StopWatch(StopWatch &&) = default;
    StopWatch & operator=(StopWatch const &) = default;
    StopWatch & operator=(StopWatch &&) = default;
    ~StopWatch() = default;

    /**
     * Return seconds between laps.
     */
    double lap()
    {
        m_start = m_stop;
        m_stop = clock_type::now();
        return std::chrono::duration<double>(m_stop - m_start).count();
    }

    /**
     * Return seconds between end and start.
     */
    double duration() const { return std::chrono::duration<double>(m_stop - m_start).count(); }

    /**
     * Return resolution in second.
     */
    static constexpr double resolution()
    {
        return static_cast<double>(clock_type::period::num) / static_cast<double>(clock_type::period::den);
    }

private:

    time_type m_start;
    time_type m_stop;

}; /* end struct StopWatch */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
