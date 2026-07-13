/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RThemeBackend.hpp>

#include <solvcon/pilot/RLinuxThemeBackend.hpp>

#include <QtGlobal>

#if defined(Q_OS_MACOS)
#include <solvcon/pilot/RMacThemeBackend.hpp>
#elif defined(Q_OS_WIN)
#include <solvcon/pilot/RWindowsThemeBackend.hpp>
#endif

namespace solvcon
{

std::unique_ptr<RThemeBackend> makeThemeBackend()
{
#if defined(Q_OS_MACOS)
    return std::make_unique<RMacThemeBackend>();
#elif defined(Q_OS_WIN)
    return std::make_unique<RWindowsThemeBackend>();
#else
    // Linux, and every other desktop without a room of its own, land here.
    return std::make_unique<RLinuxThemeBackend>();
#endif
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
