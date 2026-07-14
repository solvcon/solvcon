#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The platform axis shared by the pilot's Qt-free cores.
 *
 * Both the theme foundation and the keymap foundation carry a per-platform
 * table, so they name the same axis here rather than each defining its own.
 * Detecting the running platform is a thin Qt call made in an adapter and the
 * result passed in, which keeps this header free of Qt.
 *
 * @ingroup group_domain
 */

namespace solvcon
{

/// The platform a per-platform table or capability record is written for.
enum class PlatformId
{
    Linux,
    Mac,
    Windows
};

/**
 * The stable identifier for a platform ("linux", "mac", "windows"), used at
 * the Python boundary and in tests.
 */
inline char const * platformIdName(PlatformId platform)
{
    switch (platform)
    {
    case PlatformId::Linux:
        return "linux";
    case PlatformId::Mac:
        return "mac";
    case PlatformId::Windows:
        return "windows";
    }
    return "linux";
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
