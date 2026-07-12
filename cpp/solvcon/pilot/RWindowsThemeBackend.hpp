#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The Windows room: the native theme backend for Windows.
 *
 * Installs the windows11 style (falling back to windowsvista), reads the
 * operating system accent, and switches the title bar to its dark variant
 * through the desktop window manager. Only linked into the Windows build; the
 * factory in RThemeBackend.cpp is the sole site that selects it.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/RThemeBackend.hpp>

#include <optional>
#include <string>

namespace solvcon
{

/**
 * @brief The Windows native theme backend.
 *
 * @ingroup group_domain
 */
class RWindowsThemeBackend
    : public RThemeBackend
{

public:

    PlatformId platform() const override;
    std::string styleName() const override;
    std::optional<ThemeColor> accentColor(ThemeVariant variant) const override;
    void applyNativeChrome(QWidget * window, ThemeVariant variant) override;
    ThemeCapabilities capabilities() const override;

}; /* end class RWindowsThemeBackend */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
