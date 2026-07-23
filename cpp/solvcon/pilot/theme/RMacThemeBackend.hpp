#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The macOS room: the native theme backend for macOS.
 *
 * Installs the native macos style, reads the user's system accent, and leans
 * on the color-scheme hint the manager sets to pin a variant, title bar
 * included, on Qt 6.8 and newer. Only linked into the macOS build; the factory
 * in RThemeBackend.cpp is the sole site that selects it.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/theme/RThemeBackend.hpp>

#include <optional>
#include <string>

namespace solvcon
{

/**
 * @brief The macOS native theme backend.
 *
 * @ingroup group_domain
 */
class RMacThemeBackend
    : public RThemeBackend
{

public:

    PlatformId platform() const override;
    std::string styleName() const override;
    std::optional<ThemeColor> accentColor(ThemeVariant variant) const override;
    void applyNativeChrome(QWidget * window, ThemeVariant variant) override;
    ThemeCapabilities capabilities() const override;

}; /* end class RMacThemeBackend */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
