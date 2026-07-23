#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The Linux room: the native theme backend for Linux and other desktops.
 *
 * Keeps the desktop's own Qt style, honors the desktop accent when a
 * recognized platform theme exposes one, and otherwise leans on the curated
 * palettes. It is the room with the most variety across desktops, so it leans
 * hardest on the shared fallback. This is also the factory's default, selected
 * for any platform without a room of its own.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/theme/RThemeBackend.hpp>

#include <optional>
#include <string>

namespace solvcon
{

/**
 * @brief The Linux native theme backend.
 *
 * @ingroup group_domain
 */
class RLinuxThemeBackend
    : public RThemeBackend
{

public:

    PlatformId platform() const override;
    std::string styleName() const override;
    std::optional<ThemeColor> accentColor(ThemeVariant variant) const override;
    void applyNativeChrome(QWidget * window, ThemeVariant variant) override;
    ThemeCapabilities capabilities() const override;

}; /* end class RLinuxThemeBackend */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
