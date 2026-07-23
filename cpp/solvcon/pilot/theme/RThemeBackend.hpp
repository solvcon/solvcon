#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The backend seam: the Qt-and-native half of a theme that a color table
 * cannot express.
 *
 * A backend furnishes one platform's native look. It names the widget style to
 * install, reads the operating system accent, and applies window-manager chrome
 * a palette cannot reach. The makeThemeBackend() factory is the only place a
 * platform preprocessor branch appears, so only the running platform's backend
 * is compiled into the binary while the color tables it reads compile
 * everywhere through the Qt-free core.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/theme/theme.hpp>

#include <memory>
#include <optional>
#include <string>

class QWidget;

namespace solvcon
{

/**
 * @brief One platform's native theme levers, behind an interface so the
 * manager drives every platform the same way.
 *
 * @ingroup group_domain
 */
class RThemeBackend
{

public:

    RThemeBackend() = default;
    virtual ~RThemeBackend() = default;

    RThemeBackend(RThemeBackend const &) = delete;
    RThemeBackend & operator=(RThemeBackend const &) = delete;
    RThemeBackend(RThemeBackend &&) = delete;
    RThemeBackend & operator=(RThemeBackend &&) = delete;

    /// The platform this backend furnishes, naming the color table to read.
    virtual PlatformId platform() const = 0;

    /// The widget style to install, or an empty string to keep the platform's
    /// default style.
    virtual std::string styleName() const = 0;

    /// The operating-system accent for a variant, or an empty optional to keep
    /// the curated highlight from the color table.
    virtual std::optional<ThemeColor> accentColor(ThemeVariant variant) const = 0;

    /// Apply window-manager chrome no palette can reach, such as the Windows
    /// immersive dark title bar. A backend without such chrome does nothing.
    virtual void applyNativeChrome(QWidget * window, ThemeVariant variant) = 0;

    /// What this platform's theme can honor.
    virtual ThemeCapabilities capabilities() const = 0;

}; /* end class RThemeBackend */

/**
 * @brief Build the backend for the running platform.
 *
 * The single site of a platform preprocessor branch. Until a platform's room is
 * furnished it gets the default backend, which keeps the platform's own style.
 */
std::unique_ptr<RThemeBackend> makeThemeBackend();

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
