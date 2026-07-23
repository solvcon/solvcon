#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Applies a theme to the running QApplication and keeps it in step with the
 * operating system color scheme.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <memory>
#include <string>

#include <solvcon/pilot/theme.hpp>
#include <solvcon/pilot/RThemeBackend.hpp>

#include <QColor>
#include <QObject>
#include <QPalette>

class QWidget;

namespace solvcon
{

/**
 * @brief Drives the pilot's look: the running platform's native style with a
 * curated light or dark palette painted over it.
 *
 * @ingroup group_domain
 *
 * The manager owns the backend the factory built for the running platform. It
 * installs that backend's style, builds a QPalette from the platform's own
 * color table, folds in the platform accent when the backend exposes one, and
 * applies native window chrome. In System mode it reads the operating system
 * color scheme and follows changes to it live; Light and Dark pin the variant,
 * carried by the palette where the platform cannot force it natively.
 * themeChanged() fires after each application so widgets that cache colors can
 * refresh.
 */
class RThemeManager
    : public QObject
{
    Q_OBJECT

public:

    explicit RThemeManager(QObject * parent = nullptr);

    ~RThemeManager() override;

    /// Install the style if needed and paint the current mode's palette onto
    /// the QApplication. Safe to call before any widget exists.
    void apply();

    /// Track the main window so native chrome can follow the theme. Applying
    /// chrome at once keeps a window created before the first switch in step.
    void setWindow(QWidget * window);

    /// Switch the requested mode and re-apply. A no-op re-application is still
    /// cheap, so callers need not compare against the current mode first.
    void setMode(ThemeMode mode);

    /// Switch mode by its string id ("system", "light", "dark"); an unknown id
    /// falls back to System. Convenience for the Python console and menu.
    void setModeById(std::string const & id);

    /// Switch where the colors come from and re-apply. System shows the
    /// platform's own colors; Curated paints the plan's palettes over them.
    void setLook(ThemeLook look);

    /// Switch look by its string id ("system", "curated"); an unknown id falls
    /// back to Curated.
    void setLookById(std::string const & id);

    ThemeMode mode() const { return m_mode; }

    ThemeLook look() const { return m_look; }

    /// The concrete variant the current mode resolves to right now.
    ThemeVariant currentVariant() const;

    /// The running platform, naming the color table the theme draws from.
    PlatformId platform() const;

    /// What the running platform's theme can honor, for the interface to gate
    /// the menu against.
    ThemeCapabilities capabilities() const;

    /// String id of the current mode, for the Python boundary.
    std::string modeId() const;

    /// String id of the current look, for the Python boundary.
    std::string lookId() const;

    /// "light" or "dark" for the current variant, for the Python boundary.
    std::string variantId() const;

    /// The Window color of the palette in effect, whether curated or native, so
    /// widgets that paint their own backdrop can match it without lagging a
    /// switch behind the application palette.
    QColor windowColor() const { return m_window_color; }

signals:

    /// Emitted after a palette is applied, carrying the resolved variant.
    void themeChanged(ThemeVariant variant);

private:

    /// True when the operating system reports a dark color scheme. Unknown is
    /// read as light, the conventional default.
    bool osPrefersDark() const;

    /// Hint the platform color scheme so native chrome tracks a forced mode. A
    /// no-op on Qt below 6.8 and on desktops that do not honor the request,
    /// such as most Linux ones, where the applied palette carries the theme.
    void syncOsColorScheme();

    /// Build a QPalette from a Qt-free color table, folding in the platform
    /// accent when the backend supplies one, and filling the disabled group.
    QPalette buildPalette(ThemePalette const & spec, ThemeVariant variant) const;

    /// A thin stylesheet that adds what a QPalette cannot: a tooltip border and
    /// a focus ring on text inputs, colored from @p pal. Applied under the
    /// curated look and cleared under the system look, which stays native.
    QString supplementalStyleSheet(QPalette const & pal) const;

    /// Read the persisted mode and look, if any, into the current state.
    void restorePersisted();

    /// Write the current mode and look so the next session starts on them.
    void persist() const;

    ThemeMode m_mode = ThemeMode::System;

    /// Where the colors come from. Curated is the controlled default; System is
    /// opted into for the platform's own colors.
    ThemeLook m_look = ThemeLook::Curated;

    /// The Window color of the palette last applied, cached so the backdrop can
    /// read it without waiting for the application palette to propagate.
    QColor m_window_color;

    /// The running platform's backend, the only object that touches native
    /// levers. Never null after construction.
    std::unique_ptr<RThemeBackend> m_backend;

    /// The main window, for native chrome. Non-owning; null until set.
    QWidget * m_window = nullptr;

    /// Guards the one-time style install, so repeated apply() calls do not
    /// rebuild the style object.
    bool m_style_installed = false;
}; /* end class RThemeManager */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
