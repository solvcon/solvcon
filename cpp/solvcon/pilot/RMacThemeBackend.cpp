/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RMacThemeBackend.hpp>

#include <QtGlobal>

// The whole room is native to macOS, so the body compiles only there. On other
// platforms this is an empty translation unit and the factory never names the
// class.
#if defined(Q_OS_MACOS)

#include <cstdint>

#include <QApplication>
#include <QColor>
#include <QPalette>
#include <QStyle>

namespace solvcon
{

PlatformId RMacThemeBackend::platform() const
{
    return PlatformId::Mac;
}

std::string RMacThemeBackend::styleName() const
{
    return "macos";
}

std::optional<ThemeColor> RMacThemeBackend::accentColor(ThemeVariant /*variant*/) const
{
    // The macos style's standard palette carries the user's system accent in
    // the Accent role (Qt 6.6+), falling back to Highlight. Reading it from the
    // style rather than the application palette keeps it independent of the
    // curated palette the manager later applies over the top.
    QStyle * style = QApplication::style();
    if (style == nullptr)
    {
        return std::nullopt;
    }

    QPalette const platform = style->standardPalette();
    QColor accent;
#if QT_VERSION >= QT_VERSION_CHECK(6, 6, 0)
    accent = platform.color(QPalette::Accent);
#endif
    if (!accent.isValid())
    {
        accent = platform.color(QPalette::Highlight);
    }
    if (!accent.isValid())
    {
        return std::nullopt;
    }
    return ThemeColor{static_cast<std::uint8_t>(accent.red()),
                      static_cast<std::uint8_t>(accent.green()),
                      static_cast<std::uint8_t>(accent.blue())};
}

void RMacThemeBackend::applyNativeChrome(QWidget * /*window*/, ThemeVariant /*variant*/)
{
    // The color-scheme hint the manager sets flips the whole native appearance
    // on Qt 6.8 and newer, title bar included, so there is no extra chrome to
    // apply here.
}

ThemeCapabilities RMacThemeBackend::capabilities() const
{
    ThemeCapabilities caps = themeCapabilitiesFor(PlatformId::Mac);
#if QT_VERSION < QT_VERSION_CHECK(6, 8, 0)
    // Without the color-scheme hint the native appearance cannot be pinned, so
    // a forced variant is carried only by the curated palette and the title bar
    // stays on the operating system setting.
    caps.controls_titlebar = false;
#endif
    return caps;
}

} /* end namespace solvcon */

#endif /* defined(Q_OS_MACOS) */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
