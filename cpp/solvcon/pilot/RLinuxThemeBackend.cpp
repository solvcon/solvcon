/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RLinuxThemeBackend.hpp>

#include <cstdint>

#include <QApplication>
#include <QByteArray>
#include <QColor>
#include <QPalette>
#include <QStyle>
#include <QtGlobal>

namespace solvcon
{

namespace
{

// Whether the running desktop names a Qt platform theme the room honors, read
// from the environment through the Qt-free recognizer.
bool desktopHasNativeTheme()
{
    QByteArray const desktop = qgetenv("XDG_CURRENT_DESKTOP");
    return linuxDesktopHasNativeTheme(desktop.isEmpty() ? nullptr : desktop.constData());
}

} /* end namespace */

PlatformId RLinuxThemeBackend::platform() const
{
    return PlatformId::Linux;
}

std::string RLinuxThemeBackend::styleName() const
{
    // Keep the desktop's own Qt style so the pilot looks native; an empty name
    // tells the manager to leave the installed style alone.
    return {};
}

std::optional<ThemeColor> RLinuxThemeBackend::accentColor(ThemeVariant /*variant*/) const
{
    // Only a recognized desktop is trusted to expose an accent worth honoring;
    // elsewhere the curated highlight stays.
    if (!desktopHasNativeTheme())
    {
        return std::nullopt;
    }

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

void RLinuxThemeBackend::applyNativeChrome(QWidget * /*window*/, ThemeVariant /*variant*/)
{
    // Linux window managers own the title bar, so there is no chrome to apply.
}

ThemeCapabilities RLinuxThemeBackend::capabilities() const
{
    ThemeCapabilities caps = themeCapabilitiesFor(PlatformId::Linux);
    // The accent is only genuinely native on a recognized desktop; elsewhere
    // the highlight is the curated one, so report that plainly.
    caps.has_native_accent = desktopHasNativeTheme();
    return caps;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
