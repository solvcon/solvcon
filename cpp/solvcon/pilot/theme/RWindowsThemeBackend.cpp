/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/theme/RWindowsThemeBackend.hpp>

#include <QtGlobal>

// The whole room is native to Windows, so the body compiles only there. On
// other platforms this is an empty translation unit and the factory never names
// the class.
#if defined(Q_OS_WIN)

#include <cstdint>

#include <QApplication>
#include <QColor>
#include <QGuiApplication>
#include <QPalette>
#include <QString>
#include <QStyle>
#include <QStyleFactory>
#include <QWidget>

#include <windows.h>

namespace solvcon
{

namespace
{

// The desktop window manager attribute that switches the title bar to its dark
// variant. Defined here so the room needs no extra SDK header.
constexpr DWORD kUseImmersiveDarkMode = 20;

} /* end namespace */

PlatformId RWindowsThemeBackend::platform() const
{
    return PlatformId::Windows;
}

std::string RWindowsThemeBackend::styleName() const
{
    // Prefer the Fluent windows11 style; fall back to windowsvista on the Qt
    // builds that do not ship it.
    if (QStyleFactory::keys().contains(QStringLiteral("windows11")))
    {
        return "windows11";
    }
    return "windowsvista";
}

std::optional<ThemeColor> RWindowsThemeBackend::accentColor(ThemeVariant /*variant*/) const
{
    // The Windows style's standard palette carries the operating system accent
    // in the Accent role (Qt 6.6+), falling back to Highlight. Reading it from
    // the style keeps it independent of the curated palette applied over the
    // top.
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

void RWindowsThemeBackend::applyNativeChrome(QWidget * window, ThemeVariant variant)
{
    if (window == nullptr)
    {
        return;
    }

    // Under the offscreen or minimal platform there is no native window to
    // theme, so skip the call rather than reach for a handle that is not there.
    QString const qpa = QGuiApplication::platformName();
    if (qpa == QStringLiteral("offscreen") || qpa == QStringLiteral("minimal"))
    {
        return;
    }

    auto * const hwnd = reinterpret_cast<HWND>(window->winId());
    if (hwnd == nullptr)
    {
        return;
    }

    // dwmapi is loaded at runtime so the room needs no extra link dependency.
    HMODULE const dwm = LoadLibraryW(L"dwmapi.dll");
    if (dwm == nullptr)
    {
        return;
    }

    using DwmSetWindowAttributeFn = HRESULT(WINAPI *)(HWND, DWORD, LPCVOID, DWORD);
    auto * const set_attribute = reinterpret_cast<DwmSetWindowAttributeFn>(
        GetProcAddress(dwm, "DwmSetWindowAttribute"));
    if (set_attribute != nullptr)
    {
        BOOL const dark = (variant == ThemeVariant::Dark) ? TRUE : FALSE;
        set_attribute(hwnd, kUseImmersiveDarkMode, &dark, sizeof(dark));
    }
    FreeLibrary(dwm);
}

ThemeCapabilities RWindowsThemeBackend::capabilities() const
{
    // The title bar switches through the desktop window manager on every Qt
    // version, and the curated palette carries a forced variant where the
    // color-scheme hint is a no-op, so the seeded record already holds.
    return themeCapabilitiesFor(PlatformId::Windows);
}

} /* end namespace solvcon */

#endif /* defined(Q_OS_WIN) */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
