#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Qt-free theme foundation: the color-role vocabulary, the light and dark
 * color tables carried on a per-platform axis, the rule that resolves a
 * requested mode against the operating system color scheme, and the record of
 * what each platform's theme can honor.
 *
 * Nothing here mentions Qt, so the whole foundation compiles into the
 * no-GUI test target and every platform's tables are checked on every CI
 * runner. The Qt adapter (RThemeManager) turns a table into a QPalette by a
 * straight field-by-field copy, and the per-platform backends read the
 * capability record to decide what native levers to pull.
 *
 * @ingroup group_domain
 */

#include <cstdint>

#include <solvcon/pilot/common/platform.hpp>

namespace solvcon
{

/**
 * @brief The source a theme draws its variant from.
 *
 * System follows the operating system color scheme and tracks changes to it;
 * Light and Dark pin the palette regardless of the operating system.
 */
enum class ThemeMode
{
    System,
    Light,
    Dark,
};

/// The two concrete palettes a mode resolves to.
enum class ThemeVariant
{
    Light,
    Dark,
};

/**
 * @brief Where the pilot's colors come from, independent of the light or dark
 * mode.
 *
 * System lets the running platform's own colors show through the native style
 * untouched; Curated paints the plan's palettes over that style for a
 * controlled look that travels between machines.
 */
enum class ThemeLook
{
    System,
    Curated,
};

/**
 * @brief One sRGB color, a byte per channel.
 *
 * Deliberately free of Qt so the color tables can live in a translation unit
 * that compiles and tests without QtGui.
 */
struct ThemeColor
{
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;

    constexpr ThemeColor() = default;

    constexpr ThemeColor(std::uint8_t red, std::uint8_t green, std::uint8_t blue)
        : r(red)
        , g(green)
        , b(blue)
    {
    }
}; /* end struct ThemeColor */

/**
 * @brief The colors a palette assigns to the widget roles the pilot uses.
 *
 * Each field is named after the QPalette::ColorRole it maps to, so the Qt
 * adapter copies the struct into a QPalette one field at a time. The trailing
 * disabled_* fields feed the QPalette::Disabled color group, which keeps
 * greyed-out text legible in both variants.
 */
struct ThemePalette
{
    ThemeColor window;
    ThemeColor window_text;
    ThemeColor base;
    ThemeColor alternate_base;
    ThemeColor text;
    ThemeColor button;
    ThemeColor button_text;
    ThemeColor bright_text;
    ThemeColor highlight;
    ThemeColor highlighted_text;
    ThemeColor tool_tip_base;
    ThemeColor tool_tip_text;
    ThemeColor placeholder_text;
    ThemeColor link;
    ThemeColor link_visited;
    ThemeColor disabled_text;
    ThemeColor disabled_button_text;
    ThemeColor disabled_window_text;
    ThemeColor disabled_highlight;
}; /* end struct ThemePalette */

/**
 * @brief The colors the console's Python highlighter, its matching-bracket
 * marker, and its captured error output draw with.
 *
 * These are not QPalette roles, so they live apart from ThemePalette; the
 * console reads them directly. Like the palette they come in a light and a
 * dark table so the highlighted code stays legible under either variant, which
 * the hardcoded single set could not do once the console followed the theme.
 * bracket_match is a background wash behind a matched pair of brackets, and
 * error is the foreground the terminal paints captured stderr with, kept a
 * distinct red in each variant. Ordinary stdout takes the palette text color
 * rather than a field here.
 */
struct SyntaxColors
{
    ThemeColor keyword;
    ThemeColor builtin;
    ThemeColor string;
    ThemeColor comment;
    ThemeColor number;
    ThemeColor bracket_match;
    ThemeColor error;
}; /* end struct SyntaxColors */

/**
 * @brief What a platform's theme is able to honor.
 *
 * A plain record of yes-or-no facts, kept in the Qt-free core so it is data a
 * test can read rather than behavior a display is needed to observe. The
 * adapter exposes it to the interface, which greys out any menu choice the
 * running platform cannot deliver, so the pilot never offers a switch that
 * does nothing.
 */
struct ThemeCapabilities
{
    /// Can track the operating system light-or-dark setting and follow it live.
    bool can_follow_system = false;
    /// Can pin Light or Dark against the operating system's own preference.
    bool can_force_variant = false;
    /// Supplies a platform accent color read from the operating system.
    bool has_native_accent = false;
    /// Can theme the window's title bar, not just its client area.
    bool controls_titlebar = false;
    /// Installs a platform-native widget style rather than a portable one.
    bool has_native_style = false;
}; /* end struct ThemeCapabilities */

/**
 * The curated light palette, the shared fallback every platform table seeds
 * from until its room is furnished.
 */
ThemePalette const & lightThemePalette();

/// The curated dark palette.
ThemePalette const & darkThemePalette();

/// The palette a platform draws a resolved variant with.
ThemePalette const & themePaletteFor(PlatformId platform, ThemeVariant variant);

/// The console syntax colors tuned for a light background.
SyntaxColors const & lightSyntaxColors();

/// The console syntax colors tuned for a dark background.
SyntaxColors const & darkSyntaxColors();

/// The syntax colors a platform draws a resolved variant with.
SyntaxColors const & syntaxColorsFor(PlatformId platform, ThemeVariant variant);

/// The capability record for a platform.
ThemeCapabilities const & themeCapabilitiesFor(PlatformId platform);

/**
 * @brief Resolve a requested mode to a concrete variant.
 *
 * In System mode the choice follows @p os_prefers_dark; Light and Dark ignore
 * it and return their own variant.
 */
ThemeVariant resolveThemeVariant(ThemeMode mode, bool os_prefers_dark);

/**
 * The stable identifier for a mode, used as the menu action object name, at
 * the Python boundary, and in tests ("system", "light", "dark").
 */
char const * themeModeId(ThemeMode mode);

/// The human-readable menu label for a mode.
char const * themeModeLabel(ThemeMode mode);

/// The mode named by @p id, or ThemeMode::System when @p id matches none.
ThemeMode themeModeFromId(char const * id);

/**
 * The stable identifier for a look ("system", "curated"), used as the menu
 * action object name, at the Python boundary, and in tests.
 */
char const * themeLookId(ThemeLook look);

/// The human-readable menu label for a look.
char const * themeLookLabel(ThemeLook look);

/// The look named by @p id, or ThemeLook::Curated when @p id matches none.
ThemeLook themeLookFromId(char const * id);

/**
 * @brief Whether a Linux desktop names a theme the pilot recognizes.
 *
 * Reads the value of XDG_CURRENT_DESKTOP and reports true for GNOME or KDE,
 * whose Qt platform themes expose a palette and an accent worth honoring. An
 * empty or unrecognized value returns false, so the Linux room falls back to
 * the curated palettes. The matching is Qt-free so it is tested on every
 * runner.
 */
bool linuxDesktopHasNativeTheme(char const * xdg_current_desktop);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
