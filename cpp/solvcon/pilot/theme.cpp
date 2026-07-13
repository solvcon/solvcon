/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/theme.hpp>

#include <cstring>

namespace solvcon
{

// The curated palettes below are neutral, low-saturation greys lifted a step
// off pure black and pure white so large fills do not glare, paired with one
// calm blue accent shared by both variants. They are the shared fallback: every
// platform table starts from these values, and a platform room later tunes its
// own copy toward the native look without touching the others.

static ThemePalette makeLightPalette()
{
    ThemePalette p;
    p.window = {0xf2, 0xf2, 0xf3};
    p.window_text = {0x1c, 0x1e, 0x21};
    p.base = {0xff, 0xff, 0xff};
    p.alternate_base = {0xf6, 0xf6, 0xf7};
    p.text = {0x1c, 0x1e, 0x21};
    p.button = {0xea, 0xea, 0xec};
    p.button_text = {0x1c, 0x1e, 0x21};
    p.bright_text = {0xd3, 0x2f, 0x2f};
    p.highlight = {0x35, 0x74, 0xf0};
    p.highlighted_text = {0xff, 0xff, 0xff};
    p.tool_tip_base = {0xfa, 0xfa, 0xfb};
    p.tool_tip_text = {0x1c, 0x1e, 0x21};
    p.placeholder_text = {0x9a, 0xa0, 0xa6};
    p.link = {0x1a, 0x5f, 0xb4};
    p.link_visited = {0x7e, 0x4f, 0xb0};
    p.disabled_text = {0xa6, 0xa8, 0xac};
    p.disabled_button_text = {0xa6, 0xa8, 0xac};
    p.disabled_window_text = {0xa6, 0xa8, 0xac};
    p.disabled_highlight = {0xc8, 0xcb, 0xcf};
    return p;
}

static ThemePalette makeDarkPalette()
{
    ThemePalette p;
    p.window = {0x2d, 0x2f, 0x33};
    p.window_text = {0xe6, 0xe6, 0xe7};
    p.base = {0x23, 0x24, 0x27};
    p.alternate_base = {0x2b, 0x2d, 0x31};
    p.text = {0xe6, 0xe6, 0xe7};
    p.button = {0x35, 0x37, 0x3b};
    p.button_text = {0xe6, 0xe6, 0xe7};
    p.bright_text = {0xff, 0x6b, 0x68};
    p.highlight = {0x3d, 0x82, 0xe0};
    p.highlighted_text = {0xff, 0xff, 0xff};
    p.tool_tip_base = {0x35, 0x37, 0x3b};
    p.tool_tip_text = {0xe6, 0xe6, 0xe7};
    p.placeholder_text = {0x80, 0x84, 0x89};
    p.link = {0x3d, 0xae, 0xe9};
    p.link_visited = {0xb1, 0x86, 0xd8};
    p.disabled_text = {0x6b, 0x6e, 0x73};
    p.disabled_button_text = {0x6b, 0x6e, 0x73};
    p.disabled_window_text = {0x6b, 0x6e, 0x73};
    p.disabled_highlight = {0x3a, 0x3d, 0x42};
    return p;
}

// The syntax colors keep the light table's familiar hues (a blue keyword, a
// teal builtin, a red string, a magenta number) and lift each to a brighter,
// lower-saturation tint for the dark table so the tokens read clearly on the
// dark base instead of sinking into it.

static SyntaxColors makeLightSyntaxColors()
{
    SyntaxColors c;
    c.keyword = {0x00, 0x00, 0xb4};
    c.builtin = {0x00, 0x6e, 0x6e};
    c.string = {0xa0, 0x00, 0x00};
    c.comment = {0x80, 0x80, 0x80};
    c.number = {0x8c, 0x00, 0x8c};
    c.bracket_match = {0xb4, 0xb4, 0xff};
    return c;
}

static SyntaxColors makeDarkSyntaxColors()
{
    SyntaxColors c;
    c.keyword = {0x6a, 0xb7, 0xff};
    c.builtin = {0x4e, 0xc9, 0xb0};
    c.string = {0xe5, 0x92, 0x8b};
    c.comment = {0x80, 0x84, 0x89};
    c.number = {0xd6, 0xa4, 0xe0};
    c.bracket_match = {0x3d, 0x50, 0x6b};
    return c;
}

// The capability records state what each platform's theme can honor. macOS and
// Windows own their title bar and pin a variant through the native color-scheme
// hint, so both flags are set; Linux carries a pinned variant with the curated
// palette instead and does not reach the title bar. These seed values are
// refined as each room is furnished.

static ThemeCapabilities makeLinuxCapabilities()
{
    ThemeCapabilities c;
    c.can_follow_system = true;
    c.can_force_variant = true;
    c.has_native_accent = true;
    c.controls_titlebar = false;
    c.has_native_style = true;
    return c;
}

static ThemeCapabilities makeMacCapabilities()
{
    ThemeCapabilities c;
    c.can_follow_system = true;
    c.can_force_variant = true;
    c.has_native_accent = true;
    c.controls_titlebar = true;
    c.has_native_style = true;
    return c;
}

static ThemeCapabilities makeWindowsCapabilities()
{
    ThemeCapabilities c;
    c.can_follow_system = true;
    c.can_force_variant = true;
    c.has_native_accent = true;
    c.controls_titlebar = true;
    c.has_native_style = true;
    return c;
}

ThemePalette const & lightThemePalette()
{
    static ThemePalette const palette = makeLightPalette();
    return palette;
}

ThemePalette const & darkThemePalette()
{
    static ThemePalette const palette = makeDarkPalette();
    return palette;
}

ThemePalette const & themePaletteFor(PlatformId platform, ThemeVariant variant)
{
    // Every platform seeds from the curated tables; the platform rooms furnished
    // in later steps replace this with their own tuned copies.
    (void)platform;
    return variant == ThemeVariant::Dark ? darkThemePalette() : lightThemePalette();
}

SyntaxColors const & lightSyntaxColors()
{
    static SyntaxColors const colors = makeLightSyntaxColors();
    return colors;
}

SyntaxColors const & darkSyntaxColors()
{
    static SyntaxColors const colors = makeDarkSyntaxColors();
    return colors;
}

SyntaxColors const & syntaxColorsFor(PlatformId platform, ThemeVariant variant)
{
    (void)platform;
    return variant == ThemeVariant::Dark ? darkSyntaxColors() : lightSyntaxColors();
}

ThemeCapabilities const & themeCapabilitiesFor(PlatformId platform)
{
    static ThemeCapabilities const linux_caps = makeLinuxCapabilities();
    static ThemeCapabilities const mac_caps = makeMacCapabilities();
    static ThemeCapabilities const windows_caps = makeWindowsCapabilities();

    switch (platform)
    {
    case PlatformId::Mac:
        return mac_caps;
    case PlatformId::Windows:
        return windows_caps;
    case PlatformId::Linux:
    default:
        return linux_caps;
    }
}

ThemeVariant resolveThemeVariant(ThemeMode mode, bool os_prefers_dark)
{
    switch (mode)
    {
    case ThemeMode::Light:
        return ThemeVariant::Light;
    case ThemeMode::Dark:
        return ThemeVariant::Dark;
    case ThemeMode::System:
    default:
        return os_prefers_dark ? ThemeVariant::Dark : ThemeVariant::Light;
    }
}

char const * themeModeId(ThemeMode mode)
{
    switch (mode)
    {
    case ThemeMode::Light:
        return "light";
    case ThemeMode::Dark:
        return "dark";
    case ThemeMode::System:
    default:
        return "system";
    }
}

char const * themeModeLabel(ThemeMode mode)
{
    switch (mode)
    {
    case ThemeMode::Light:
        return "Light";
    case ThemeMode::Dark:
        return "Dark";
    case ThemeMode::System:
    default:
        return "Follow system";
    }
}

ThemeMode themeModeFromId(char const * id)
{
    if (id != nullptr)
    {
        if (0 == std::strcmp(id, "light"))
        {
            return ThemeMode::Light;
        }
        if (0 == std::strcmp(id, "dark"))
        {
            return ThemeMode::Dark;
        }
    }
    return ThemeMode::System;
}

char const * platformIdName(PlatformId platform)
{
    switch (platform)
    {
    case PlatformId::Mac:
        return "mac";
    case PlatformId::Windows:
        return "windows";
    case PlatformId::Linux:
    default:
        return "linux";
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
