/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/theme.hpp>

#include <string>

#include <gtest/gtest.h>

using solvcon::darkSyntaxColors;
using solvcon::darkThemePalette;
using solvcon::lightSyntaxColors;
using solvcon::lightThemePalette;
using solvcon::PlatformId;
using solvcon::platformIdName;
using solvcon::resolveThemeVariant;
using solvcon::syntaxColorsFor;
using solvcon::themeCapabilitiesFor;
using solvcon::ThemeMode;
using solvcon::themeModeFromId;
using solvcon::themeModeId;
using solvcon::themeModeLabel;
using solvcon::themePaletteFor;
using solvcon::ThemeVariant;

TEST(PilotThemeResolve, ForcedModesIgnoreTheOs)
{
    EXPECT_EQ(resolveThemeVariant(ThemeMode::Light, true), ThemeVariant::Light);
    EXPECT_EQ(resolveThemeVariant(ThemeMode::Light, false), ThemeVariant::Light);
    EXPECT_EQ(resolveThemeVariant(ThemeMode::Dark, true), ThemeVariant::Dark);
    EXPECT_EQ(resolveThemeVariant(ThemeMode::Dark, false), ThemeVariant::Dark);
}

TEST(PilotThemeResolve, SystemFollowsTheOs)
{
    EXPECT_EQ(resolveThemeVariant(ThemeMode::System, true), ThemeVariant::Dark);
    EXPECT_EQ(resolveThemeVariant(ThemeMode::System, false), ThemeVariant::Light);
}

TEST(PilotThemeId, RoundTripsThroughItsId)
{
    EXPECT_EQ(std::string("system"), themeModeId(ThemeMode::System));
    EXPECT_EQ(std::string("light"), themeModeId(ThemeMode::Light));
    EXPECT_EQ(std::string("dark"), themeModeId(ThemeMode::Dark));

    EXPECT_EQ(themeModeFromId("system"), ThemeMode::System);
    EXPECT_EQ(themeModeFromId("light"), ThemeMode::Light);
    EXPECT_EQ(themeModeFromId("dark"), ThemeMode::Dark);
}

TEST(PilotThemeId, UnknownIdFallsBackToSystem)
{
    EXPECT_EQ(themeModeFromId("solarized"), ThemeMode::System);
    EXPECT_EQ(themeModeFromId(nullptr), ThemeMode::System);
}

TEST(PilotThemeId, EveryModeHasALabel)
{
    EXPECT_GT(std::string(themeModeLabel(ThemeMode::System)).size(), 0U);
    EXPECT_GT(std::string(themeModeLabel(ThemeMode::Light)).size(), 0U);
    EXPECT_GT(std::string(themeModeLabel(ThemeMode::Dark)).size(), 0U);
}

TEST(PilotThemePlatform, EveryPlatformHasAName)
{
    EXPECT_EQ(std::string("linux"), platformIdName(PlatformId::Linux));
    EXPECT_EQ(std::string("mac"), platformIdName(PlatformId::Mac));
    EXPECT_EQ(std::string("windows"), platformIdName(PlatformId::Windows));
}

TEST(PilotThemePalette, LightAndDarkDifferAndAreConsistent)
{
    auto const & light = lightThemePalette();
    auto const & dark = darkThemePalette();

    // The two variants must actually differ, or the switch is cosmetic only.
    EXPECT_NE(light.window.r, dark.window.r);

    // A light window is brighter than a dark one; its text is darker. This
    // guards against the two tables being swapped.
    EXPECT_GT(light.window.g, dark.window.g);
    EXPECT_LT(light.text.g, dark.text.g);
}

TEST(PilotThemePalette, EveryPlatformSelectsTheVariantTable)
{
    // The platform axis is seeded from the curated tables, so every platform
    // resolves a variant to the same base table until its room is furnished.
    // The lookup must still select the requested variant on each platform.
    for (PlatformId platform : {PlatformId::Linux, PlatformId::Mac, PlatformId::Windows})
    {
        EXPECT_GT(themePaletteFor(platform, ThemeVariant::Light).window.g,
                  themePaletteFor(platform, ThemeVariant::Dark).window.g);
        EXPECT_EQ(themePaletteFor(platform, ThemeVariant::Light).window.r,
                  lightThemePalette().window.r);
        EXPECT_EQ(themePaletteFor(platform, ThemeVariant::Dark).window.r,
                  darkThemePalette().window.r);
    }
}

TEST(PilotThemeSyntax, DarkTokensAreBrighterAndSelectByVariant)
{
    auto const & light = lightSyntaxColors();
    auto const & dark = darkSyntaxColors();

    auto sum = [](solvcon::ThemeColor c)
    { return static_cast<int>(c.r) + static_cast<int>(c.g) + static_cast<int>(c.b); };

    // The two tables must differ, or the console cannot follow the theme.
    EXPECT_NE(sum(light.keyword), sum(dark.keyword));

    // Dark tokens sit on a dark base, so each category is lifted brighter than
    // its light-table counterpart; this guards against the tables being
    // swapped.
    EXPECT_GT(sum(dark.keyword), sum(light.keyword));
    EXPECT_GT(sum(dark.string), sum(light.string));
    EXPECT_GT(sum(dark.number), sum(light.number));

    // syntaxColorsFor selects the matching table on every platform.
    for (PlatformId platform : {PlatformId::Linux, PlatformId::Mac, PlatformId::Windows})
    {
        EXPECT_EQ(syntaxColorsFor(platform, ThemeVariant::Light).keyword.b, light.keyword.b);
        EXPECT_EQ(syntaxColorsFor(platform, ThemeVariant::Dark).keyword.b, dark.keyword.b);
    }
}

TEST(PilotThemeCapabilities, DifferByPlatform)
{
    auto const & linux_caps = themeCapabilitiesFor(PlatformId::Linux);
    auto const & mac_caps = themeCapabilitiesFor(PlatformId::Mac);
    auto const & windows_caps = themeCapabilitiesFor(PlatformId::Windows);

    // Every platform follows the system and can pin a variant, whether through
    // the native hint or a carried palette.
    EXPECT_TRUE(linux_caps.can_follow_system);
    EXPECT_TRUE(mac_caps.can_follow_system);
    EXPECT_TRUE(windows_caps.can_follow_system);
    EXPECT_TRUE(linux_caps.can_force_variant);
    EXPECT_TRUE(mac_caps.can_force_variant);
    EXPECT_TRUE(windows_caps.can_force_variant);

    // macOS and Windows own their title bar; Linux does not, so the record
    // genuinely distinguishes the platforms.
    EXPECT_TRUE(mac_caps.controls_titlebar);
    EXPECT_TRUE(windows_caps.controls_titlebar);
    EXPECT_FALSE(linux_caps.controls_titlebar);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
