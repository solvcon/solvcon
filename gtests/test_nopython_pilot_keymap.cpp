/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/keymap.hpp>

#include <array>
#include <variant>

#include <gtest/gtest.h>

namespace
{

constexpr std::array<solvcon::PlatformId, 3> PLATFORMS = {
    solvcon::PlatformId::Linux, solvcon::PlatformId::Mac, solvcon::PlatformId::Windows};

} /* end namespace */

TEST(PilotKeymapTable, SeedsUndoRedoAndResetOnEveryPlatform)
{
    for (auto platform : PLATFORMS)
    {
        auto const * undo = solvcon::bindingFor(platform, solvcon::ShortcutCommand::Undo);
        ASSERT_NE(undo, nullptr);
        EXPECT_EQ(std::get<solvcon::StandardAction>(undo->key), solvcon::StandardAction::Undo);
        EXPECT_EQ(undo->context, solvcon::ShortcutContext::Window);

        auto const * redo = solvcon::bindingFor(platform, solvcon::ShortcutCommand::Redo);
        ASSERT_NE(redo, nullptr);
        EXPECT_EQ(std::get<solvcon::StandardAction>(redo->key), solvcon::StandardAction::Redo);

        auto const * reset = solvcon::bindingFor(platform, solvcon::ShortcutCommand::CameraReset);
        ASSERT_NE(reset, nullptr);
        EXPECT_EQ(std::get<solvcon::KeyChord>(reset->key),
                  (solvcon::KeyChord{solvcon::KeyMod::None, solvcon::Key::Escape}));
        EXPECT_EQ(reset->context, solvcon::ShortcutContext::Widget);
    }
}

TEST(PilotKeymapMac, AddsQuitRoleAndAppMenuOverTheSeed)
{
    EXPECT_TRUE(solvcon::capabilitiesFor(solvcon::PlatformId::Mac).movesItemsToApplicationMenu);
    EXPECT_EQ(solvcon::bindingTable(solvcon::PlatformId::Mac).size(),
              solvcon::bindingTable(solvcon::PlatformId::Linux).size() + 1);

    auto const * exit = solvcon::bindingFor(solvcon::PlatformId::Mac, solvcon::ShortcutCommand::Exit);
    ASSERT_NE(exit, nullptr);
    EXPECT_TRUE(std::holds_alternative<solvcon::Unbound>(exit->key));
    EXPECT_EQ(exit->role, solvcon::MenuRole::Quit);
}

TEST(PilotKeymapCapabilities, OnlyMacMovesItemsToApplicationMenu)
{
    EXPECT_TRUE(solvcon::capabilitiesFor(solvcon::PlatformId::Mac).movesItemsToApplicationMenu);
    EXPECT_FALSE(solvcon::capabilitiesFor(solvcon::PlatformId::Linux).movesItemsToApplicationMenu);
    EXPECT_FALSE(solvcon::capabilitiesFor(solvcon::PlatformId::Windows).movesItemsToApplicationMenu);
}

TEST(PilotKeymapContext, FollowsConservativeOverlapRule)
{
    EXPECT_TRUE(solvcon::contextsOverlap(solvcon::ShortcutContext::Application, solvcon::ShortcutContext::Widget));
    EXPECT_TRUE(solvcon::contextsOverlap(solvcon::ShortcutContext::Window, solvcon::ShortcutContext::Widget));
    EXPECT_TRUE(solvcon::contextsOverlap(solvcon::ShortcutContext::Widget, solvcon::ShortcutContext::Window));
    // Two widget scopes stay conservatively overlapping until proven exclusive.
    EXPECT_TRUE(solvcon::contextsOverlap(solvcon::ShortcutContext::Widget, solvcon::ShortcutContext::Widget));
}

TEST(PilotKeymapConflicts, DefaultTablesHaveNoDeclaredConflicts)
{
    for (auto platform : PLATFORMS)
    {
        EXPECT_TRUE(solvcon::findDeclaredConflicts(platform).empty());
    }
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
