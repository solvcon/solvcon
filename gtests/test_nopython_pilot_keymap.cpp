/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/keymap.hpp>

#include <array>
#include <optional>
#include <variant>

#include <gtest/gtest.h>

namespace
{

constexpr std::array<solvcon::PlatformId, 3> PLATFORMS = {
    solvcon::PlatformId::Linux, solvcon::PlatformId::Mac, solvcon::PlatformId::Windows};

} /* end namespace */

TEST(PilotKeymapTable, SeedsSharedBindingsOnEveryPlatform)
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

        auto const * console = solvcon::bindingFor(platform, solvcon::ShortcutCommand::Console);
        ASSERT_NE(console, nullptr);
        EXPECT_EQ(std::get<solvcon::KeyChord>(console->key),
                  (solvcon::KeyChord{solvcon::KeyMod::Primary, solvcon::Key::Grave}));
        EXPECT_EQ(console->context, solvcon::ShortcutContext::Window);

        struct PanelChord
        {
            solvcon::ShortcutCommand command;
            solvcon::Key key;
        }; /* end struct PanelChord */
        for (auto const & panel : {
                 PanelChord{solvcon::ShortcutCommand::AgentPanel, solvcon::Key::A},
                 PanelChord{solvcon::ShortcutCommand::InspectorPanel, solvcon::Key::I},
                 PanelChord{solvcon::ShortcutCommand::PainterPanel, solvcon::Key::P},
             })
        {
            auto const * binding = solvcon::bindingFor(platform, panel.command);
            ASSERT_NE(binding, nullptr);
            EXPECT_EQ(std::get<solvcon::KeyChord>(binding->key),
                      (solvcon::KeyChord{solvcon::KeyMod::Primary | solvcon::KeyMod::Shift, panel.key}));
        }

        auto const * blank = solvcon::bindingFor(platform, solvcon::ShortcutCommand::New2DCanvas);
        ASSERT_NE(blank, nullptr);
        EXPECT_EQ(std::get<solvcon::StandardAction>(blank->key), solvcon::StandardAction::New);

        auto const * exit = solvcon::bindingFor(platform, solvcon::ShortcutCommand::Exit);
        ASSERT_NE(exit, nullptr);
        EXPECT_EQ(std::get<solvcon::StandardAction>(exit->key), solvcon::StandardAction::Quit);
        EXPECT_EQ(exit->context, solvcon::ShortcutContext::Application);
    }
}

TEST(PilotKeymapMac, AddsQuitRoleOverTheSharedExitBinding)
{
    EXPECT_TRUE(solvcon::capabilitiesFor(solvcon::PlatformId::Mac).movesItemsToApplicationMenu);
    EXPECT_EQ(solvcon::bindingTable(solvcon::PlatformId::Mac).size(),
              solvcon::bindingTable(solvcon::PlatformId::Linux).size());

    auto const * exit = solvcon::bindingFor(solvcon::PlatformId::Mac, solvcon::ShortcutCommand::Exit);
    ASSERT_NE(exit, nullptr);
    EXPECT_EQ(std::get<solvcon::StandardAction>(exit->key), solvcon::StandardAction::Quit);
    EXPECT_EQ(exit->role, solvcon::MenuRole::Quit);

    EXPECT_EQ(solvcon::bindingFor(solvcon::PlatformId::Linux, solvcon::ShortcutCommand::Exit)->role,
              solvcon::MenuRole::None);
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

TEST(PilotKeymapId, CommandFromIdInvertsCommandId)
{
    for (auto command : solvcon::ALL_SHORTCUT_COMMANDS)
    {
        auto found = solvcon::commandFromId(solvcon::commandId(command));
        ASSERT_TRUE(found.has_value());
        EXPECT_EQ(*found, command);
    }
    EXPECT_FALSE(solvcon::commandFromId("no.such.command").has_value());
}

TEST(PilotKeymapId, ContextAndRoleHaveStableNames)
{
    EXPECT_EQ(solvcon::contextName(solvcon::ShortcutContext::Application), "application");
    EXPECT_EQ(solvcon::contextName(solvcon::ShortcutContext::Window), "window");
    EXPECT_EQ(solvcon::contextName(solvcon::ShortcutContext::Widget), "widget");
    EXPECT_EQ(solvcon::roleName(solvcon::MenuRole::None), "none");
    EXPECT_EQ(solvcon::roleName(solvcon::MenuRole::Quit), "quit");
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
