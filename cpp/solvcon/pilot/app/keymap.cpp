/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/app/keymap.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <variant>

namespace solvcon
{

namespace
{

/**
 * Bindings shared by every platform. The per-platform tables append only
 * their Exit accent on top of this seed.
 */
std::vector<ShortcutBinding> sharedSeed()
{
    return {
        {ShortcutCommand::Undo, StandardAction::Undo, ShortcutContext::Window},
        {ShortcutCommand::Redo, StandardAction::Redo, ShortcutContext::Window},
        {ShortcutCommand::CameraReset, KeyChord{KeyMod::None, Key::Escape}, ShortcutContext::Widget},
        {ShortcutCommand::AgentPanel,
         KeyChord{KeyMod::Primary | KeyMod::Shift, Key::A},
         ShortcutContext::Window},
        {ShortcutCommand::InspectorPanel,
         KeyChord{KeyMod::Primary | KeyMod::Shift, Key::I},
         ShortcutContext::Window},
        {ShortcutCommand::PainterPanel,
         KeyChord{KeyMod::Primary | KeyMod::Shift, Key::P},
         ShortcutContext::Window},
        {ShortcutCommand::New2DCanvas, StandardAction::New, ShortcutContext::Window},

        // Like VSCode, the pilot binds Console to physical Ctrl+Grave on every platform.
        {ShortcutCommand::Console, KeyChord{KeyMod::Control, Key::Grave}, ShortcutContext::Window},
    };
}

std::vector<ShortcutBinding> tableWith(std::vector<ShortcutBinding> extra)
{
    std::vector<ShortcutBinding> rows = sharedSeed();
    rows.insert(rows.end(), extra.begin(), extra.end());
    return rows;
}

std::vector<ShortcutBinding> const & linuxTable()
{
    static std::vector<ShortcutBinding> const table = tableWith({
        {ShortcutCommand::Exit, StandardAction::Quit, ShortcutContext::Application},
    });
    return table;
}

std::vector<ShortcutBinding> const & windowsTable()
{
    // Windows shares the Linux Control accent for curated chords.
    return linuxTable();
}

std::vector<ShortcutBinding> const & macTable()
{
    // macOS routes Quit through the application menu with its Quit role.
    static std::vector<ShortcutBinding> const table = tableWith({
        {ShortcutCommand::Exit,
         StandardAction::Quit,
         ShortcutContext::Application,
         MenuRole::Quit},
    });
    return table;
}

} /* end namespace */

std::string_view commandId(ShortcutCommand command)
{
    switch (command)
    {
    case ShortcutCommand::Undo:
        return "edit.undo";
    case ShortcutCommand::Redo:
        return "edit.redo";
    case ShortcutCommand::CameraReset:
        return "camera.reset";
    case ShortcutCommand::Exit:
        return "file.exit";
    case ShortcutCommand::Console:
        return "window.console";
    case ShortcutCommand::AgentPanel:
        return "panel.agent_console";
    case ShortcutCommand::InspectorPanel:
        return "panel.inspector";
    case ShortcutCommand::PainterPanel:
        return "panel.painter";
    case ShortcutCommand::New2DCanvas:
        return "canvas.blank_2d";
    }
    throw std::logic_error("Unexpected command");
}

std::string_view contextName(ShortcutContext context)
{
    switch (context)
    {
    case ShortcutContext::Application:
        return "application";
    case ShortcutContext::Window:
        return "window";
    case ShortcutContext::Widget:
        return "widget";
    }
    throw std::logic_error("Unexpected context");
}

std::string_view roleName(MenuRole role)
{
    switch (role)
    {
    case MenuRole::None:
        return "none";
    case MenuRole::Quit:
        return "quit";
    case MenuRole::Preferences:
        return "preferences";
    case MenuRole::About:
        return "about";
    }
    throw std::logic_error("Unexpected role");
}

std::optional<ShortcutCommand> commandFromId(std::string_view id)
{
    for (auto command : ALL_SHORTCUT_COMMANDS)
    {
        if (commandId(command) == id)
        {
            return command;
        }
    }
    return std::nullopt;
}

std::vector<ShortcutBinding> const & bindingTable(PlatformId platform)
{
    switch (platform)
    {
    case PlatformId::Mac:
        return macTable();
    case PlatformId::Windows:
        return windowsTable();
    case PlatformId::Linux:
        return linuxTable();
    }
    throw std::logic_error("Unexpected platform");
}

ShortcutBinding const * bindingFor(PlatformId platform, ShortcutCommand command)
{
    for (auto const & binding : bindingTable(platform))
    {
        if (binding.command == command)
        {
            return &binding;
        }
    }
    return nullptr;
}

ShortcutCapabilities capabilitiesFor(PlatformId platform)
{
    ShortcutCapabilities caps;
    caps.platform = platform;
    caps.movesItemsToApplicationMenu = platform == PlatformId::Mac;
    // Plain W/A/S/D belong to RDomainWidget::keyPressEvent on every platform.
    caps.reservedSequences = {
        {KeyMod::None, Key::W},
        {KeyMod::None, Key::A},
        {KeyMod::None, Key::S},
        {KeyMod::None, Key::D},
    };

    switch (platform)
    {
    case PlatformId::Linux:
    case PlatformId::Windows:
        // Window manager chords; curated bindings must stay clear of them.
        caps.reservedSequences.push_back({KeyMod::Alt, Key::F4});
        caps.reservedSequences.push_back({KeyMod::Alt, Key::Tab});
        if (platform == PlatformId::Windows)
        {
            caps.reservedSequences.push_back({KeyMod::Alt, Key::Space});
        }
        break;
    case PlatformId::Mac:
        // App switcher, Spotlight, and the in-app window cycle. Console
        // sits on physical Ctrl+Grave, so Cmd+Grave stays a system chord.
        caps.reservedSequences.push_back({KeyMod::Primary, Key::Tab});
        caps.reservedSequences.push_back({KeyMod::Primary, Key::Space});
        caps.reservedSequences.push_back({KeyMod::Primary, Key::Grave});
        break;
    }
    return caps;
}

bool isReservedSequence(PlatformId platform, KeyChord chord)
{
    ShortcutCapabilities const caps = capabilitiesFor(platform);
    for (KeyChord const & reserved : caps.reservedSequences)
    {
        if (reserved == chord)
        {
            return true;
        }
    }
    return false;
}

bool contextsOverlap(ShortcutContext lhs, ShortcutContext rhs)
{
    auto covers = [](ShortcutContext wide, ShortcutContext narrow)
    {
        switch (wide)
        {
        case ShortcutContext::Application:
            return true;
        case ShortcutContext::Window:
            return narrow == ShortcutContext::Window || narrow == ShortcutContext::Widget;
        case ShortcutContext::Widget:
            return narrow == ShortcutContext::Widget;
        }
        return false;
    };
    return covers(lhs, rhs) || covers(rhs, lhs);
}

std::vector<ShortcutConflict> findDeclaredConflictsIn(std::vector<ShortcutBinding> const & table)
{
    auto isChord = [&table](std::size_t i)
    { return std::holds_alternative<KeyChord>(table[i].key); };
    auto chordsEqual = [&table](std::size_t i, std::size_t j)
    { return std::get<KeyChord>(table[i].key) == std::get<KeyChord>(table[j].key); };
    return findConflictsIn(table, isChord, chordsEqual);
}

std::vector<ShortcutConflict> findDeclaredConflicts(PlatformId platform)
{
    return findDeclaredConflictsIn(bindingTable(platform));
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
