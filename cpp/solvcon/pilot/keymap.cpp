/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/keymap.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <variant>

namespace solvcon
{

namespace
{

std::vector<ShortcutBinding> tableWith(std::vector<ShortcutBinding> extra)
{
    std::vector<ShortcutBinding> rows = {
        {ShortcutCommand::Undo, StandardAction::Undo, ShortcutContext::Window},
        {ShortcutCommand::Redo, StandardAction::Redo, ShortcutContext::Window},
        {ShortcutCommand::CameraReset, KeyChord{KeyMod::None, Key::Escape}, ShortcutContext::Widget},
    };
    rows.insert(rows.end(), extra.begin(), extra.end());
    return rows;
}

std::vector<ShortcutBinding> const & linuxTable()
{
    static std::vector<ShortcutBinding> const table = tableWith({});
    return table;
}

std::vector<ShortcutBinding> const & windowsTable()
{
    static std::vector<ShortcutBinding> const table = tableWith({});
    return table;
}

std::vector<ShortcutBinding> const & macTable()
{
    // macOS carries the Quit role from the start; the key sequence for Quit
    // is added when file.exit begins routing through the manager.
    static std::vector<ShortcutBinding> const table =
        tableWith({{ShortcutCommand::Exit, Unbound{}, ShortcutContext::Application, MenuRole::Quit}});
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
    bool const appMenu = platform == PlatformId::Mac;
    return {platform, appMenu, {}};
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

std::vector<ShortcutConflict> findDeclaredConflicts(PlatformId platform)
{
    std::vector<ShortcutBinding> const & table = bindingTable(platform);
    auto isChord = [&table](std::size_t i)
    { return std::holds_alternative<KeyChord>(table[i].key); };
    auto chordsEqual = [&table](std::size_t i, std::size_t j)
    { return std::get<KeyChord>(table[i].key) == std::get<KeyChord>(table[j].key); };
    return findConflicts(platform, isChord, chordsEqual);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
