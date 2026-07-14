#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Qt-free core of the pilot keyboard-shortcut system: the shared command
 * vocabulary, the per-platform binding tables, the capability records, and
 * the declared-conflict rule. Kept free of Qt so every CI runner can unit
 * test all three platforms' tables without a display. The Qt adapter that
 * resolves these into live QKeySequence values lives in RShortcutManager.
 *
 * @ingroup group_domain
 */

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

#include <solvcon/pilot/platform.hpp>

namespace solvcon
{

enum class ShortcutContext
{
    Application,
    Window,
    Widget
};

enum class MenuRole
{
    None,
    Quit,
    Preferences,
    About
};

enum class StandardAction
{
    Undo,
    Redo
};

/**
 * A modifier role, written against the command modifier rather than a
 * physical key: Primary is Command on macOS and Control elsewhere. A flag
 * set so a chord can name more than one modifier; the roof adds the bitwise
 * combine and test helpers when it resolves a chord into Qt modifiers.
 */
enum class KeyMod : unsigned
{
    None = 0,
    Primary = 1u << 0,
    Shift = 1u << 1,
    Alt = 1u << 2
};

/**
 * A physical key a curated chord names. The vocabulary stays small: only the
 * keys the pilot binds today, growing as later steps add curated chords.
 * Arrow keys are deliberately absent; they belong to
 * RDomainWidget::keyPressEvent, not the action system.
 */
enum class Key
{
    Escape
};

struct KeyChord
{
    KeyMod mods;
    Key key;
};

constexpr bool operator==(KeyChord lhs, KeyChord rhs)
{
    return lhs.mods == rhs.mods && lhs.key == rhs.key;
}

struct Unbound
{
};

/**
 * A key spelling is either a standard action, a curated chord, or unbound.
 * Unbound is a placeholder for a command that has no binding on a platform yet.
 * StandardAction is a placeholder for a command that defers to Qt's standard sequence for that action.
 * KeyChord is a curated chord that the pilot binds to a command.
 */
using KeySpelling = std::variant<Unbound, StandardAction, KeyChord>;

/**
 * A command named by the objectName the pilot already gives its action, so
 * the vocabulary names existing commands across both the C++ and Python
 * layers rather than inventing ids.
 */
enum class ShortcutCommand
{
    Undo,
    Redo,
    CameraReset,
    Exit,
    Console,
    AgentPanel
};

struct ShortcutBinding
{
    ShortcutCommand command;
    KeySpelling key;
    ShortcutContext context;
    MenuRole role = MenuRole::None;
};

/**
 * Every command the vocabulary names, the single list the id lookups and their
 * tests iterate. Adding a command here is what forces commandId and
 * commandFromId to stay in step, so neither can silently miss one.
 */
inline constexpr auto ALL_SHORTCUT_COMMANDS = std::to_array<ShortcutCommand>(
    {ShortcutCommand::Undo,
     ShortcutCommand::Redo,
     ShortcutCommand::CameraReset,
     ShortcutCommand::Exit,
     ShortcutCommand::Console,
     ShortcutCommand::AgentPanel});

struct ShortcutCapabilities
{
    PlatformId platform;
    bool movesItemsToApplicationMenu;
    std::vector<KeyChord> reservedSequences;
};

struct ShortcutConflict
{
    ShortcutCommand first;
    ShortcutCommand second;
};

std::string_view commandId(ShortcutCommand command);

/**
 * The stable id of a context ("application", "window", "widget"), for the
 * Python boundary and tests.
 */
std::string_view contextName(ShortcutContext context);

/**
 * The stable id of a menu role ("none", "quit", "preferences", "about"), for
 * the Python boundary and tests.
 */
std::string_view roleName(MenuRole role);

/**
 * The command named by @p id (the action objectName), or an empty optional
 * when no command carries that id. The inverse of commandId, so the Python
 * layer can resolve a binding from an action's objectName.
 */
std::optional<ShortcutCommand> commandFromId(std::string_view id);

std::vector<ShortcutBinding> const & bindingTable(PlatformId platform);

/**
 * The binding for @p command on @p platform, or nullptr when the command
 * carries none on that platform yet.
 */
ShortcutBinding const * bindingFor(PlatformId platform, ShortcutCommand command);

ShortcutCapabilities capabilitiesFor(PlatformId platform);

/**
 * Whether two contexts can be active together, under the conservative rule:
 * Application overlaps every context, Window overlaps Window and Widget, and
 * Widget overlaps Widget. Symmetric.
 */
bool contextsOverlap(ShortcutContext lhs, ShortcutContext rhs);

/**
 * The pairwise-conflict skeleton over @p platform's binding table: a command
 * pair collides when both indices are @p eligible, their contexts overlap, and
 * @p equal reports their keys the same. The core injects a symbolic chord
 * comparison; the Qt roof injects a resolved-sequence comparison, so the loop
 * and the overlap rule live here once. Both predicates take table indices.
 */
template <typename Eligible, typename Equal>
std::vector<ShortcutConflict>
findConflicts(PlatformId platform, Eligible eligible, Equal equal)
{
    std::vector<ShortcutBinding> const & table = bindingTable(platform);

    std::vector<ShortcutConflict> conflicts;
    for (std::size_t i = 0; i < table.size(); ++i)
    {
        if (!eligible(i))
        {
            continue;
        }
        for (std::size_t j = i + 1; j < table.size(); ++j)
        {
            if (!eligible(j) || !contextsOverlap(table[i].context, table[j].context))
            {
                continue;
            }
            if (equal(i, j))
            {
                conflicts.push_back({table[i].command, table[j].command});
            }
        }
    }
    return conflicts;
}

std::vector<ShortcutConflict> findDeclaredConflicts(PlatformId platform);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
