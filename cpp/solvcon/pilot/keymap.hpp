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

#include <string_view>
#include <variant>
#include <vector>

namespace solvcon
{

enum class PlatformId
{
    Linux,
    Mac,
    Windows
};

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

/// A modifier role, written against the command modifier rather than a
/// physical key: Primary is Command on macOS and Control elsewhere. A flag
/// set so a chord can name more than one modifier; the roof adds the bitwise
/// combine and test helpers when it resolves a chord into Qt modifiers.
enum class KeyMod : unsigned
{
    None = 0,
    Primary = 1u << 0,
    Shift = 1u << 1,
    Alt = 1u << 2
};

/// A physical key a curated chord names. The vocabulary stays small: only the
/// keys the pilot binds today, growing as later steps add curated chords.
/// Arrow keys are deliberately absent; they belong to
/// RDomainWidget::keyPressEvent, not the action system.
enum class Key
{
    Escape
};

struct KeyChord
{
    KeyMod mods;
    Key key;
}; /* end struct KeyChord */

constexpr bool operator==(KeyChord lhs, KeyChord rhs)
{
    return lhs.mods == rhs.mods && lhs.key == rhs.key;
}

struct Unbound
{
}; /* end struct Unbound */

/// A key spelling is either a standard action, a curated chord, or unbound.
/// Unbound is a placeholder for a command that has no binding on a platform yet.
/// StandardAction is a placeholder for a command that defers to Qt's standard sequence for that action.
/// KeyChord is a curated chord that the pilot binds to a command.
using KeySpelling = std::variant<Unbound, StandardAction, KeyChord>;

/// A command named by the objectName the pilot already gives its action, so
/// the vocabulary names existing commands across both the C++ and Python
/// layers rather than inventing ids.
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
}; /* end struct ShortcutBinding */

struct ShortcutCapabilities
{
    PlatformId platform;
    bool movesItemsToApplicationMenu;
    std::vector<KeyChord> reservedSequences;
}; /* end struct ShortcutCapabilities */

struct ShortcutConflict
{
    ShortcutCommand first;
    ShortcutCommand second;
}; /* end struct ShortcutConflict */

std::string_view commandId(ShortcutCommand command);

std::vector<ShortcutBinding> const & bindingTable(PlatformId platform);

/// The binding for @p command on @p platform, or nullptr when the command
/// carries none on that platform yet.
ShortcutBinding const * bindingFor(PlatformId platform, ShortcutCommand command);

ShortcutCapabilities capabilitiesFor(PlatformId platform);

/// Whether two contexts can be active together, under the conservative rule:
/// Application overlaps every context, Window overlaps Window and Widget, and
/// Widget overlaps Widget. Symmetric.
bool contextsOverlap(ShortcutContext lhs, ShortcutContext rhs);

std::vector<ShortcutConflict> findDeclaredConflicts(PlatformId platform);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
