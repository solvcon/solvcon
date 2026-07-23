/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/app/RShortcutManager.hpp> // Must be the first include.

#include <cstddef>
#include <stdexcept>
#include <variant>

#include <QAction>
#include <QKeyCombination>
#include <Qt>
#include <QtGlobal>

namespace solvcon
{

namespace
{

PlatformId detectRunningPlatform()
{
    // A thin Qt call, the one place the roof asks the operating system which
    // phrasebook to read. The keymap core stays free of this branch.
#if defined(Q_OS_MACOS)
    return PlatformId::Mac;
#elif defined(Q_OS_WIN)
    return PlatformId::Windows;
#else
    return PlatformId::Linux;
#endif
}

bool hasMod(KeyMod set, KeyMod flag)
{
    return (static_cast<unsigned>(set) & static_cast<unsigned>(flag)) != 0;
}

QKeySequence::StandardKey toStandardKey(StandardAction action)
{
    switch (action)
    {
    case StandardAction::Undo:
        return QKeySequence::Undo;
    case StandardAction::Redo:
        return QKeySequence::Redo;
    case StandardAction::Quit:
        return QKeySequence::Quit;
    case StandardAction::New:
        return QKeySequence::New;
    }
    throw std::logic_error("Unexpected standard action");
}

std::string standardKeyName(StandardAction action)
{
    switch (action)
    {
    case StandardAction::Undo:
        return "Undo";
    case StandardAction::Redo:
        return "Redo";
    case StandardAction::Quit:
        return "Quit";
    case StandardAction::New:
        return "New";
    }
    throw std::logic_error("Unexpected standard action");
}

Qt::Key toQtKey(Key key)
{
    switch (key)
    {
    case Key::Escape:
        return Qt::Key_Escape;
    case Key::Grave:
        return Qt::Key_QuoteLeft;
    case Key::Tab:
        return Qt::Key_Tab;
    case Key::Space:
        return Qt::Key_Space;
    case Key::F4:
        return Qt::Key_F4;
    case Key::A:
        return Qt::Key_A;
    case Key::D:
        return Qt::Key_D;
    case Key::I:
        return Qt::Key_I;
    case Key::P:
        return Qt::Key_P;
    case Key::S:
        return Qt::Key_S;
    case Key::W:
        return Qt::Key_W;
    case Key::Z:
        return Qt::Key_Z;
    }
    throw std::logic_error("Unexpected key");
}

Qt::ShortcutContext toQtContext(ShortcutContext context)
{
    switch (context)
    {
    case ShortcutContext::Application:
        return Qt::ApplicationShortcut;
    case ShortcutContext::Window:
        return Qt::WindowShortcut;
    case ShortcutContext::Widget:
        return Qt::WidgetShortcut;
    }
    throw std::logic_error("Unexpected context");
}

QAction::MenuRole toQtMenuRole(MenuRole role)
{
    switch (role)
    {
    case MenuRole::None:
        return QAction::NoRole;
    case MenuRole::Quit:
        return QAction::QuitRole;
    case MenuRole::Preferences:
        return QAction::PreferencesRole;
    case MenuRole::About:
        return QAction::AboutRole;
    }
    throw std::logic_error("Unexpected role");
}

} /* end namespace */

RShortcutManager::RShortcutManager(QObject * parent)
    : QObject(parent)
    , m_platform(detectRunningPlatform())
{
}

RShortcutManager::~RShortcutManager() = default;

ShortcutCapabilities RShortcutManager::capabilities() const
{
    return capabilitiesFor(m_platform);
}

QList<QKeySequence> RShortcutManager::sequencesForBinding(ShortcutBinding const & binding) const
{
    if (auto const * standard = std::get_if<StandardAction>(&binding.key))
    {
        return QKeySequence::keyBindings(toStandardKey(*standard));
    }
    if (auto const * chord = std::get_if<KeyChord>(&binding.key))
    {
        // Qt maps ControlModifier to Command on macOS, so a Primary chord
        // pronounces itself on every platform without a native branch here.
        // The physical Control key is a separate key on macOS, reached
        // through MetaModifier because Qt swaps Control and Meta there.
        Qt::KeyboardModifiers mods = Qt::NoModifier;
        if (hasMod(chord->mods, KeyMod::Primary))
        {
            mods |= Qt::ControlModifier;
        }
        if (hasMod(chord->mods, KeyMod::Control))
        {
            mods |= m_platform == PlatformId::Mac ? Qt::MetaModifier : Qt::ControlModifier;
        }
        if (hasMod(chord->mods, KeyMod::Shift))
        {
            mods |= Qt::ShiftModifier;
        }
        if (hasMod(chord->mods, KeyMod::Alt))
        {
            mods |= Qt::AltModifier;
        }
        return {QKeySequence(QKeyCombination(mods, toQtKey(chord->key)))};
    }
    return {};
}

QList<QKeySequence> RShortcutManager::sequencesFor(ShortcutCommand command) const
{
    ShortcutBinding const * binding = bindingFor(m_platform, command);
    if (binding == nullptr)
    {
        return {};
    }
    return sequencesForBinding(*binding);
}

void RShortcutManager::applyTo(QAction * action, ShortcutCommand command) const
{
    if (action == nullptr)
    {
        return;
    }
    ShortcutBinding const * binding = bindingFor(m_platform, command);
    if (binding == nullptr)
    {
        return;
    }

    // The role is set before the action reaches a menu, as macOS requires to
    // move Quit into the application menu. The context applies to every case,
    // so re-applying a binding never leaves a stale scope behind.
    action->setMenuRole(toQtMenuRole(binding->role));
    action->setShortcutContext(toQtContext(binding->context));

    action->setShortcuts(sequencesForBinding(*binding));
}

ResolvedShortcut RShortcutManager::resolve(ShortcutCommand command) const
{
    ResolvedShortcut resolved;
    ShortcutBinding const * binding = bindingFor(m_platform, command);
    if (binding == nullptr)
    {
        return resolved;
    }

    resolved.context = binding->context;
    resolved.role = binding->role;

    if (auto const * standard = std::get_if<StandardAction>(&binding->key))
    {
        resolved.bound = true;
        resolved.standard = true;
        resolved.standard_key = standardKeyName(*standard);
    }
    else if (std::holds_alternative<KeyChord>(binding->key))
    {
        resolved.bound = true;
    }

    for (QKeySequence const & seq : sequencesForBinding(*binding))
    {
        resolved.sequences.push_back(seq.toString(QKeySequence::PortableText).toStdString());
    }
    return resolved;
}

ResolvedShortcut RShortcutManager::resolveById(std::string const & id) const
{
    if (auto command = commandFromId(id))
    {
        return resolve(*command);
    }
    return {};
}

std::vector<ShortcutConflict> RShortcutManager::findResolvedConflicts() const
{
    return findResolvedConflictsIn(bindingTable(m_platform));
}

std::vector<ShortcutConflict> RShortcutManager::findResolvedConflictsIn(std::vector<ShortcutBinding> const & table) const
{
    std::vector<QList<QKeySequence>> resolved;
    resolved.reserve(table.size());
    for (ShortcutBinding const & binding : table)
    {
        resolved.push_back(sequencesForBinding(binding));
    }

    auto bound = [&resolved](std::size_t i)
    { return !resolved[i].isEmpty(); };
    auto sequencesClash = [&resolved](std::size_t i, std::size_t j)
    {
        for (QKeySequence const & lhs : resolved[i])
        {
            if (resolved[j].contains(lhs))
            {
                return true;
            }
        }
        return false;
    };
    return findConflictsIn(table, bound, sequencesClash);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
