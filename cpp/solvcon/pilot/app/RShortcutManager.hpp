#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The Qt adapter over the keymap core: the roof of the pilot's shortcut
 * system. It detects the running platform, turns a command id into the native
 * key sequences that platform spells it with, applies them to a QAction, and
 * checks the resolved bindings for collisions Qt's standard keys can hide from
 * the Qt-free core. Half the pilot's commands are born in Python, so it also
 * reports a binding as portable values a PySide helper can apply.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <string>
#include <vector>

#include <solvcon/pilot/app/keymap.hpp>

#include <QList>
#include <QKeySequence>
#include <QObject>

class QAction;

namespace solvcon
{

/**
 * @brief A command's binding resolved to values that cross the Python boundary.
 *
 * The manager cannot hand a QAction across the pybind and PySide divide, so a
 * Python-registered action is bound from these plain values instead. A standard
 * binding names its Qt standard key so PySide can call setShortcuts with it;
 * a curated binding carries its resolved sequence strings. bound is false when
 * the command has no key on this platform, though role and context still apply
 * for an Unbound table entry.
 */
struct ResolvedShortcut
{
    bool bound = false;
    bool standard = false;
    std::string standard_key;
    std::vector<std::string> sequences;
    ShortcutContext context = ShortcutContext::Window;
    MenuRole role = MenuRole::None;
};

/**
 * @brief Resolves the keymap core's per-platform tables into live Qt bindings.
 *
 * @ingroup group_domain
 *
 * Built once and asked for a command's binding by id. The C++ side applies a
 * binding straight to its QAction through applyTo; the Python side asks for the
 * same binding as portable values and applies them with PySide. shortcutsChanged
 * is scaffolding for the later customization layer and has no consumer yet.
 */
class RShortcutManager
    : public QObject
{
    Q_OBJECT

public:

    explicit RShortcutManager(QObject * parent = nullptr);

    ~RShortcutManager() override;

    /// The running platform, naming the table the bindings resolve from.
    PlatformId platform() const { return m_platform; }

    /// What the running platform's keymap can honor.
    ShortcutCapabilities capabilities() const;

    /**
     * The native key sequences for @p command on the running platform, empty
     * when the command carries no binding. A standard action can expose more
     * than one native sequence, so the result is a list.
     */
    QList<QKeySequence> sequencesFor(ShortcutCommand command) const;

    /**
     * Apply @p command's binding to @p action: its sequences, shortcut context,
     * and menu role. The role is set first, as macOS requires before placement.
     * A no-op on a null action or a command with no binding.
     */
    void applyTo(QAction * action, ShortcutCommand command) const;

    /// @p command's binding as portable values for the Python layer to apply.
    ResolvedShortcut resolve(ShortcutCommand command) const;

    /**
     * The binding for the action objectName @p id, or an unbound result when
     * no command carries that id.
     */
    ResolvedShortcut resolveById(std::string const & id) const;

    /**
     * Commands whose resolved sequences collide in contexts that can be active
     * together, under the same conservative overlap rule the core uses. This
     * catches a curated chord clashing with a Qt standard key, which the
     * declared checker cannot see because it leaves standard keys symbolic.
     */
    std::vector<ShortcutConflict> findResolvedConflicts() const;

    /**
     * Same as findResolvedConflicts, but over an explicit table so a test can
     * rebind one command and prove the resolved checker reports the clash.
     */
    std::vector<ShortcutConflict> findResolvedConflictsIn(std::vector<ShortcutBinding> const & table) const;

signals:

    /**
     * Emitted when the resolved bindings change. No caller consumes it in the
     * default-binding steps; it is scaffolding for user rebinding later.
     */
    void shortcutsChanged();

private:

    QList<QKeySequence> sequencesForBinding(ShortcutBinding const & binding) const;

    PlatformId m_platform;

}; /* end class RShortcutManager */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
