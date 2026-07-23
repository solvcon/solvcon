#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Live model of the pilot menu bar: menus by path, actions by id and weight.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <QObject>
#include <QPointer>
#include <QString>

class QAction;
class QActionGroup;
class QMainWindow;
class QMenu;
class QMenuBar;

namespace solvcon
{

/**
 * @brief A live, incrementally built model of the pilot menu bar.
 *
 * @ingroup group_domain
 *
 * Menus are addressed by a slash-separated path ("View/Panels"). menu()
 * resolves a path, creating any missing ancestor on demand, and place() drops
 * an action (or a separator) into the menu at a path. Every entry in a menu,
 * whether a submenu or an item, is ordered by a weight: a smaller weight sits
 * earlier, equal weights keep arrival order, and a negative weight appends.
 * The weight-with-gaps ordering follows the convention plugin systems use for
 * contributed items (for example Drupal's #weight): a declared integer key,
 * with room left between values to slot an entry in without renumbering.
 *
 * Placed actions are registered under their objectName, so action(id) looks
 * one up and remove(id) takes it out. group(id) returns a named QActionGroup
 * for exclusive selections. The model owns the menus and groups it creates, so
 * clear() empties the bar and lets a fresh setUp rebuild it.
 */
class RMenuModel
    : public QObject
{
    Q_OBJECT

public:

    explicit RMenuModel(QMainWindow * mainWindow, QObject * parent = nullptr);

    ~RMenuModel() override;

    /// Resolve or create the menu at @p path, creating ancestors on demand.
    /// @p weight orders the final node among its siblings; a negative weight
    /// leaves an existing node's order untouched and appends a new one.
    QMenu * menu(std::string const & path, int weight = -1);

    /// Place @p action in the menu at @p path, ordered by @p weight among the
    /// menu's current entries. The action is registered under its objectName
    /// when that name is set.
    void place(std::string const & path, QAction * action, int weight = 50);

    /// Place a separator in the menu at @p path, ordered by @p weight.
    void placeSeparator(std::string const & path, int weight = 50);

    /// The action registered under @p id, or nullptr when none is.
    QAction * action(std::string const & id) const;

    /// Remove and unregister the action registered under @p id, if any.
    void remove(std::string const & id);

    /// The named action group, created on first use.
    QActionGroup * group(std::string const & id);

    /// Delete every menu and group the model created and empty the bar.
    void clear();

private:

    struct Node;

    struct Entry
    {
        int weight = -1;
        QPointer<QAction> action; // the entry's QAction (item, separator, or
                                  // a submenu's menuAction)
        std::unique_ptr<Node> submenu; // non-null only for a submenu entry
    }; /* end struct Entry */

    struct Node
    {
        QString name;
        int weight = -1;
        QPointer<QMenu> menu; // nullptr for the root, which stands for the bar
        Node * parent = nullptr;
        std::vector<Entry> entries;
    }; /* end struct Node */

    Node * resolve(std::string const & path, int weight);
    Node * ensureChild(Node * parent, QString const & name, int weight);
    void placeAction(Node * node, QAction * action, int weight);
    int insertionIndex(Node * node, int weight) const;
    void reweightEntry(Node * node, int index, int weight);
    QAction * beforeAt(Node * node, int index) const;
    void containerInsert(Node * node, QAction * action, QAction * before);
    void containerRemove(Node * node, QAction * action);
    bool removeAction(Node * node, QAction * target);

    QMainWindow * m_mainWindow = nullptr;
    QMenuBar * m_bar = nullptr;
    Node m_root;
    std::vector<QPointer<QMenu>> m_menus;
    std::map<std::string, QPointer<QAction>> m_actions;
    std::map<std::string, QPointer<QActionGroup>> m_groups;
}; /* end class RMenuModel */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
