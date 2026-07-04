/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RMenuModel.hpp> // Must be the first include.

#include <limits>

#include <QAction>
#include <QActionGroup>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QStringList>
#include <Qt>

namespace solvcon
{

namespace
{

/// A negative weight sorts last, so an entry declared without a weight lands
/// after every explicitly weighted sibling.
int effective_weight(int weight)
{
    return weight < 0 ? std::numeric_limits<int>::max() : weight;
}

} /* end namespace */

RMenuModel::RMenuModel(QMainWindow * mainWindow, QObject * parent)
    : QObject(parent)
    , m_mainWindow(mainWindow)
    , m_bar(mainWindow->menuBar())
{
}

RMenuModel::~RMenuModel()
{
    clear();
}

QMenu * RMenuModel::menu(std::string const & path, int weight)
{
    Node * node = resolve(path, weight);
    return node ? node->menu.data() : nullptr;
}

void RMenuModel::place(std::string const & path, QAction * action, int weight)
{
    Node * node = resolve(path, -1);
    if (node == nullptr || action == nullptr)
    {
        return;
    }
    placeAction(node, action, weight);

    QString const id = action->objectName();
    if (!id.isEmpty())
    {
        m_actions[id.toStdString()] = action;
    }
}

void RMenuModel::placeSeparator(std::string const & path, int weight)
{
    Node * node = resolve(path, -1);
    if (node == nullptr)
    {
        return;
    }
    // Parent the separator to its menu so it is deleted with the menu; a
    // separator carries no id and stays out of the registry.
    auto * separator = new QAction(node->menu);
    separator->setSeparator(true);
    placeAction(node, separator, weight);
}

QAction * RMenuModel::action(std::string const & id) const
{
    auto const it = m_actions.find(id);
    return it != m_actions.end() ? it->second.data() : nullptr;
}

void RMenuModel::remove(std::string const & id)
{
    auto const it = m_actions.find(id);
    if (it == m_actions.end())
    {
        return;
    }
    if (QAction * target = it->second.data())
    {
        removeAction(&m_root, target);
    }
    m_actions.erase(it);
}

QActionGroup * RMenuModel::group(std::string const & id)
{
    QPointer<QActionGroup> & slot = m_groups[id];
    if (!slot)
    {
        slot = new QActionGroup(this);
    }
    return slot.data();
}

void RMenuModel::clear()
{
    for (auto const & entry : m_groups)
    {
        if (entry.second)
        {
            delete entry.second;
        }
    }
    m_groups.clear();

    // QPointer entries drop to null when Qt already deleted a menu (for
    // example when the whole main window is torn down), so guard each delete.
    for (QPointer<QMenu> const & menu : m_menus)
    {
        if (menu)
        {
            delete menu;
        }
    }
    m_menus.clear();
    m_actions.clear();
    m_root.entries.clear();
}

RMenuModel::Node * RMenuModel::resolve(std::string const & path, int weight)
{
    QStringList const parts =
        QString::fromStdString(path).split('/', Qt::SkipEmptyParts);
    if (parts.isEmpty())
    {
        return nullptr;
    }

    Node * cur = &m_root;
    for (int i = 0; i < parts.size(); ++i)
    {
        bool const last = (i + 1 == parts.size());
        cur = ensureChild(cur, parts.at(i), last ? weight : -1);
    }
    return cur;
}

RMenuModel::Node * RMenuModel::ensureChild(Node * parent, QString const & name, int weight)
{
    for (int i = 0; i < static_cast<int>(parent->entries.size()); ++i)
    {
        Entry & entry = parent->entries.at(i);
        if (entry.submenu && entry.submenu->name == name)
        {
            if (weight >= 0 && weight != entry.weight)
            {
                reweightEntry(parent, i, weight);
            }
            return entry.submenu.get();
        }
    }

    auto * qmenu = new QMenu(name, m_mainWindow);
    m_menus.emplace_back(qmenu);

    int const idx = insertionIndex(parent, weight);
    containerInsert(parent, qmenu->menuAction(), beforeAt(parent, idx));

    Entry entry;
    entry.weight = weight;
    entry.action = qmenu->menuAction();
    entry.submenu = std::make_unique<Node>();
    entry.submenu->name = name;
    entry.submenu->weight = weight;
    entry.submenu->menu = qmenu;
    entry.submenu->parent = parent;

    Node * raw = entry.submenu.get();
    parent->entries.insert(parent->entries.begin() + idx, std::move(entry));
    return raw;
}

void RMenuModel::placeAction(Node * node, QAction * action, int weight)
{
    int const idx = insertionIndex(node, weight);
    containerInsert(node, action, beforeAt(node, idx));

    Entry entry;
    entry.weight = weight;
    entry.action = action;
    node->entries.insert(node->entries.begin() + idx, std::move(entry));
}

int RMenuModel::insertionIndex(Node * node, int weight) const
{
    int const ew = effective_weight(weight);
    int idx = 0;
    for (; idx < static_cast<int>(node->entries.size()); ++idx)
    {
        if (effective_weight(node->entries.at(idx).weight) > ew)
        {
            break;
        }
    }
    return idx;
}

void RMenuModel::reweightEntry(Node * node, int index, int weight)
{
    // Detach the entry from the Qt container and the sibling vector, then
    // re-insert it where the new weight places it.
    QAction * act = node->entries.at(index).action;
    containerRemove(node, act);
    Entry held = std::move(node->entries.at(index));
    node->entries.erase(node->entries.begin() + index);

    held.weight = weight;
    if (held.submenu)
    {
        held.submenu->weight = weight;
    }
    int const idx = insertionIndex(node, weight);
    containerInsert(node, act, beforeAt(node, idx));
    node->entries.insert(node->entries.begin() + idx, std::move(held));
}

QAction * RMenuModel::beforeAt(Node * node, int index) const
{
    return index < static_cast<int>(node->entries.size())
               ? node->entries.at(index).action.data()
               : nullptr;
}

void RMenuModel::containerInsert(Node * node, QAction * action, QAction * before)
{
    // QMenuBar and QMenu are both QWidgets, and QWidget::insertAction renders
    // an action that carries a submenu as that submenu, so one path serves
    // items, separators, and submenus alike.
    QWidget * container = node == &m_root
                              ? static_cast<QWidget *>(m_bar)
                              : static_cast<QWidget *>(node->menu);
    if (before)
    {
        container->insertAction(before, action);
    }
    else
    {
        container->addAction(action);
    }
}

void RMenuModel::containerRemove(Node * node, QAction * action)
{
    QWidget * container = node == &m_root
                              ? static_cast<QWidget *>(m_bar)
                              : static_cast<QWidget *>(node->menu);
    container->removeAction(action);
}

bool RMenuModel::removeAction(Node * node, QAction * target)
{
    for (int i = 0; i < static_cast<int>(node->entries.size()); ++i)
    {
        Entry & entry = node->entries.at(i);
        if (!entry.submenu && entry.action == target)
        {
            containerRemove(node, target);
            node->entries.erase(node->entries.begin() + i);
            return true;
        }
        if (entry.submenu && removeAction(entry.submenu.get(), target))
        {
            return true;
        }
    }
    return false;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
