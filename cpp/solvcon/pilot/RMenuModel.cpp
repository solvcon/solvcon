/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RMenuModel.hpp> // Must be the first include.

#include <limits>

#include <QAction>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QStringList>
#include <Qt>

namespace solvcon
{

namespace
{

/// A negative weight sorts last, so a node declared without a weight lands
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
    return cur->menu;
}

void RMenuModel::clear()
{
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
    m_root.children.clear();
}

RMenuModel::Node * RMenuModel::ensureChild(Node * parent, QString const & name, int weight)
{
    for (std::unique_ptr<Node> const & child : parent->children)
    {
        if (child->name == name)
        {
            if (weight >= 0 && weight != child->weight)
            {
                reweight(parent, child.get(), weight);
            }
            return child.get();
        }
    }

    auto node = std::make_unique<Node>();
    node->name = name;
    node->weight = weight;
    node->parent = parent;
    node->menu = new QMenu(name, m_mainWindow);
    m_menus.emplace_back(node->menu);

    int const idx = insertionIndex(parent, weight);
    QAction * before = idx < static_cast<int>(parent->children.size())
                           ? parent->children.at(idx)->menu->menuAction()
                           : nullptr;
    containerInsert(parent, node->menu, before);

    Node * raw = node.get();
    parent->children.insert(parent->children.begin() + idx, std::move(node));
    return raw;
}

void RMenuModel::reweight(Node * parent, Node * node, int weight)
{
    // Detach from the Qt container and the sibling vector, then re-insert
    // where the new weight places the node.
    containerRemove(parent, node->menu);

    int old = 0;
    for (; old < static_cast<int>(parent->children.size()); ++old)
    {
        if (parent->children.at(old).get() == node)
        {
            break;
        }
    }
    std::unique_ptr<Node> held = std::move(parent->children.at(old));
    parent->children.erase(parent->children.begin() + old);

    node->weight = weight;
    int const idx = insertionIndex(parent, weight);
    QAction * before = idx < static_cast<int>(parent->children.size())
                           ? parent->children.at(idx)->menu->menuAction()
                           : nullptr;
    containerInsert(parent, node->menu, before);
    parent->children.insert(parent->children.begin() + idx, std::move(held));
}

int RMenuModel::insertionIndex(Node * parent, int weight) const
{
    int const ew = effective_weight(weight);
    int idx = 0;
    for (; idx < static_cast<int>(parent->children.size()); ++idx)
    {
        if (effective_weight(parent->children.at(idx)->weight) > ew)
        {
            break;
        }
    }
    return idx;
}

void RMenuModel::containerInsert(Node * parent, QMenu * menu, QAction * before)
{
    if (parent == &m_root)
    {
        before ? m_bar->insertMenu(before, menu) : m_bar->addMenu(menu);
    }
    else
    {
        before ? parent->menu->insertMenu(before, menu) : parent->menu->addMenu(menu);
    }
}

void RMenuModel::containerRemove(Node * parent, QMenu * menu)
{
    QWidget * container = parent == &m_root
                              ? static_cast<QWidget *>(m_bar)
                              : static_cast<QWidget *>(parent->menu);
    container->removeAction(menu->menuAction());
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
