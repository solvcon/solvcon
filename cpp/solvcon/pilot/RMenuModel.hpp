#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Live model of the pilot menu bar: menus addressed by path and weight.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <memory>
#include <string>
#include <vector>

#include <QObject>
#include <QPointer>
#include <QString>

class QAction;
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
 * resolves a path, creating any missing ancestor on demand, and orders each
 * node among its siblings (and on the bar for a top-level path) by a weight
 * declared once. A smaller weight sits earlier; equal weights keep arrival
 * order; a negative weight leaves an existing node untouched and appends a new
 * one. The model owns the menus it creates, so clear() empties the bar and
 * lets a fresh setUp rebuild it.
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

    /// Delete every menu the model created and empty the bar.
    void clear();

private:

    struct Node
    {
        QString name;
        int weight = -1;
        QPointer<QMenu> menu; // nullptr for the root, which stands for the bar
        Node * parent = nullptr;
        std::vector<std::unique_ptr<Node>> children;
    };

    Node * ensureChild(Node * parent, QString const & name, int weight);
    void reweight(Node * parent, Node * node, int weight);
    int insertionIndex(Node * parent, int weight) const;
    void containerInsert(Node * parent, QMenu * menu, QAction * before);
    void containerRemove(Node * parent, QMenu * menu);

    QMainWindow * m_mainWindow = nullptr;
    QMenuBar * m_bar = nullptr;
    Node m_root;
    std::vector<QPointer<QMenu>> m_menus;
}; /* end class RMenuModel */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
