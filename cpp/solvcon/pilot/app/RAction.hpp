#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * A QAction that runs a user-supplied callback when triggered.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <QAction>

namespace solvcon
{

/**
 * A QAction that invokes a stored callback when the action is triggered.
 *
 * @ingroup group_domain
 *
 * The constructor wires the QAction text and status tip and connects the
 * triggered signal to the supplied callback, so callers can build a menu or
 * toolbar action from a single function object.
 */
class RAction
    : public QAction
{
public:

    RAction(
        QString const & text,
        QString const & tipText,
        std::function<void(void)> callback,
        QObject * parent = nullptr);
}; /* end class RAction */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
