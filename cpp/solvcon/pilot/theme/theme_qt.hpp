#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The one bridge from the Qt-free theme foundation into Qt's own color type.
 *
 * theme.hpp deliberately mentions no Qt so the color tables compile into the
 * no-GUI test target, which leaves every Qt consumer needing the same three
 * field copies. They belong here, once, rather than as a private lambda in
 * each adapter.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/theme/theme.hpp>

#include <QColor>

namespace solvcon
{

/// The Qt color for one theme color.
inline QColor qcolor(ThemeColor c) { return QColor(c.r, c.g, c.b); }

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
