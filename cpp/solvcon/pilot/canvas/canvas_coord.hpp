#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Map points between world and screen coordinates on the 2D canvas. A world
 * point is a Point3dFp64 and a screen point is a QPointF.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/universe/ViewTransform2d.hpp>
#include <solvcon/universe/coord.hpp>

#include <QPointF>

namespace solvcon
{

inline QPointF screen_from_world(ViewTransform2dFp64 const & view, Point3dFp64 const & w)
{
    double sx = 0.0, sy = 0.0;
    view.screen_from_world(w.x(), w.y(), sx, sy);
    return QPointF(sx, sy);
}

inline Point3dFp64 world_from_screen(ViewTransform2dFp64 const & view, QPointF const & s)
{
    double wx = 0.0, wy = 0.0;
    view.world_from_screen(s.x(), s.y(), wx, wy);
    return Point3dFp64(wx, wy);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
