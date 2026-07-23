#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Renders a world's geometry into a QPainter through a 2D view transform.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/universe/ViewTransform2d.hpp>
#include <solvcon/universe/World.hpp>

#include <cstdint>
#include <vector>

#include <QPointF>
#include <QRectF>

class QPainter;

namespace solvcon
{

/**
 * Toggleable, legibility-only annotations over the 2D canvas: shape ids,
 * bounding boxes, grid coordinate labels, advanced geometric labels, and one
 * highlighted shape id. Draws only stored geometry and the view's own grid,
 * never derived diagnostics.
 *
 * @ingroup group_domain
 */
struct Overlay2dOptions
{
    bool shape_ids = false; ///< Label each live shape with its id and type.
    bool bounding_boxes = false; ///< Draw each live shape's axis-aligned box.
    bool coordinate_labels = false; ///< Label grid lines with world coordinates.
    bool advanced_labels = false; ///< More details than just the id.
    int32_t highlight_id = -1; ///< Emphasize this shape id; -1 draws none. Independent of selection.

    bool shape_annotations() const
    {
        return shape_ids || bounding_boxes || advanced_labels || highlight_id >= 0;
    }

    bool any() const { return coordinate_labels || shape_annotations(); }
}; /* end struct Overlay2dOptions */

/**
 * Renders a world's live points, segments, and curves into a QPainter in
 * screen space, mapping math-convention world coordinates through a 2D view
 * transform. paint() draws geometry only; paint_canvas() adds the backdrop
 * and optional grid/axes/origin chrome, then the optional annotation overlay.
 * m_world is non-owning and may be null.
 *
 * @ingroup group_domain
 */
class RWorldRenderer2d
{
public:
    RWorldRenderer2d(WorldFp64 const * world, ViewTransform2dFp64 const & view, Overlay2dOptions const & overlay = {})
        : m_world(world)
        , m_view(view)
        , m_overlay(overlay)
    {
    }

    /// Paint backdrop, geometry, optional chrome, and the annotation overlay.
    /// @param painter Target painter in screen space.
    /// @param width Canvas width in pixels.
    /// @param height Canvas height in pixels.
    /// @param full_canvas If true, draw grid, axes, and the origin marker.
    void paint_canvas(QPainter & painter, int width, int height, bool full_canvas) const;

private:
    void paint(QPainter & painter) const;

    void paint_overlay(QPainter & painter, int width, int height) const;
    void paint_shape_annotations(QPainter & painter, int width, int height, std::vector<QRectF> const & reserved) const;

    // Map math-convention world (x, y) to Qt screen pixels; z is dropped.
    QPointF map(double world_x, double world_y) const;

    WorldFp64 const * m_world;
    ViewTransform2dFp64 const & m_view;
    Overlay2dOptions m_overlay;
}; /* end class RWorldRenderer2d */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
