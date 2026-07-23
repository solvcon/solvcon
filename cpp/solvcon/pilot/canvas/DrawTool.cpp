/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/canvas/DrawTool.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <QColor>
#include <QPainter>
#include <QPen>
#include <QPointF>
#include <QPolygonF>
#include <QRectF>

namespace solvcon
{

void DrawToolBase::paint_preview(
    QPainter & painter,
    ViewTransform2dFp64 const & view,
    std::span<DrawPoint const> points) const
{
    QPen pen(QColor(240, 200, 120));
    pen.setCosmetic(true);
    pen.setWidthF(1.5);
    pen.setStyle(Qt::DashLine);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    paint_outline(painter, view, points);
}

namespace
{

/// Map a world-space gesture point to widget screen coordinates.
QPointF to_screen_qpoint(ViewTransform2dFp64 const & view, DrawPoint const & p)
{
    double sx = 0.0;
    double sy = 0.0;
    view.screen_from_world(p.x, p.y, sx, sy);
    return QPointF(sx, sy);
}

/// The default navigation tool: a left-button drag pans and zooms the view instead of drawing.
class PanTool : public DrawToolBase
{

public:

    static constexpr char const * NAME = "pan";

    std::string name() const override { return NAME; }

    bool can_draw_shape() const override { return false; }

    void commit(WorldFp64 &, std::span<DrawPoint const>) const override {}

protected:

    void paint_outline(QPainter &, ViewTransform2dFp64 const &, std::span<DrawPoint const>) const override {}

}; /* end class PanTool */

/// Circle defined by two gesture points: the center and a point on the rim.
class CircleTool : public DrawToolBase
{

public:

    static constexpr char const * NAME = "circle";

    std::string name() const override { return NAME; }

    bool can_draw_shape() const override { return true; }

    void commit(WorldFp64 & world, std::span<DrawPoint const> points) const override
    {
        if (points.size() < 2)
        {
            throw std::invalid_argument("CircleTool::commit: need at least two points");
        }
        DrawPoint const & center = points.front();
        DrawPoint const & rim = points.back();
        world.add_circle(center.x, center.y, std::hypot(rim.x - center.x, rim.y - center.y));
    }

protected:

    void paint_outline(QPainter & painter,
                       ViewTransform2dFp64 const & view,
                       std::span<DrawPoint const> points) const override
    {
        if (points.size() < 2)
        {
            throw std::invalid_argument("CircleTool::paint_outline: need at least two points");
        }
        DrawPoint const & center = points.front();
        DrawPoint const & rim = points.back();
        double const radius = std::hypot(rim.x - center.x, rim.y - center.y);
        if (radius <= 0.0)
        {
            throw std::invalid_argument("CircleTool::paint_outline: radius must be positive");
        }
        double const radius_px = view.zoom() * radius;
        painter.drawEllipse(to_screen_qpoint(view, center), radius_px, radius_px);
    }

}; /* end class CircleTool */

/// Straight line segment from the first gesture point to the second.
class LineTool : public DrawToolBase
{

public:

    static constexpr char const * NAME = "line";

    std::string name() const override { return NAME; }

    bool can_draw_shape() const override { return true; }

    void commit(WorldFp64 & world, std::span<DrawPoint const> points) const override
    {
        if (points.size() < 2)
        {
            throw std::invalid_argument("LineTool::commit: need at least two points");
        }

        DrawPoint const & p0 = points.front();
        DrawPoint const & p1 = points.back();
        world.add_line(p0.x, p0.y, p1.x, p1.y);
    }

protected:

    void paint_outline(QPainter & painter,
                       ViewTransform2dFp64 const & view,
                       std::span<DrawPoint const> points) const override
    {
        if (points.size() < 2)
        {
            throw std::invalid_argument("LineTool::paint_outline: need at least two points");
        }

        painter.drawLine(to_screen_qpoint(view, points.front()), to_screen_qpoint(view, points.back()));
    }

}; /* end class LineTool */

/// Isosceles triangle inscribed in the rectangle closed by the first gesture
/// point and the cursor (opposite corners).
/// The base is the horizontal segment between the two corners, and the apex is
/// the midpoint of the base shifted vertically to the cursor's y coordinate.
class TriangleTool : public DrawToolBase
{

public:

    static constexpr char const * NAME = "triangle";

    std::string name() const override { return NAME; }

    bool can_draw_shape() const override { return true; }

    void commit(WorldFp64 & world, std::span<DrawPoint const> points) const override
    {
        if (points.size() < 2)
        {
            throw std::invalid_argument("TriangleTool::commit: need at least two points");
        }

        DrawPoint const & anchor = points.front();
        DrawPoint const & corner = points.back();
        double const apex_x = (anchor.x + corner.x) * 0.5;
        world.add_triangle(anchor.x, anchor.y, corner.x, anchor.y, apex_x, corner.y);
    }

protected:

    void paint_outline(QPainter & painter,
                       ViewTransform2dFp64 const & view,
                       std::span<DrawPoint const> points) const override
    {
        if (points.size() < 2)
        {
            throw std::invalid_argument("TriangleTool::paint_outline: need at least two points");
        }

        DrawPoint const & anchor = points.front();
        DrawPoint const & corner = points.back();
        double const apex_x = (anchor.x + corner.x) * 0.5;
        QPolygonF const corners{to_screen_qpoint(view, anchor),
                                to_screen_qpoint(view, {corner.x, anchor.y}),
                                to_screen_qpoint(view, {apex_x, corner.y})};
        painter.drawPolygon(corners);
    }

}; /* end class TriangleTool */

/// Axis-aligned rectangle spanning the two gesture points as opposite corners.
class RectangleTool : public DrawToolBase
{

public:

    static constexpr char const * NAME = "rectangle";

    std::string name() const override { return NAME; }

    bool can_draw_shape() const override { return true; }

    void commit(WorldFp64 & world, std::span<DrawPoint const> points) const override
    {
        if (points.size() < 2)
        {
            throw std::invalid_argument("RectangleTool::commit: need at least two points");
        }

        DrawPoint const & corner0 = points.front();
        DrawPoint const & corner1 = points.back();
        double const min_x = std::min(corner0.x, corner1.x);
        double const max_x = std::max(corner0.x, corner1.x);
        double const min_y = std::min(corner0.y, corner1.y);
        double const max_y = std::max(corner0.y, corner1.y);
        world.add_rectangle(min_x, min_y, max_x, max_y);
    }

protected:

    void paint_outline(QPainter & painter,
                       ViewTransform2dFp64 const & view,
                       std::span<DrawPoint const> points) const override
    {
        if (points.size() < 2)
        {
            throw std::invalid_argument("RectangleTool::paint_outline: need at least two points");
        }

        painter.drawRect(QRectF(to_screen_qpoint(view, points.front()), to_screen_qpoint(view, points.back())).normalized());
    }

}; /* end class RectangleTool */

/// Axis-aligned ellipse anchored like the circle tool: the first gesture point
/// is the center, the second fixes a bounding-box corner, so the two
/// half-axes are the absolute x and y offsets between them.
class EllipseTool : public DrawToolBase
{

public:

    static constexpr char const * NAME = "ellipse";

    std::string name() const override { return NAME; }

    bool can_draw_shape() const override { return true; }

    void commit(WorldFp64 & world, std::span<DrawPoint const> points) const override
    {
        if (points.size() < 2)
        {
            throw std::invalid_argument("EllipseTool::commit: need at least two points");
        }

        DrawPoint const & center = points.front();
        DrawPoint const & corner = points.back();
        world.add_ellipse(center.x, center.y, std::fabs(corner.x - center.x), std::fabs(corner.y - center.y));
    }

protected:

    void paint_outline(QPainter & painter,
                       ViewTransform2dFp64 const & view,
                       std::span<DrawPoint const> points) const override
    {
        if (points.size() < 2)
        {
            throw std::invalid_argument("EllipseTool::paint_outline: need at least two points");
        }

        DrawPoint const & center = points.front();
        DrawPoint const & corner = points.back();
        double const rx_px = view.zoom() * std::fabs(corner.x - center.x);
        double const ry_px = view.zoom() * std::fabs(corner.y - center.y);
        painter.drawEllipse(to_screen_qpoint(view, center), rx_px, ry_px);
    }

}; /* end class EllipseTool */

/// The tool registry. The first entry is the default tool a fresh canvas
/// starts with. Add a new shape by writing a `DrawToolBase` subclass above
/// and appending one entry here.
struct ToolEntry
{
    char const * name;
    std::unique_ptr<DrawToolBase> (*make)();
}; /* end struct ToolEntry */

ToolEntry const TOOL_TABLE[] = {
    {PanTool::NAME, []() -> std::unique_ptr<DrawToolBase>
     { return std::make_unique<PanTool>(); }},
    {LineTool::NAME, []() -> std::unique_ptr<DrawToolBase>
     { return std::make_unique<LineTool>(); }},
    {TriangleTool::NAME, []() -> std::unique_ptr<DrawToolBase>
     { return std::make_unique<TriangleTool>(); }},
    {RectangleTool::NAME, []() -> std::unique_ptr<DrawToolBase>
     { return std::make_unique<RectangleTool>(); }},
    {EllipseTool::NAME, []() -> std::unique_ptr<DrawToolBase>
     { return std::make_unique<EllipseTool>(); }},
    {CircleTool::NAME, []() -> std::unique_ptr<DrawToolBase>
     { return std::make_unique<CircleTool>(); }},
};

} /* end namespace */

std::vector<std::string> const & draw_tool_names()
{
    static std::vector<std::string> const names = []
    {
        std::vector<std::string> out;
        for (ToolEntry const & entry : TOOL_TABLE)
        {
            out.emplace_back(entry.name);
        }
        return out;
    }();
    return names;
}

std::string const & default_draw_tool_name()
{
    static_assert(std::size(TOOL_TABLE) > 0, "TOOL_TABLE must have at least one entry");
    return draw_tool_names().front();
}

std::unique_ptr<DrawToolBase> make_draw_tool(std::string const & name)
{
    for (ToolEntry const & entry : TOOL_TABLE)
    {
        if (name == entry.name)
        {
            return entry.make();
        }
    }
    throw std::invalid_argument("make_draw_tool: unknown draw tool '" + name + "'");
}

bool is_draw_tool(std::string const & name)
{
    for (ToolEntry const & entry : TOOL_TABLE)
    {
        if (name == entry.name)
        {
            return true;
        }
    }
    return false;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
