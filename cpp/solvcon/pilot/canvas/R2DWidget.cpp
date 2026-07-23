/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/canvas/R2DWidget.hpp> // Must be the first include.

#include <array>
#include <cmath>

#include <QColor>
#include <QImage>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QPen>
#include <QPolygonF>
#include <QResizeEvent>
#include <QWheelEvent>

namespace solvcon
{

namespace
{

// One wheel revolution (360 degrees) doubles the zoom.
constexpr double ZOOM_STEP_PER_DEGREE = 1.0 / 360.0;

constexpr double MIN_ZOOM = 1.0e-6;
constexpr double MAX_ZOOM = 1.0e6;

// Drags spanning fewer than this many screen pixels commit nothing, so a
// stray click in a shape tool does not drop a degenerate shape into the
// world.
constexpr double MIN_DRAW_DRAG_PX = 2.0;

// Screen-pixel slop for picking a shape with the pan tool. Converted to a
// world tolerance through the current zoom so thin shapes stay selectable.
constexpr double PICK_TOLERANCE_PX = 5.0;

// Rotate-handle geometry in cosmetic screen pixels: the outward gap from the
// shape's corner, the drawn knob radius, and the hit radius.
constexpr double ROTATE_HANDLE_GAP_PX = 16.0;
constexpr double ROTATE_HANDLE_RADIUS_PX = 5.0;
constexpr double ROTATE_HANDLE_HIT_PX = 9.0;

QColor const SELECTION(120, 200, 255);

double clamp_zoom(double zoom)
{
    if (!std::isfinite(zoom))
    {
        return 1.0;
    }
    if (zoom < MIN_ZOOM)
    {
        return MIN_ZOOM;
    }
    if (zoom > MAX_ZOOM)
    {
        return MAX_ZOOM;
    }
    return zoom;
}

bool is_finite_view(ViewTransform2dFp64 const & v)
{
    return std::isfinite(v.pan_x()) && std::isfinite(v.pan_y()) && std::isfinite(v.zoom());
}

} /* end namespace */

R2DWidget::R2DWidget(QWidget * parent, Qt::WindowFlags f)
    : QWidget(parent, f)
    , m_tool(make_draw_tool(default_draw_tool_name()))
{
    setFocusPolicy(Qt::StrongFocus);
    setAttribute(Qt::WA_OpaquePaintEvent, true);
    // Defer centering to the first resizeEvent, when geometry is real.
}

void R2DWidget::setViewTransform(ViewTransform2dFp64 const & v)
{
    if (!is_finite_view(v))
    {
        return;
    }
    m_view = v;
    m_view.set_zoom(clamp_zoom(m_view.zoom()));
    // An explicit view disables the deferred auto-centering.
    m_view_modified = true;
    update();
}

void R2DWidget::resetView()
{
    m_view.reset();
    centerViewOnOrigin();
    // Re-enable auto-centering on later resizes.
    m_view_modified = false;
    update();
}

void R2DWidget::setDrawTool(std::string const & name)
{
    if (name == drawTool())
    {
        return;
    }
    // make_draw_tool throws for an unknown name; let it propagate before any
    // state changes so an invalid request leaves the current tool untouched.
    std::unique_ptr<DrawToolBase> tool = make_draw_tool(name);
    m_tool = std::move(tool);

    finishEdit();
    m_drawing = false;
    m_selected = -1;
    m_drag = EditDrag::None;
    // A crosshair signals draw mode; the pan tool keeps the default arrow.
    if (m_tool->can_draw_shape())
    {
        setCursor(Qt::CrossCursor);
    }
    else
    {
        unsetCursor();
    }
    update();
}

void R2DWidget::updateWorld(std::shared_ptr<WorldFp64> const & world)
{
    // Close any edit gesture on the outgoing world before swapping it out.
    finishEdit();
    m_world = world;
    // A new world invalidates any shape id we held selected or highlighted;
    // the display toggles persist so the overlay mode carries across worlds.
    m_selected = -1;
    m_overlay.highlight_id = -1;
    m_drag = EditDrag::None;
    update();
}

void R2DWidget::centerViewOnOrigin()
{
    m_view.set_pan_x(static_cast<double>(width()) * 0.5);
    m_view.set_pan_y(static_cast<double>(height()) * 0.5);
}

void R2DWidget::paintEvent(QPaintEvent * /*event*/)
{
    QPainter painter(this);
    constexpr bool full_canvas = true;
    RWorldRenderer2d(m_world.get(), m_view, m_overlay).paint_canvas(painter, width(), height(), full_canvas);

    // Rubber-band preview of the shape currently being dragged, if any.
    paintDrawPreview(painter);

    // Selection box and rotate handle for the pan tool, if any.
    paintSelection(painter);
}

QImage R2DWidget::renderImage(Overlay2dOptions const & overlay) const
{
    // Match grab()'s device-pixel-ratio-aware output: back the image with
    // device pixels and paint in logical coordinates, so a high-DPI export
    // keeps the same resolution as the on-screen frame.
    double const dpr = devicePixelRatioF();
    QImage image(size() * dpr, QImage::Format_ARGB32_Premultiplied);
    image.setDevicePixelRatio(dpr);
    QPainter painter(&image);
    constexpr bool full_canvas = true;
    RWorldRenderer2d(m_world.get(), m_view, overlay)
        .paint_canvas(painter, width(), height(), full_canvas);
    return image;
}

void R2DWidget::paintDrawPreview(QPainter & painter) const
{
    if (!m_drawing)
    {
        return;
    }
    // avoid painting when the canvas is just re-entered
    if (m_draw_current_x == m_draw_start_x && m_draw_current_y == m_draw_start_y)
    {
        return;
    }
    std::array<DrawPoint, 2> const points{{{m_draw_start_x, m_draw_start_y}, {m_draw_current_x, m_draw_current_y}}};
    m_tool->paint_preview(painter, m_view, points);
}

void R2DWidget::paintSelection(QPainter & painter) const
{
    if (m_tool->can_draw_shape() || m_selected < 0 || !m_world || !m_world->shape_is_live(m_selected))
    {
        return;
    }
    // The oriented bounding box wraps the shape at any orientation, so the
    // box and its top-left handle rotate together and never separate.
    obb_array_type const obb = m_world->shape_obb(m_selected);
    QPolygonF box;

    double sx = 0.0, sy = 0.0;
    for (size_t i = 0; i < 4; ++i)
    {
        m_view.screen_from_world(obb[2 * i], obb[2 * i + 1], sx, sy);
        box << QPointF(sx, sy);
    }

    // Draw the box and the rotate handle knob.
    QPen pen(SELECTION);
    pen.setCosmetic(true);
    pen.setWidthF(1.5);
    pen.setStyle(Qt::DashLine);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    painter.drawPolygon(box);

    // Short stem from the box's top-left corner out to the rotate knob.
    QPointF const handle = rotateHandlePos();
    pen.setStyle(Qt::SolidLine);
    painter.setPen(pen);
    painter.drawLine(box.front(), handle);
    painter.setBrush(SELECTION);
    painter.drawEllipse(handle, ROTATE_HANDLE_RADIUS_PX, ROTATE_HANDLE_RADIUS_PX);
}

int32_t R2DWidget::pickShapeAt(QPointF const & screen_pos) const
{
    if (!m_world)
    {
        return -1;
    }
    double wx = 0.0, wy = 0.0;
    m_view.world_from_screen(screen_pos.x(), screen_pos.y(), wx, wy);
    double const tol = PICK_TOLERANCE_PX / m_view.zoom();
    return m_world->pick_shape(wx, wy, tol);
}

R2DWidget::coord2_type R2DWidget::selectionCenterWorld() const
{
    // Center of the oriented bounding box: midpoint of opposite corners.
    obb_array_type const obb = m_world->shape_obb(m_selected);
    return {(obb[0] + obb[4]) * 0.5, (obb[1] + obb[5]) * 0.5};
}

QPointF R2DWidget::rotateHandlePos() const
{
    // The handle anchor is always at the box's top-left corner (obb[0..1]).
    obb_array_type const obb = m_world->shape_obb(m_selected);
    double hx = 0.0, hy = 0.0, cx = 0.0, cy = 0.0;
    m_view.screen_from_world(obb[0], obb[1], hx, hy);
    m_view.screen_from_world((obb[0] + obb[4]) * 0.5, (obb[1] + obb[5]) * 0.5, cx, cy);
    double const dx = hx - cx, dy = hy - cy;
    double const len = std::hypot(dx, dy);
    if (len > 1.0e-9)
    {
        hx += dx / len * ROTATE_HANDLE_GAP_PX;
        hy += dy / len * ROTATE_HANDLE_GAP_PX;
    }
    return QPointF(hx, hy);
}

R2DWidget::coord2_type R2DWidget::rotateHandleScreen() const
{
    if (m_selected < 0 || !m_world || !m_world->shape_is_live(m_selected))
    {
        return {-1.0, -1.0};
    }
    QPointF const h = rotateHandlePos();
    return {h.x(), h.y()};
}

bool R2DWidget::isOnRotateHandle(QPointF const & screen_pos) const
{
    if (m_selected < 0 || !m_world || !m_world->shape_is_live(m_selected))
    {
        return false;
    }
    QPointF const d = screen_pos - rotateHandlePos();
    return std::hypot(d.x(), d.y()) <= ROTATE_HANDLE_HIT_PX;
}

void R2DWidget::finishEdit()
{
    if (m_world && (m_drag == EditDrag::Move || m_drag == EditDrag::Rotate))
    {
        m_world->end_operation();
    }
}

void R2DWidget::wheelEvent(QWheelEvent * event)
{
    QPointF const pos = event->position();
    double const degrees = static_cast<double>(event->angleDelta().y()) / 8.0;
    double const factor = std::exp(degrees * ZOOM_STEP_PER_DEGREE * std::log(2.0));

    if (!std::isfinite(factor) || !(factor > 0.0))
    {
        event->ignore();
        return;
    }
    m_view.zoom_at_clamped(factor, pos.x(), pos.y(), MIN_ZOOM, MAX_ZOOM);
    m_view_modified = true;
    update();
    event->accept();
}

void R2DWidget::mousePressEvent(QMouseEvent * event)
{
    if (event->button() == Qt::LeftButton)
    {
        QPointF const pos = event->position();
        if (m_tool->can_draw_shape())
        {
            // Anchor the drag in world space so it is robust to any pan or
            // zoom that happens mid-stroke.
            m_view.world_from_screen(pos.x(), pos.y(), m_draw_start_x, m_draw_start_y);
            m_draw_current_x = m_draw_start_x;
            m_draw_current_y = m_draw_start_y;
            m_drawing = true;
            event->accept();
            return;
        }
        // Pan tool: rotate the selection, move a picked shape, or fall
        // back to panning the view on empty space.
        if (isOnRotateHandle(pos))
        {
            coord2_type const c = selectionCenterWorld();
            m_rotate_cx = c[0];
            m_rotate_cy = c[1];
            double wx = 0.0, wy = 0.0;
            m_view.world_from_screen(pos.x(), pos.y(), wx, wy);
            m_rotate_last_angle = std::atan2(wy - m_rotate_cy, wx - m_rotate_cx);
            // Bracket the rotate gesture so its incremental steps undo as one.
            m_world->begin_operation();
            m_drag = EditDrag::Rotate;
            event->accept();
            return;
        }
        int32_t const hit = pickShapeAt(pos);
        if (hit >= 0)
        {
            m_selected = hit;
            m_view.world_from_screen(pos.x(), pos.y(), m_move_last_x, m_move_last_y);
            // Bracket the move gesture so its incremental steps undo as one.
            m_world->begin_operation();
            m_drag = EditDrag::Move;
            setCursor(Qt::SizeAllCursor);
            update();
            event->accept();
            return;
        }
        // Empty space: drop the selection and pan the view.
        m_selected = -1;
        m_drag = EditDrag::View;
        m_last_mouse_pos = pos;
        setCursor(Qt::ClosedHandCursor);
        update();
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void R2DWidget::mouseMoveEvent(QMouseEvent * event)
{
    QPointF const pos = event->position();
    if (m_drawing)
    {
        m_view.world_from_screen(pos.x(), pos.y(), m_draw_current_x, m_draw_current_y);
        update();
        event->accept();
        return;
    }
    if (m_drag == EditDrag::Move)
    {
        double wx = 0.0, wy = 0.0;
        m_view.world_from_screen(pos.x(), pos.y(), wx, wy);
        if (m_world && m_world->shape_is_live(m_selected))
        {
            m_world->translate_shape(m_selected, wx - m_move_last_x, wy - m_move_last_y);
        }
        m_move_last_x = wx;
        m_move_last_y = wy;
        update();
        event->accept();
        return;
    }
    if (m_drag == EditDrag::Rotate)
    {
        double wx = 0.0, wy = 0.0;
        m_view.world_from_screen(pos.x(), pos.y(), wx, wy);
        double const angle = std::atan2(wy - m_rotate_cy, wx - m_rotate_cx);
        if (m_world && m_world->shape_is_live(m_selected))
        {
            m_world->rotate_shape(m_selected, angle - m_rotate_last_angle, m_rotate_cx, m_rotate_cy);
        }
        m_rotate_last_angle = angle;
        update();
        event->accept();
        return;
    }
    if (m_drag == EditDrag::View)
    {
        QPointF const delta = pos - m_last_mouse_pos;
        m_last_mouse_pos = pos;
        m_view.pan(delta.x(), delta.y());
        m_view_modified = true;
        update();
        event->accept();
        return;
    }
    QWidget::mouseMoveEvent(event);
}

void R2DWidget::mouseReleaseEvent(QMouseEvent * event)
{
    if (event->button() == Qt::LeftButton && m_drawing)
    {
        m_drawing = false;

        double const diff = std::hypot(m_draw_current_x - m_draw_start_x, m_draw_current_y - m_draw_start_y);
        double const drag_px = m_view.zoom() * diff;

        if (m_world && drag_px >= MIN_DRAW_DRAG_PX)
        {
            std::array<DrawPoint, 2> const points{
                {{m_draw_start_x, m_draw_start_y},
                 {m_draw_current_x, m_draw_current_y}}};
            m_tool->commit(*m_world, points);
        }

        update();
        event->accept();
        return;
    }
    if (event->button() == Qt::LeftButton && m_drag != EditDrag::None)
    {
        // A move or rotate gesture ends here; commit it as one undo step.
        finishEdit();
        m_drag = EditDrag::None;
        unsetCursor();
        update();
        event->accept();
        return;
    }
    QWidget::mouseReleaseEvent(event);
}

void R2DWidget::resizeEvent(QResizeEvent * event)
{
    QWidget::resizeEvent(event);
    // Auto-center the origin until the view is set explicitly.
    if (!m_view_modified && width() > 0 && height() > 0)
    {
        centerViewOnOrigin();
        update();
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
