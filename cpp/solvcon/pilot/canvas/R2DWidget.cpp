/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/canvas/R2DWidget.hpp> // Must be the first include.

#include <array>
#include <cmath>

#include <solvcon/pilot/theme/RThemeManager.hpp>
#include <solvcon/pilot/theme/theme_qt.hpp>

#include <QColor>
#include <QImage>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QPen>
#include <QPolygonF>
#include <QRectF>
#include <QResizeEvent>
#include <QString>
#include <QSvgGenerator>
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

// Screen-pixel slop for picking a shape with the select tool. Converted to a
// world tolerance through the current zoom so thin shapes stay selectable.
constexpr double PICK_TOLERANCE_PX = 5.0;

// Rotate-handle geometry in cosmetic screen pixels: the outward gap from the
// shape's corner, the drawn knob radius, and the hit radius.
constexpr double ROTATE_HANDLE_GAP_PX = 16.0;
constexpr double ROTATE_HANDLE_RADIUS_PX = 5.0;
constexpr double ROTATE_HANDLE_HIT_PX = 9.0;

// Node-edit handle geometry in cosmetic screen pixels: anchor points (curve
// endpoints) draw as squares, control points as circles, both sharing one
// hit radius.
constexpr double NODE_ANCHOR_RADIUS_PX = 4.0;
constexpr double NODE_CONTROL_RADIUS_PX = 3.5;
constexpr double NODE_HANDLE_HIT_PX = 8.0;

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
    m_node_edit = false;
    // A crosshair signals draw mode; the select tool keeps the default arrow.
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

void R2DWidget::setSelectedShape(int32_t shape_id)
{
    // A draw tool paints no selection; accepting one would leave a rotate
    // handle answering for a selection nothing draws.
    if (m_tool->can_draw_shape())
    {
        return;
    }
    if (shape_id >= 0 && (!m_world || !m_world->shape_is_live(shape_id)))
    {
        return;
    }
    // Normalize every negative to the sentinel picking itself writes.
    if (shape_id < 0)
    {
        shape_id = -1;
    }
    if (shape_id == m_selected)
    {
        return;
    }
    // End a gesture before the selection moves out from under its undo bracket.
    endEditDrag();

    m_selected = shape_id;
    m_node_edit = false;
    update();
}

void R2DWidget::enterNodeEdit()
{
    if (m_tool->can_draw_shape() || m_selected < 0 || !m_world ||
        !m_world->shape_is_live(m_selected) ||
        m_world->shape_type_of(m_selected) != ShapeType::BEZIER_PATH)
    {
        return;
    }
    m_node_edit = true;
    update();
}

void R2DWidget::exitNodeEdit()
{
    if (!m_node_edit)
    {
        return;
    }
    // Close a live node drag the way a release would, so it does not linger.
    endEditDrag();
    m_node_edit = false;
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
    m_node_edit = false;
    update();
}

void R2DWidget::followTheme(RThemeManager * manager)
{
    if (manager == nullptr)
    {
        return;
    }

    auto apply = [this, manager](ThemeVariant)
    {
        setCanvasPalette(manager->canvas2dPalette());
    };
    apply(manager->currentVariant());
    connect(manager, &RThemeManager::themeChanged, this, apply);
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
    RWorldRenderer2d(m_world.get(), m_view, m_palette, m_overlay)
        .paint_canvas(painter, width(), height(), full_canvas);

    // Rubber-band preview of the shape currently being dragged, if any.
    paintDrawPreview(painter);

    // Selection box and rotate handle for the select tool, if any.
    paintSelection(painter);

    // Bezier node handles, if node-edit mode is active.
    paintNodeHandles(painter);
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
    RWorldRenderer2d(m_world.get(), m_view, m_palette, overlay)
        .paint_canvas(painter, width(), height(), full_canvas);
    return image;
}

bool R2DWidget::saveSvg(std::string const & filename, Overlay2dOptions const & overlay) const
{
    // SVG is resolution-independent: no devicePixelRatioF scaling here,
    // unlike renderImage's raster output.
    QSvgGenerator generator;
    generator.setFileName(QString::fromStdString(filename));
    generator.setSize(size());
    // Without an explicit viewBox, renderers map the unitless drawing
    // coordinates onto the physical size using the CSS 96-DPI convention,
    // which does not match the DPI QSvgGenerator itself used to compute
    // that physical size; the content then renders shrunk into a corner
    // of the canvas, padded with blank space. An identity viewBox pins
    // pixel coordinates 1:1 to the canvas regardless of DPI.
    generator.setViewBox(QRect(QPoint(0, 0), size()));
    generator.setTitle(QStringLiteral("solvcon 2D canvas"));

    QPainter painter;
    if (!painter.begin(&generator))
    {
        return false;
    }
    constexpr bool full_canvas = true;
    RWorldRenderer2d(m_world.get(), m_view, m_palette, overlay)
        .paint_canvas(painter, width(), height(), full_canvas);
    painter.end();
    return true;
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
    m_tool->paint_preview(painter, m_view, qcolor(m_palette.draw_preview), points);
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
    QColor const selection = qcolor(m_palette.selection);
    QPen pen(selection);
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
    painter.setBrush(selection);
    painter.drawEllipse(handle, ROTATE_HANDLE_RADIUS_PX, ROTATE_HANDLE_RADIUS_PX);
}

void R2DWidget::paintNodeHandles(QPainter & painter) const
{
    if (!m_node_edit || m_selected < 0 || !m_world || !m_world->shape_is_live(m_selected) ||
        m_world->shape_type_of(m_selected) != ShapeType::BEZIER_PATH)
    {
        return;
    }

    QColor const selection = qcolor(m_palette.selection);
    QPen guide_pen(selection);
    guide_pen.setCosmetic(true);
    guide_pen.setWidthF(1.0);
    guide_pen.setStyle(Qt::DotLine);

    size_t const ncurve = m_world->shape_curve_count(m_selected);
    for (size_t i = 0; i < ncurve; ++i)
    {
        auto const curve = m_world->shape_curve(m_selected, static_cast<uint32_t>(i));
        double sx0 = 0.0, sy0 = 0.0, sx1 = 0.0, sy1 = 0.0, sx2 = 0.0, sy2 = 0.0, sx3 = 0.0, sy3 = 0.0;
        m_view.screen_from_world(curve.x0(), curve.y0(), sx0, sy0);
        m_view.screen_from_world(curve.x1(), curve.y1(), sx1, sy1);
        m_view.screen_from_world(curve.x2(), curve.y2(), sx2, sy2);
        m_view.screen_from_world(curve.x3(), curve.y3(), sx3, sy3);
        QPointF const p0(sx0, sy0), p1(sx1, sy1), p2(sx2, sy2), p3(sx3, sy3);

        // Guide lines from each anchor to its adjacent control point.
        painter.setPen(guide_pen);
        painter.drawLine(p0, p1);
        painter.drawLine(p2, p3);

        // Anchors (curve endpoints) as filled squares.
        painter.setPen(Qt::NoPen);
        painter.setBrush(selection);
        painter.drawRect(QRectF(p0.x() - NODE_ANCHOR_RADIUS_PX, p0.y() - NODE_ANCHOR_RADIUS_PX, 2 * NODE_ANCHOR_RADIUS_PX, 2 * NODE_ANCHOR_RADIUS_PX));
        painter.drawRect(QRectF(p3.x() - NODE_ANCHOR_RADIUS_PX, p3.y() - NODE_ANCHOR_RADIUS_PX, 2 * NODE_ANCHOR_RADIUS_PX, 2 * NODE_ANCHOR_RADIUS_PX));

        // Control points as hollow circles.
        QPen control_pen(selection);
        control_pen.setCosmetic(true);
        control_pen.setWidthF(1.5);
        painter.setPen(control_pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawEllipse(p1, NODE_CONTROL_RADIUS_PX, NODE_CONTROL_RADIUS_PX);
        painter.drawEllipse(p2, NODE_CONTROL_RADIUS_PX, NODE_CONTROL_RADIUS_PX);
    }
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

std::optional<std::pair<uint32_t, uint8_t>> R2DWidget::hitNodeHandle(QPointF const & screen_pos) const
{
    if (!m_node_edit || m_selected < 0 || !m_world || !m_world->shape_is_live(m_selected) ||
        m_world->shape_type_of(m_selected) != ShapeType::BEZIER_PATH)
    {
        return std::nullopt;
    }

    std::optional<std::pair<uint32_t, uint8_t>> best;
    double best_dist = 0.0;

    size_t const ncurve = m_world->shape_curve_count(m_selected);
    for (size_t i = 0; i < ncurve; ++i)
    {
        auto const curve = m_world->shape_curve(m_selected, static_cast<uint32_t>(i));
        std::array<std::pair<double, double>, 4> const world_pts{
            {{curve.x0(), curve.y0()}, {curve.x1(), curve.y1()}, {curve.x2(), curve.y2()}, {curve.x3(), curve.y3()}}};
        for (uint8_t p = 0; p < 4; ++p)
        {
            double sx = 0.0, sy = 0.0;
            m_view.screen_from_world(world_pts[p].first, world_pts[p].second, sx, sy);
            double const dist = std::hypot(screen_pos.x() - sx, screen_pos.y() - sy);
            // Strictly closer only, so an exact tie between two curves
            // sharing an endpoint (no tangent linking) keeps the first
            // (lower-index) handle found.
            if (dist <= NODE_HANDLE_HIT_PX && (!best || dist < best_dist))
            {
                best_dist = dist;
                best = std::make_pair(static_cast<uint32_t>(i), p);
            }
        }
    }
    return best;
}

void R2DWidget::finishEdit()
{
    if (m_world && (m_drag == EditDrag::Move || m_drag == EditDrag::Rotate || m_drag == EditDrag::NodePoint))
    {
        m_world->end_operation();
    }
}

void R2DWidget::endEditDrag()
{
    if (m_drag == EditDrag::None)
    {
        return;
    }
    finishEdit();
    m_drag = EditDrag::None;
    unsetCursor();
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
        // Select tool: drag a node handle, rotate the selection, move a
        // picked shape, or fall back to panning the view on empty space.
        if (auto const node_hit = hitNodeHandle(pos))
        {
            m_node_curve_idx = node_hit->first;
            m_node_point_idx = node_hit->second;
            m_view.world_from_screen(pos.x(), pos.y(), m_node_last_x, m_node_last_y);
            // Bracket the node drag so its incremental steps undo as one.
            m_world->begin_operation();
            m_drag = EditDrag::NodePoint;
            event->accept();
            return;
        }
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
            if (hit != m_selected)
            {
                m_node_edit = false;
            }
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
        m_node_edit = false;
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
    if (m_drag == EditDrag::NodePoint)
    {
        double wx = 0.0, wy = 0.0;
        m_view.world_from_screen(pos.x(), pos.y(), wx, wy);
        if (m_world && m_world->shape_is_live(m_selected))
        {
            m_world->move_shape_curve_point(m_selected, m_node_curve_idx, m_node_point_idx, wx - m_node_last_x, wy - m_node_last_y);
        }
        m_node_last_x = wx;
        m_node_last_y = wy;
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
        endEditDrag();
        update();
        event->accept();
        return;
    }
    QWidget::mouseReleaseEvent(event);
}

void R2DWidget::mouseDoubleClickEvent(QMouseEvent * event)
{
    if (event->button() == Qt::LeftButton && !m_tool->can_draw_shape() &&
        m_selected >= 0 && m_world && m_world->shape_is_live(m_selected) &&
        m_world->shape_type_of(m_selected) == ShapeType::BEZIER_PATH)
    {
        // The press half of this double-click already ran the ordinary
        // select-tool press logic and opened a Move drag bracket (Qt's
        // sequence is Press -> Release -> Press -> DoubleClick -> Release);
        // close it before switching modes so it does not linger as a
        // dangling, ultimately no-op, undo step.
        endEditDrag();
        enterNodeEdit();
        event->accept();
        return;
    }
    QWidget::mouseDoubleClickEvent(event);
}

void R2DWidget::keyPressEvent(QKeyEvent * event)
{
    if (event->key() == Qt::Key_Escape && m_node_edit)
    {
        exitNodeEdit();
        event->accept();
        return;
    }
    QWidget::keyPressEvent(event);
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
