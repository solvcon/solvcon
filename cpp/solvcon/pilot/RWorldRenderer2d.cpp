/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RWorldRenderer2d.hpp> // Must be the first include.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <QColor>
#include <QFontMetricsF>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPointF>
#include <QRectF>
#include <QString>
#include <QStringList>

namespace solvcon
{

namespace
{

QColor const BACKGROUND(32, 32, 36);
QColor const MINOR_GRID(64, 64, 70);
QColor const AXIS(200, 200, 80);
QColor const ORIGIN(220, 80, 80);
QColor const GEOMETRY(120, 180, 240);
QColor const OVERLAY_TEXT(225, 225, 230);
QColor const OVERLAY_BBOX(150, 200, 150);
QColor const OVERLAY_HIGHLIGHT(240, 180, 60);

constexpr double BASE_GRID_SPACING_PX = 64.0;
constexpr double MIN_GRID_SPACING_PX = 16.0;
constexpr double MAX_GRID_SPACING_PX = 256.0;

// Cosmetic (zoom-independent) screen widths for world geometry.
constexpr double GEOMETRY_LINE_WIDTH_PX = 1.5;
constexpr int GEOMETRY_POINT_WIDTH_PX = 5;

void paint_chrome(QPainter & painter, ViewTransform2dFp64 const & view, int width, int height)
{
    double const widget_w = static_cast<double>(width);
    double const widget_h = static_cast<double>(height);

    // Snap grid spacing to a power of ten times {1, 2, 5} for a readable band.
    double const target_world = BASE_GRID_SPACING_PX / view.zoom();
    double const exponent = std::floor(std::log10(target_world));
    double const base = std::pow(10.0, exponent);
    double spacing_world = base;
    for (double mult : {1.0, 2.0, 5.0, 10.0})
    {
        double const candidate = base * mult;
        double const candidate_px = view.zoom() * candidate;
        if (candidate_px >= MIN_GRID_SPACING_PX && candidate_px <= MAX_GRID_SPACING_PX)
        {
            spacing_world = candidate;
            break;
        }
        spacing_world = candidate;
    }
    double const spacing_px = view.zoom() * spacing_world;

    // Draw minor grid lines in screen space directly.
    QPen minor_pen(MINOR_GRID);
    minor_pen.setCosmetic(true);
    minor_pen.setWidth(1);
    painter.setPen(minor_pen);

    double const first_x = std::fmod(view.pan_x(), spacing_px);
    for (double sx = first_x; sx < widget_w; sx += spacing_px)
    {
        painter.drawLine(QPointF(sx, 0.0), QPointF(sx, widget_h));
    }
    for (double sx = first_x - spacing_px; sx > 0.0; sx -= spacing_px)
    {
        painter.drawLine(QPointF(sx, 0.0), QPointF(sx, widget_h));
    }
    double const first_y = std::fmod(view.pan_y(), spacing_px);
    for (double sy = first_y; sy < widget_h; sy += spacing_px)
    {
        painter.drawLine(QPointF(0.0, sy), QPointF(widget_w, sy));
    }
    for (double sy = first_y - spacing_px; sy > 0.0; sy -= spacing_px)
    {
        painter.drawLine(QPointF(0.0, sy), QPointF(widget_w, sy));
    }

    // Draw the world axes through the origin, if visible.
    QPen axis_pen(AXIS);
    axis_pen.setCosmetic(true);
    axis_pen.setWidth(1);
    painter.setPen(axis_pen);
    if (view.pan_y() >= 0.0 && view.pan_y() <= widget_h)
    {
        painter.drawLine(QPointF(0.0, view.pan_y()), QPointF(widget_w, view.pan_y()));
    }
    if (view.pan_x() >= 0.0 && view.pan_x() <= widget_w)
    {
        painter.drawLine(QPointF(view.pan_x(), 0.0), QPointF(view.pan_x(), widget_h));
    }
}

/**
 * Uniform-grid index of screen rectangles a label must not overlap: placed
 * labels, each shape's padded bbox, and strips over the visible world axes.
 * add() files a rectangle under the CELL-pixel cells it spans; hits() tests a
 * trial only against rectangles in the cells it spans. Cells are clamped to
 * the canvas, bounding the cost of a far-zoomed shape.
 */
struct LabelObstacles
{
    static constexpr double CELL = 128.0;

    explicit LabelObstacles(double canvas_w, double canvas_h)
        : max_cx(cell_of(canvas_w))
        , max_cy(cell_of(canvas_h))
    {
    }

    std::vector<QRectF> rects;
    std::unordered_map<int64_t, std::vector<size_t>> cells;
    int32_t max_cx;
    int32_t max_cy;

    static int64_t key(int32_t cx, int32_t cy)
    {
        return (static_cast<int64_t>(cy) << 32) ^ static_cast<uint32_t>(cx);
    }

    static int32_t cell_of(double v)
    {
        return static_cast<int32_t>(std::floor(v / CELL));
    }

    int32_t clamp_cx(int32_t cx) const { return std::clamp(cx, 0, max_cx); }
    int32_t clamp_cy(int32_t cy) const { return std::clamp(cy, 0, max_cy); }

    void add(QRectF const & r)
    {
        if (!(r.isValid() && r.width() > 0.0 && r.height() > 0.0))
        {
            return;
        }

        size_t const idx = rects.size();
        rects.push_back(r);
        int32_t const cy0 = clamp_cy(cell_of(r.top()));
        int32_t const cy1 = clamp_cy(cell_of(r.bottom()));
        int32_t const cx0 = clamp_cx(cell_of(r.left()));
        int32_t const cx1 = clamp_cx(cell_of(r.right()));
        for (int32_t cy = cy0; cy <= cy1; ++cy)
        {
            for (int32_t cx = cx0; cx <= cx1; ++cx)
            {
                cells[key(cx, cy)].push_back(idx);
            }
        }
    }

    bool hits(QRectF const & trial) const
    {
        int32_t const cy0 = clamp_cy(cell_of(trial.top()));
        int32_t const cy1 = clamp_cy(cell_of(trial.bottom()));
        int32_t const cx0 = clamp_cx(cell_of(trial.left()));
        int32_t const cx1 = clamp_cx(cell_of(trial.right()));
        for (int32_t cy = cy0; cy <= cy1; ++cy)
        {
            for (int32_t cx = cx0; cx <= cx1; ++cx)
            {
                auto const it = cells.find(key(cx, cy));
                if (it == cells.end())
                {
                    continue;
                }
                for (size_t const idx : it->second)
                {
                    if (trial.intersects(rects[idx]))
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }
};

/**
 * Top-left for a block_w by block_h label placed outside `box`, on-canvas and
 * clear of `obstacles`. Tries ten fixed seats flanking the box, then rings at
 * growing radius, and keeps the clear seat nearest the box center. Falls back
 * to the nearest blocked on-canvas seat, or the top-left corner if none fits.
 */
QPointF place_label_block(
    QRectF const & box,
    double block_w,
    double block_h,
    double canvas_w,
    double canvas_h,
    LabelObstacles const & obstacles)
{
    constexpr double MARGIN = 10.0;
    constexpr double PI = 3.14159265358979323846;
    constexpr int MAX_RINGS = 8;

    auto on_canvas = [&](QPointF const & tl)
    {
        return tl.x() >= 2.0 && tl.y() >= 2.0 && tl.x() + block_w <= canvas_w - 2.0 && tl.y() + block_h <= canvas_h - 2.0;
    };

    bool found = false;
    QPointF best(2.0, 2.0);
    double best_dist2 = 1.0e300;
    QPointF fallback(2.0, 2.0);
    double fallback_dist2 = 1.0e300;
    bool have_fallback = false;
    QPointF const anchor = box.center();

    auto consider = [&](QPointF const & tl)
    {
        if (!on_canvas(tl))
        {
            return;
        }
        QRectF const trial(tl.x(), tl.y(), block_w, block_h);
        double const dx = trial.center().x() - anchor.x();
        double const dy = trial.center().y() - anchor.y();
        double const dist2 = dx * dx + dy * dy;
        if (obstacles.hits(trial))
        {
            if (!have_fallback || dist2 < fallback_dist2)
            {
                have_fallback = true;
                fallback_dist2 = dist2;
                fallback = tl;
            }
            return;
        }
        if (!found || dist2 < best_dist2)
        {
            found = true;
            best_dist2 = dist2;
            best = tl;
        }
    };

    consider(QPointF(box.left(), box.top() - block_h - MARGIN));
    consider(QPointF(box.right() + MARGIN, box.top()));
    consider(QPointF(box.right() + MARGIN, box.center().y() - block_h * 0.5));
    consider(QPointF(box.right() + MARGIN, box.bottom() - block_h));
    consider(QPointF(box.left(), box.bottom() + MARGIN));
    consider(QPointF(box.left() - block_w - MARGIN, box.bottom() - block_h));
    consider(QPointF(box.left() - block_w - MARGIN, box.center().y() - block_h * 0.5));
    consider(QPointF(box.left() - block_w - MARGIN, box.top()));
    consider(QPointF(box.center().x() - block_w * 0.5, box.top() - block_h - MARGIN));
    consider(QPointF(box.center().x() - block_w * 0.5, box.bottom() + MARGIN));
    if (found)
    {
        return best;
    }

    double const rx = 0.5 * box.width() + 0.5 * block_w;
    double const ry = 0.5 * box.height() + 0.5 * block_h;
    for (int ring = 0; ring < MAX_RINGS; ++ring)
    {
        double const clear = MARGIN + 8.0 * static_cast<double>(ring);
        int const n_ang = 12 + ring * 4;
        for (int a = 0; a < n_ang; ++a)
        {
            double const ang = (2.0 * PI * static_cast<double>(a)) / static_cast<double>(n_ang);
            double const cx = anchor.x() + (rx + clear) * std::cos(ang);
            double const cy = anchor.y() + (ry + clear) * std::sin(ang);
            consider(QPointF(cx - 0.5 * block_w, cy - 0.5 * block_h));
        }
        if (found)
        {
            return best;
        }
    }
    return have_fallback ? fallback : best;
}

} // anonymous namespace

QPointF RWorldRenderer2d::map(double world_x, double world_y) const
{
    double screen_x = 0.0;
    double screen_y = 0.0;
    m_view.screen_from_world(world_x, world_y, screen_x, screen_y);
    return QPointF(screen_x, screen_y);
}

void RWorldRenderer2d::paint(QPainter & painter) const
{
    if (!m_world)
    {
        return;
    }

    // Segments and flattened curves share one cosmetic stroke pen.
    QPen geom_pen(GEOMETRY);
    geom_pen.setCosmetic(true);
    geom_pen.setWidthF(GEOMETRY_LINE_WIDTH_PX);
    painter.setPen(geom_pen);

    // 1D straight segments
    std::shared_ptr<SegmentPadFp64> segments = m_world->collect_live_segments();
    for (size_t i = 0; i < segments->size(); ++i)
    {
        painter.drawLine(map(segments->x0(i), segments->y0(i)),
                         map(segments->x1(i), segments->y1(i)));
    }

    // Cubic Beziers; QPainterPath flattens them adaptively, so no sampling.
    std::shared_ptr<CurvePadFp64> curves = m_world->collect_live_curves();
    if (curves->size() > 0)
    {
        QPainterPath path;
        for (size_t i = 0; i < curves->size(); ++i)
        {
            Bezier3dFp64 const c = curves->get(i);
            path.moveTo(map(c.x0(), c.y0()));
            path.cubicTo(map(c.x1(), c.y1()), map(c.x2(), c.y2()), map(c.x3(), c.y3()));
        }
        painter.setBrush(Qt::NoBrush); // stroke the outline only, never fill
        painter.drawPath(path);
    }

    // 0D standalone points as dots with a fixed pixel size at any zoom.
    std::shared_ptr<PointPadFp64> const & points = m_world->points();
    if (points->size() > 0)
    {
        QPen point_pen(GEOMETRY);
        point_pen.setCosmetic(true);
        point_pen.setWidth(GEOMETRY_POINT_WIDTH_PX);
        point_pen.setCapStyle(Qt::RoundCap);
        painter.setPen(point_pen);
        for (size_t i = 0; i < points->size(); ++i)
        {
            painter.drawPoint(map(points->x(i), points->y(i)));
        }
    }
}

void RWorldRenderer2d::paint_shape_annotations(QPainter & painter, int width, int height) const
{
    // Visible world rectangle from the two screen corners; world +Y is up, so
    // screen-top maps to the larger world y.
    double left = 0.0, top = 0.0, right = 0.0, bottom = 0.0;
    m_view.world_from_screen(0.0, 0.0, left, top);
    m_view.world_from_screen(static_cast<double>(width), static_cast<double>(height), right, bottom);

    std::vector<int32_t> const ids = m_world->query_visible(
        std::min(left, right), std::min(top, bottom), std::max(left, right), std::max(top, bottom));

    QPen bbox_pen(OVERLAY_BBOX);
    bbox_pen.setCosmetic(true);
    bbox_pen.setWidthF(1.0);
    bbox_pen.setStyle(Qt::DashLine);
    QPen highlight_pen(OVERLAY_HIGHLIGHT);
    highlight_pen.setCosmetic(true);
    highlight_pen.setWidthF(2.0);
    highlight_pen.setStyle(Qt::DashLine);

    bool const place_labels = m_overlay.shape_ids;
    constexpr double PAD = 4.0;
    constexpr double AXIS_HALF = 5.0;
    LabelObstacles obstacles(static_cast<double>(width), static_cast<double>(height));
    std::vector<QRectF> boxes;
    boxes.reserve(ids.size());
    for (int32_t const id : ids)
    {
        auto const bb = m_world->shape_bbox(id);
        QRectF box(map(bb[0], bb[3]), map(bb[2], bb[1]));
        box = box.normalized().adjusted(-PAD, -PAD, PAD, PAD);
        boxes.push_back(box);
        if (place_labels)
        {
            obstacles.add(box);
        }
    }
    if (place_labels && m_view.pan_y() >= 0.0 && m_view.pan_y() <= static_cast<double>(height))
    {
        obstacles.add(QRectF(
            0.0, m_view.pan_y() - AXIS_HALF, static_cast<double>(width), 2.0 * AXIS_HALF));
    }
    if (place_labels && m_view.pan_x() >= 0.0 && m_view.pan_x() <= static_cast<double>(width))
    {
        obstacles.add(QRectF(
            m_view.pan_x() - AXIS_HALF, 0.0, 2.0 * AXIS_HALF, static_cast<double>(height)));
    }

    // Cap how many shapes get a text label. Past this many visible shapes the
    // labels overlap into an unreadable mass, and each placement searches the
    // obstacle index, so an uncapped count degrades to quadratic work on a
    // dense canvas. Bounding boxes and the highlight box stay drawn for all
    // shapes (both are cheap), and the highlighted shape is always labeled.
    constexpr size_t MAX_LABELED_SHAPES = 200;

    QFontMetricsF const metrics(painter.font());
    double const line_h = metrics.lineSpacing();
    size_t labels_drawn = 0;
    for (size_t i = 0; i < ids.size(); ++i)
    {
        int32_t const id = ids[i];
        QRectF const & box = boxes[i];
        bool const highlighted = (id == m_overlay.highlight_id);

        if (m_overlay.bounding_boxes || highlighted)
        {
            painter.setPen(highlighted ? highlight_pen : bbox_pen);
            painter.setBrush(Qt::NoBrush);
            painter.drawRect(box.adjusted(PAD, PAD, -PAD, -PAD));
        }
        bool const label_this = place_labels && (highlighted || labels_drawn < MAX_LABELED_SHAPES);
        if (!label_this)
        {
            continue;
        }
        ++labels_drawn;

        QStringList rows;
        rows << QStringLiteral("#%1 %2")
                    .arg(id)
                    .arg(QString::fromStdString(shape_type_name(m_world->shape_type_of(id))));
        double widest = 0.0;
        for (QString const & row : rows)
        {
            widest = std::max(widest, metrics.horizontalAdvance(row));
        }
        double const block_w = widest + 4.0;
        double const block_h = line_h * static_cast<double>(rows.size()) + 2.0;

        QPointF const tl = place_label_block(
            box, block_w, block_h, static_cast<double>(width), static_cast<double>(height), obstacles);
        QRectF const placed(tl.x(), tl.y(), block_w, block_h);
        obstacles.add(placed);

        painter.setPen(highlighted ? OVERLAY_HIGHLIGHT : OVERLAY_TEXT);
        painter.drawText(placed, Qt::AlignLeft | Qt::AlignTop, rows.join(QChar('\n')));
    }
}

void RWorldRenderer2d::paint_overlay(QPainter & painter, int width, int height) const
{
    if (m_world && m_overlay.shape_annotations())
    {
        paint_shape_annotations(painter, width, height);
    }
}

void RWorldRenderer2d::paint_canvas(QPainter & painter, int width, int height, bool full_canvas) const
{
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.fillRect(0, 0, width, height, BACKGROUND);

    if (full_canvas)
    {
        paint_chrome(painter, m_view, width, height);
    }

    paint(painter);

    if (full_canvas)
    {
        // Origin dot (cosmetic, fixed pixel size regardless of zoom).
        QPen origin_pen(ORIGIN);
        origin_pen.setCosmetic(true);
        origin_pen.setWidth(6);
        origin_pen.setCapStyle(Qt::RoundCap);
        painter.setPen(origin_pen);
        painter.drawPoint(QPointF(m_view.pan_x(), m_view.pan_y()));
    }

    if (m_overlay.any())
    {
        paint_overlay(painter, width, height);
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
