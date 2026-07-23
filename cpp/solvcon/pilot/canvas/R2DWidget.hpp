#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Strictly-2D drawing widget that paints world geometry with QPainter.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/buffer/small_vector.hpp>
#include <solvcon/pilot/canvas/DrawTool.hpp>
#include <solvcon/pilot/canvas/RWorldRenderer2d.hpp>
#include <solvcon/universe/ViewTransform2d.hpp>
#include <solvcon/universe/World.hpp>

#include <cstdint>
#include <memory>
#include <string>

#include <QPointF>
#include <QWidget>

class QImage;
class QMouseEvent;
class QPainter;
class QPaintEvent;
class QResizeEvent;
class QWheelEvent;

namespace solvcon
{

/**
 * Strictly-2D drawing widget for the pilot. Paints the geometry of a
 * `World<double>` by mapping it to screen space through a 2D view
 * transform, drawn with QPainter.
 *
 * @ingroup group_domain
 */
class R2DWidget
    : public QWidget
{
    Q_OBJECT

    using coord2_type = small_vector<double, 2>;
    using obb_array_type = small_vector<double, 8>;

public:

    explicit R2DWidget(QWidget * parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());

    /// Read-only access to the current view state.
    ViewTransform2dFp64 const & viewTransform() const { return m_view; }

    /// Get the name of the active draw tool.
    std::string drawTool() const { return m_tool->name(); }

    /// Select the active pointer tool by name.
    void setDrawTool(std::string const & name);

    /**
     * Id of the shape selected with the pan tool, or -1 when none is selected.
     * Selection is cleared by switching tools or worlds.
     */
    int32_t selectedShape() const { return m_selected; }

    /**
     * Screen position [x, y] of the selection's rotate handle, or [-1, -1]
     * when nothing is selected. Exposed for tests and tooling.
     */
    coord2_type rotateHandleScreen() const;

    /**
     * Replace the view state. Non-finite inputs are ignored and zoom is
     * clamped to the widget's internal bounds, so the widget's invariants
     * survive any caller. Setting a well-formed transform also disables the
     * deferred auto-centering that runs on the first positive-size resize.
     */
    void setViewTransform(ViewTransform2dFp64 const & v);

    /// Re-center the view so the world origin sits at the widget center.
    void resetView();

    /// Update the world being painted by this widget.
    void updateWorld(std::shared_ptr<WorldFp64> const & world);

    /// Get the world currently being painted.
    std::shared_ptr<WorldFp64> const & world() const { return m_world; }

    Overlay2dOptions const & overlayOptions() const { return m_overlay; }

    void setOverlayOptions(Overlay2dOptions const & overlay)
    {
        m_overlay = overlay;
        update();
    }

    /**
     * Paint the world offscreen into an image at the widget's current size and
     * view, using the given overlay instead of the on-screen one. Draws only
     * the backdrop, chrome, and geometry overlay; the selection box and any
     * in-progress rubber band stay out of the exported frame.
     */
    QImage renderImage(Overlay2dOptions const & overlay) const;

    /// Hook for subsequent stages; current implementation triggers a repaint.
    void requestRepaint() { update(); }

protected:

    void paintEvent(QPaintEvent * event) override;
    void wheelEvent(QWheelEvent * event) override;
    void mousePressEvent(QMouseEvent * event) override;
    void mouseMoveEvent(QMouseEvent * event) override;
    void mouseReleaseEvent(QMouseEvent * event) override;
    void resizeEvent(QResizeEvent * event) override;

private:

    void centerViewOnOrigin();

    /**
     * Paint the active tool's rubber-band preview while a drag is in progress;
     * a no-op for the pan tool or when no drag is underway.
     */
    void paintDrawPreview(QPainter & painter) const;

    /**
     * Paint the selection box and rotate handle for the active selection; a
     * no-op unless the pan tool has a live shape selected.
     */
    void paintSelection(QPainter & painter) const;

    /**
     * Pick the shape under a screen point, or -1. Uses a pixel-sized world
     * tolerance so thin shapes (lines) stay selectable at any zoom.
     */
    int32_t pickShapeAt(QPointF const & screen_pos) const;

    /**
     * Screen position of the rotate handle for the current selection; only
     * meaningful while a live shape is selected.
     */
    QPointF rotateHandlePos() const;

    /**
     * World center of the selection's oriented bounding box, the pivot a
     * rotate drag turns about. Only meaningful while a shape is selected.
     */
    coord2_type selectionCenterWorld() const;

    /// True when `screen_pos` lands on the selection's rotate handle.
    bool isOnRotateHandle(QPointF const & screen_pos) const;

    /**
     * Close the world's compound operation that brackets a move or rotate
     * drag, so the whole gesture is a single undo step. Safe to call when no
     * such drag is active; a no-op then.
     */
    void finishEdit();

    /// Which gesture the current pan-tool left-drag performs.
    enum class EditDrag
    {
        None, ///< No left-drag in progress.
        View, ///< Panning the view.
        Move, ///< Translating the selected shape.
        Rotate, ///< Rotating the selected shape about a fixed pivot.
    };

    ViewTransform2dFp64 m_view;
    std::shared_ptr<WorldFp64> m_world;
    Overlay2dOptions m_overlay;
    bool m_view_modified = false;
    QPointF m_last_mouse_pos;

    std::unique_ptr<DrawToolBase> m_tool; ///< Active tool, shared by every 2D canvas.
    bool m_drawing = false; ///< Whether a shape drag is in progress.
    double m_draw_start_x = 0.0; ///< Press anchor, world coordinates.
    double m_draw_start_y = 0.0;
    double m_draw_current_x = 0.0; ///< Live pointer, world coordinates.
    double m_draw_current_y = 0.0;

    int32_t m_selected = -1; ///< Pan-tool selected shape id, or -1.
    EditDrag m_drag = EditDrag::None; ///< Active pan-tool left-drag gesture.
    double m_move_last_x = 0.0; ///< Previous pointer, world coordinates (Move).
    double m_move_last_y = 0.0;
    double m_rotate_cx = 0.0; ///< Rotation pivot, world coordinates (Rotate).
    double m_rotate_cy = 0.0;
    double m_rotate_last_angle = 0.0; ///< Previous pointer angle about pivot.

}; /* end class R2DWidget */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
