#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Interactive QRhi widget that renders spatial domains and fields on
 * unstructured meshes and routes camera control from Python.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/RAxisGizmo.hpp>
#include <solvcon/pilot/RScene.hpp>
#include <solvcon/pilot/RDrawable.hpp>

#include <solvcon/solvcon.hpp>

#include <rhi/qrhi.h>

#include <QImage>
#include <QPoint>
#include <QRhiWidget>
#include <QVector3D>

#include <memory>
#include <string>

namespace solvcon
{

/**
 * @brief Interactive 2D/3D viewer for spatial domains and fields on
 * unstructured meshes, rendered with QRhi and controlled from Python.
 *
 * It derived from QRhiWidget: Qt owns the swapchain, color and depth buffers,
 * and drives the render loop through initialize()/render(). The widget hosts
 * an RScene (the drawables, the domain bounding box, and the framing
 * camera) and drives it.
 *
 * @ingroup group_domain
 */
class RDomainWidget
    : public QRhiWidget
{
    Q_OBJECT

public:

    explicit RDomainWidget(QWidget * parent = nullptr);
    ~RDomainWidget() override;

    /// Replace the rendered mesh with the wireframe of @p mesh.
    void updateMesh(std::shared_ptr<StaticMesh> const & mesh);

    /// Show or hide the mesh, in whatever representation is active.
    void showMesh(bool show);

    /// Show or hide one mesh style independently of the others, so any
    /// combination can be drawn at once (a wireframe over a lit surface, say).
    /// @p name is "surface" (lit shaded), "wireframe" (edges), or "points"
    /// (nodes); an unknown name is ignored.
    void showMeshStyle(std::string const & name, bool show);
    bool meshStyleShown(std::string const & name) const;

    /// Set the mesh wireframe opacity, a [0, 1] fraction (1 opaque).
    void setMeshOpacity(float opacity);

    /// Set the color-field surface opacity, a [0, 1] fraction (1 opaque).
    /// Lower it to read the wireframe drawn over the shaded surface.
    void setFieldOpacity(float opacity);

    /// Replace the colored field: per-vertex-colored triangles from a vertex
    /// table (nvert, 3), a matching color table (nvert, 3), and a triangle
    /// index table (ntri, 3). Swappable at runtime.
    void updateColorField(
        SimpleArray<float> const & vertices,
        SimpleArray<float> const & colors,
        SimpleArray<uint32_t> const & indices);

    /// Show or hide the highlight ribbon for boundary set @p ibc.
    void showBoundary(int ibc, bool show);

    /// Show or hide the domain's boundary (feature) edges as one bold overlay.
    void showFeatureEdges(bool show);

    /// Show or hide a short arrow at every face center along its normal.
    void showNormals(bool show);

    /// Frame the camera so the whole domain is in view.
    void fitCameraToScene();

    /// Show or hide the orientation-guide triad in the corner.
    void showAxis(bool show);

    /// Select the camera mode: "pan" (2D pan/zoom), "fps" (3D fly-through), or
    /// "orbit" (3D orbit around the target).
    void setCameraMode(std::string const & name);
    std::string cameraMode() const;

    // Programmatic camera pose, so Python navigates as well as the mouse.
    QVector3D cameraPosition() const;
    void setCameraPosition(QVector3D const & position);
    QVector3D cameraTarget() const;
    void setCameraTarget(QVector3D const & target);
    QVector3D cameraUp() const;
    void setCameraUp(QVector3D const & up);

    // Mode-aware interaction primitives (what the mouse and wheel drive).
    void rotateCamera(float dx, float dy);
    void panCamera(float dx, float dy);
    void zoomCamera(float steps);
    /// Zoom by a multiplicative pinch @p factor (what a trackpad/touch pinch
    /// drives); greater than 1 zooms in, less than 1 zooms out.
    void pinchCamera(float factor);

    std::shared_ptr<StaticMesh> mesh() const { return m_mesh; }

    /// Render the current frame offscreen and return it as a QImage. Thin
    /// wrapper over QRhiWidget::grabFramebuffer() for the Python control path.
    QImage grabImage();

protected:

    void initialize(QRhiCommandBuffer * cb) override;
    void render(QRhiCommandBuffer * cb) override;
    void releaseResources() override;

    void mousePressEvent(QMouseEvent * event) override;
    void mouseMoveEvent(QMouseEvent * event) override;
    void mouseReleaseEvent(QMouseEvent * event) override;
    void wheelEvent(QWheelEvent * event) override;
    void keyPressEvent(QKeyEvent * event) override;
    bool event(QEvent * event) override;

private:

    float viewportAspect() const;

    /// Push the per-style show flags and the overall mesh-shown flag onto the
    /// three mesh drawables' visibility.
    void applyMeshVisibility();

    QRhi * m_rhi = nullptr; ///< Tracked to detect device changes.
    QRhiRenderPassDescriptor * m_rpdesc = nullptr; ///< Tracked to detect target changes.
    int m_sample_count = 0; ///< Tracked to detect MSAA changes.

    RScene m_scene;
    RAxisGizmo m_gizmo;
    // Non-owning; the drawables live in the scene. One per mesh representation.
    RDrawable * m_mesh_surface = nullptr;
    RDrawable * m_mesh_frame = nullptr;
    RDrawable * m_mesh_points = nullptr;
    RDrawable * m_field = nullptr;

    // Per-style show flags; any combination may be on at once. Only the
    // wireframe is on by default, so a fresh viewer looks unchanged.
    bool m_show_surface = false;
    bool m_show_wireframe = true;
    bool m_show_points = false;
    bool m_mesh_shown = true; ///< The showMesh toggle, applied atop the styles.

    std::shared_ptr<StaticMesh> m_mesh;

    QPoint m_last_mouse_pos; ///< Last cursor position during a drag.
    bool m_panning = false; ///< A non-left-button drag pans in both modes.

}; /* end class RDomainWidget */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
