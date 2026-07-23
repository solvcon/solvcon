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

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/RAxisGizmo.hpp>
#include <solvcon/pilot/RColormap.hpp>
#include <solvcon/pilot/RScalarBar.hpp>
#include <solvcon/pilot/RScene.hpp>
#include <solvcon/pilot/RTextOverlay.hpp>
#include <solvcon/pilot/RDrawable.hpp>

#include <solvcon/solvcon.hpp>

#include <rhi/qrhi.h>

#include <QImage>
#include <QPoint>
#include <QRhiWidget>
#include <QVector3D>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace solvcon
{

class RScalarField;

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

    // A scene of several named mesh objects, each with its own model transform
    // and visibility, listed by an outliner. addObject registers a mesh as a
    // named, lit surface; the setters drive it by name (an unknown name is a
    // no-op).
    void addObject(std::string const & name, std::shared_ptr<StaticMesh> const & mesh);
    void setObjectTransform(
        std::string const & name,
        float tx,
        float ty,
        float tz,
        float sx,
        float sy,
        float sz);
    void setObjectVisible(std::string const & name, bool visible);
    void setObjectOpacity(std::string const & name, float opacity);
    std::vector<std::string> objectNames() const;

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

    /// Replace the scalar field: triangles from a vertex table (nvert, 3), a
    /// per-vertex scalar table (nvert,), and a triangle index table
    /// (ntri, 3), colored on the GPU through the active colormap LUT.
    /// Swappable at runtime.
    void updateScalarField(
        SimpleArray<float> const & vertices,
        SimpleArray<float> const & scalars,
        SimpleArray<uint32_t> const & indices);

    /// Select the named colormap ("viridis", "coolwarm", "jet",
    /// "grayscale") for the scalar field and the scalar bar.
    void setColormap(std::string const & name);
    std::string colormap() const { return m_colormap.name(); }

    /// Pin the scalar-to-color mapping range. Until called, each
    /// updateScalarField maps its own data min/max.
    void setScalarRange(float lo, float hi);
    std::pair<float, float> scalarRange() const;

    /// Show or hide the on-screen scalar bar.
    void showScalarBar(bool show);

    /// Set the title text over the scalar bar.
    void setScalarBarTitle(std::string const & title);

    /// Show or hide the highlight ribbon for boundary set @p ibc.
    void showBoundary(int ibc, bool show);

    /// Show or hide the domain's boundary (feature) edges as one bold overlay.
    void showFeatureEdges(bool show);

    /// Show or hide a short arrow at every face center along its normal.
    void showNormals(bool show);

    /// The result of a pick: the entity kind ("cell", "node", "face", or
    /// "none" for a miss), its id, its element type (for a cell), a measure
    /// (cell volume or face area), and its centroid.
    struct PickResult
    {
        std::string kind = "none";
        int id = -1;
        int type = -1;
        double measure = 0.0;
        QVector3D centroid;

        bool hit() const { return "none" != kind; }
    }; /* end struct PickResult */

    // Pick the entity under the widget pixel (x, y): a cell (by ray-cast
    // against the surface), the nearest node, or the nearest boundary face.
    // The picked entity is highlighted and kept as the selection.
    PickResult pickCell(int x, int y);
    PickResult pickNode(int x, int y);
    PickResult pickFace(int x, int y);

    /// Drop the current selection and its highlight.
    void clearSelection();
    bool hasSelection() const { return m_has_selection; }

    /// Measure and draw the distance between two world points; returns the
    /// distance.
    double measureDistance(QVector3D const & p0, QVector3D const & p1);

    /// Measure and draw the angle at @p p1 between the arms to @p p0 and
    /// @p p2; returns the angle in degrees.
    double measureAngle(QVector3D const & p0, QVector3D const & p1, QVector3D const & p2);

    /// Remove the measurement ruler.
    void clearMeasurements();

    /// Clip the mesh by a plane, keeping the side the normal points away from
    /// (the near half, toward the normal, is removed). Returns the number of
    /// surface primitives kept.
    int addClip(QVector3D const & origin, QVector3D const & normal);

    /// Slice the mesh by a plane, drawing the cross-section outline where the
    /// plane cuts the cells. Returns the number of segments drawn.
    int addSlice(QVector3D const & origin, QVector3D const & normal);

    /// Remove the slice and clip filters.
    void clearFilters();

    // Color the mesh cells by a categorical attribute through the qualitative
    // colormap with a legend: element type, cell group, or boundary set. Each
    // replaces the field with a per-cell-colored surface; clearCellColoring
    // drops it.
    void colorByCellType();
    void colorByCellGroup();
    void colorByBoundary();
    void clearCellColoring();

    /// Color the mesh by a per-cell geometric quality metric through the
    /// continuous colormap and scalar bar. @p metric is one of "volume",
    /// "aspect_ratio", "skewness", "min_angle", "max_angle".
    void colorByQuality(std::string const & metric);

    /// The (min, max) range of the named quality metric over the cells.
    std::pair<float, float> qualityRange(std::string const & metric) const;

    /// Frame the camera so the whole domain is in view.
    void fitCameraToScene();

    /// Point the camera along a named view preset and frame the scene:
    /// "front", "back", "left", "right", "top", "bottom", the axis names
    /// "+x".."-z", or "iso".
    void setView(std::string const & name);

    /// Set the projection independently of the 2D/3D default: "auto",
    /// "parallel", or "perspective".
    void setProjection(std::string const & name);
    std::string projection() const;

    /// Select the orbit style: "turntable" (up axis fixed, horizon level) or
    /// "trackball" (free tumble that can roll the horizon).
    void setOrbitStyle(std::string const & name);
    std::string orbitStyle() const;

    /// Set the orbit pivot, the point the orbit swings the eye around.
    void setPivot(float x, float y, float z);

    /// Recenter and frame the whole scene (the selection once picking lands).
    void frameSelected();

    /// Frame the current selection (from a pick), or the whole scene when
    /// nothing is selected.
    void zoomToSelection();

    /// Reset the camera to the fit-to-scene default: auto projection,
    /// turntable orbit, the whole scene framed.
    void resetCamera();

    /// Choose the mouse navigation mapping: "default" (left rotates, other
    /// buttons pan) or "blender" (middle orbits; Shift+middle pans;
    /// Ctrl+middle zooms; Alt+middle recenters the pivot; Alt+left aliases the
    /// middle button for trackpads without one).
    void setNavigationMapping(std::string const & name);
    std::string navigationMapping() const;

    /// Scale the orbit/look speed for drags.
    void setOrbitSensitivity(float factor);

    /// Orbit by a fixed number of degrees, a discrete step.
    void orbitStep(float yaw_deg, float pitch_deg);

    /// Show or hide the orientation-guide triad in the corner.
    void showAxis(bool show);

    /// Show or hide a bounding-box cube-axes grid with tick marks.
    void showCubeAxes(bool show);
    /// The cube-axes tick coordinates per axis (x, y, z), in mesh units.
    std::vector<float> cubeAxesTicks(int axis) const;

    /// Set the figure title drawn as a top overlay; an empty string clears it.
    void setTitle(std::string const & text);
    std::string title() const;

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

    /// Render the scene offscreen at a requested width and height, decoupled
    /// from the widget size, optionally over a transparent background. The
    /// widget size is restored afterward.
    QImage renderToImage(int width, int height, bool transparent = false);

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

    /// Swap in a new field drawable (color or scalar) and re-frame the
    /// scene around its bounding box.
    template <typename FieldT>
    void installField(std::unique_ptr<FieldT> field);

    /// Build a categorical scalar field over the mesh surface primitives (2D
    /// cells, or 3D boundary faces) from a per-primitive category value, and
    /// install it colored through the qualitative colormap with a legend
    /// titled @p title. @p primitive_category is indexed in the surface build
    /// order.
    void installCategoryField(
        std::vector<int32_t> const & primitive_category,
        std::string const & title);

    /// Install a continuous scalar field over the surface primitives from a
    /// per-primitive value, colored through the current colormap and
    /// auto-ranged, with a legend titled @p title.
    void installMetricField(
        std::vector<float> const & primitive_value,
        std::string const & title);

    /// Fan-triangulate the mesh surface primitives (2D cells, or 3D boundary
    /// faces) into the interleaved position/scalar/index arrays, tagging every
    /// vertex of a primitive with its @p primitive_scalar (indexed in build
    /// order) so each face reads one value.
    void collectSurfaceScalars(
        std::vector<float> const & primitive_scalar,
        std::vector<float> & verts,
        std::vector<float> & scals,
        std::vector<uint32_t> & tris) const;

    /// Back-project the widget pixel (x, y) to a world-space ray. Returns
    /// false when the viewport or the view-projection is degenerate.
    bool computePickRay(int x, int y, QVector3D & origin, QVector3D & dir) const;

    /// Highlight the picked cell (its surface triangles) and record it as the
    /// selection with its bounding box; a kind other than "cell" records the
    /// selection point without a surface highlight.
    void setSelection(std::string const & kind, int id, QVector3D const & lo, QVector3D const & hi);

    QRhi * m_rhi = nullptr; ///< Tracked to detect device changes.
    QRhiRenderPassDescriptor * m_rpdesc = nullptr; ///< Tracked to detect target changes.
    int m_sample_count = 0; ///< Tracked to detect MSAA changes.

    RScene m_scene;
    RAxisGizmo m_gizmo;
    RScalarBar m_scalar_bar;
    // Non-owning; the drawables live in the scene. One per mesh representation.
    RDrawable * m_mesh_surface = nullptr;
    RDrawable * m_mesh_frame = nullptr;
    RDrawable * m_mesh_points = nullptr;
    RDrawable * m_field = nullptr;
    RScalarField * m_scalar_field = nullptr; ///< m_field when it is scalar.

    // Per-style show flags; any combination may be on at once. Only the
    // wireframe is on by default, so a fresh viewer looks unchanged.
    bool m_show_surface = false;
    bool m_show_wireframe = true;
    bool m_show_points = false;
    bool m_mesh_shown = true; ///< The showMesh toggle, applied atop the styles.

    RDrawable * m_selection = nullptr; ///< Highlight of the picked entity.
    RDrawable * m_ruler = nullptr; ///< The measurement ruler segments.
    RDrawable * m_cube_axes = nullptr; ///< The bounding-box cube-axes grid.
    RDrawable * m_clip = nullptr; ///< The clipped surface drawable.
    RDrawable * m_slice = nullptr; ///< The slice cross-section outline.

    /// A named scene object: its drawable (owned by the scene) and its mesh,
    /// kept so a transform can re-extend the framing box.
    struct ObjectEntry
    {
        RDrawable * drawable = nullptr;
        std::shared_ptr<StaticMesh> mesh;
    }; /* end struct ObjectEntry */
    std::map<std::string, ObjectEntry> m_objects;
    std::vector<float> m_ticks_x; ///< Cube-axes tick coordinates per axis.
    std::vector<float> m_ticks_y;
    std::vector<float> m_ticks_z;
    RTextOverlay m_title; ///< The figure-title overlay.
    bool m_has_selection = false;
    std::string m_selection_kind = "none";
    int m_selection_id = -1;
    QVector3D m_selection_lo;
    QVector3D m_selection_hi;

    RColormap m_colormap = RColormap::named("viridis");
    bool m_range_pinned = false; ///< setScalarRange overrides auto-ranging.
    float m_range_lo = 0.0f;
    float m_range_hi = 1.0f;
    QVector3D m_field_lo; ///< Field bounding box, kept for re-framing.
    QVector3D m_field_hi;
    bool m_has_field_bbox = false;

    std::shared_ptr<StaticMesh> m_mesh;

    bool m_transparent_capture = false; ///< Clear alpha 0 for a capture.

    QPoint m_last_mouse_pos; ///< Last cursor position during a drag.

    /// What the current drag does, chosen at press time from the button and
    /// modifiers through the active navigation mapping.
    enum class DragAction
    {
        Rotate,
        Pan,
        Zoom,
    };
    DragAction m_drag_action = DragAction::Rotate;
    std::string m_nav_mapping = "blender"; ///< "default" or "blender".

}; /* end class RDomainWidget */

template <typename FieldT>
void RDomainWidget::installField(std::unique_ptr<FieldT> field)
{
    // Drop the previous field and replace it; the field is swappable.
    m_scene.removeDrawable(m_field);
    m_field = nullptr;
    m_scalar_field = nullptr;
    m_has_field_bbox = false;

    if (field->hasGeometry())
    {
        QVector3D const lo = field->bboxLo();
        QVector3D const hi = field->bboxHi();
        // With no mesh to set the dimensionality, infer it: a field with no
        // depth extent is viewed head-on like a 2D domain.
        if (!m_scene.hasBoundingBox())
        {
            float const span = (hi - lo).length();
            m_scene.setDimension(((hi.z() - lo.z()) > 1.0e-6f * span) ? 3 : 2);
        }
        m_scene.extendBoundingBox(lo, hi);
        m_field_lo = lo;
        m_field_hi = hi;
        m_has_field_bbox = true;
        m_field = field.get();
        m_scene.addDrawable(std::move(field));
        m_scene.fitCameraToScene(viewportAspect());
    }

    update();
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
