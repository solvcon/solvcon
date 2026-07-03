/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RDomainWidget.hpp> // Must be the first include.

#include <solvcon/pilot/RBoundary.hpp>
#include <solvcon/pilot/RField.hpp>
#include <solvcon/pilot/RMeshFrame.hpp>

#include <QGestureEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QNativeGestureEvent>
#include <QPinchGesture>
#include <QWheelEvent>

#include <algorithm>
#include <limits>

namespace solvcon
{

RDomainWidget::RDomainWidget(QWidget * parent)
    : QRhiWidget(parent)
{
    // Accept keyboard focus so the first-person movement keys reach the
    // widget, and track the mouse for drag-based navigation.
    setFocusPolicy(Qt::StrongFocus);
    // Receive touchscreen pinch gestures (trackpad pinches arrive as native
    // gesture events, which need no grab).
    grabGesture(Qt::PinchGesture);
}

float RDomainWidget::viewportAspect() const
{
    int const h = height();
    return (h > 0) ? static_cast<float>(width()) / static_cast<float>(h) : 1.0f;
}

RDomainWidget::~RDomainWidget() = default;

QImage RDomainWidget::grabImage()
{
    return grabFramebuffer();
}

void RDomainWidget::updateMesh(std::shared_ptr<StaticMesh> const & mesh)
{
    // Drop the previous mesh drawables and replace them; a new mesh redefines
    // the framing, so the bounding box is recomputed from scratch.
    m_scene.removeDrawable(m_mesh_surface);
    m_scene.removeDrawable(m_mesh_frame);
    m_scene.removeDrawable(m_mesh_points);
    m_mesh_surface = nullptr;
    m_mesh_frame = nullptr;
    m_mesh_points = nullptr;

    m_mesh = mesh;

    // Build one drawable per representation and switch between them by
    // visibility; rebuilding on every toggle would be wasteful.
    auto surface = std::make_unique<RMeshFrame>(mesh, RMeshFrame::Style::Surface);
    m_mesh_surface = surface.get();
    m_scene.addDrawable(std::move(surface));
    auto frame = std::make_unique<RMeshFrame>(mesh, RMeshFrame::Style::Wireframe);
    m_mesh_frame = frame.get();
    m_scene.addDrawable(std::move(frame));
    auto points = std::make_unique<RMeshFrame>(mesh, RMeshFrame::Style::Points);
    m_mesh_points = points.get();
    m_scene.addDrawable(std::move(points));
    applyMeshVisibility();

    StaticMesh const & mh = *mesh;
    m_scene.setDimension(mh.ndim());
    // For now, limit 2D domain to the in-plane pan/zoom whose wheel scales the
    // orthographic box.
    if (2 == mh.ndim())
    {
        m_scene.camera().setMode(RCameraController::Mode::PanZoom);
    }
    QVector3D lo(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    QVector3D hi(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest());
    bool const is_3d = (3 == mh.ndim());
    for (uint32_t ind = 0; ind < mh.nnode(); ++ind)
    {
        float const x = static_cast<float>(mh.ndcrd(ind, 0));
        float const y = static_cast<float>(mh.ndcrd(ind, 1));
        float const z = is_3d ? static_cast<float>(mh.ndcrd(ind, 2)) : 0.0f;
        lo = QVector3D(std::min(lo.x(), x), std::min(lo.y(), y), std::min(lo.z(), z));
        hi = QVector3D(std::max(hi.x(), x), std::max(hi.y(), y), std::max(hi.z(), z));
    }
    m_scene.resetBoundingBox();
    if (mh.nnode() > 0)
    {
        m_scene.extendBoundingBox(lo, hi);
    }
    // A field set earlier still draws, so keep it inside the framed box.
    if (nullptr != m_field)
    {
        auto * field = static_cast<RField *>(m_field);
        m_scene.extendBoundingBox(field->bboxLo(), field->bboxHi());
    }

    m_scene.fitCameraToScene(viewportAspect());
    update();
}

void RDomainWidget::showMesh(bool show)
{
    m_mesh_shown = show;
    applyMeshVisibility();
    update();
}

void RDomainWidget::showMeshStyle(std::string const & name, bool show)
{
    if ("surface" == name)
    {
        m_show_surface = show;
    }
    else if ("wireframe" == name)
    {
        m_show_wireframe = show;
    }
    else if ("points" == name)
    {
        m_show_points = show;
    }
    else
    {
        return; // Ignore an unknown name.
    }
    applyMeshVisibility();
    update();
}

bool RDomainWidget::meshStyleShown(std::string const & name) const
{
    if ("surface" == name)
    {
        return m_show_surface;
    }
    if ("wireframe" == name)
    {
        return m_show_wireframe;
    }
    if ("points" == name)
    {
        return m_show_points;
    }
    return false;
}

void RDomainWidget::applyMeshVisibility()
{
    if (nullptr != m_mesh_surface)
    {
        m_mesh_surface->setVisible(m_mesh_shown && m_show_surface);
    }
    if (nullptr != m_mesh_frame)
    {
        m_mesh_frame->setVisible(m_mesh_shown && m_show_wireframe);
    }
    if (nullptr != m_mesh_points)
    {
        m_mesh_points->setVisible(m_mesh_shown && m_show_points);
    }
}

void RDomainWidget::updateColorField(
    SimpleArray<float> const & vertices,
    SimpleArray<float> const & colors,
    SimpleArray<uint32_t> const & indices)
{
    // Drop the previous field and replace it; the field is swappable.
    m_scene.removeDrawable(m_field);
    m_field = nullptr;

    auto field = std::make_unique<RField>(vertices, colors, indices);
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
        m_field = field.get();
        m_scene.addDrawable(std::move(field));
        m_scene.fitCameraToScene(viewportAspect());
    }

    update();
}

void RDomainWidget::showBoundary(int ibc, bool show)
{
    // Remove an existing highlight for this set so a re-show stays single and
    // a hide leaves none behind.
    m_scene.removeDrawableIf(
        [ibc](RDrawable const * d)
        {
            auto const * boundary = dynamic_cast<RBoundary const *>(d);
            return nullptr != boundary && boundary->ibc() == ibc;
        });

    if (show && nullptr != m_mesh)
    {
        auto boundary = std::make_unique<RBoundary>(m_mesh, ibc);
        if (boundary->hasGeometry())
        {
            m_scene.addDrawable(std::move(boundary));
        }
    }

    update();
}

void RDomainWidget::fitCameraToScene()
{
    m_scene.fitCameraToScene(viewportAspect());
    update();
}

void RDomainWidget::setCameraMode(std::string const & name)
{
    m_scene.camera().setMode(RCameraController::modeFromName(name));
    update();
}

std::string RDomainWidget::cameraMode() const
{
    return RCameraController::modeName(m_scene.camera().mode());
}

QVector3D RDomainWidget::cameraPosition() const
{
    return m_scene.camera().position();
}

void RDomainWidget::setCameraPosition(QVector3D const & position)
{
    m_scene.camera().setPosition(position);
    update();
}

QVector3D RDomainWidget::cameraTarget() const
{
    return m_scene.camera().target();
}

void RDomainWidget::setCameraTarget(QVector3D const & target)
{
    m_scene.camera().setTarget(target);
    update();
}

QVector3D RDomainWidget::cameraUp() const
{
    return m_scene.camera().up();
}

void RDomainWidget::setCameraUp(QVector3D const & up)
{
    m_scene.camera().setUp(up);
    update();
}

void RDomainWidget::rotateCamera(float dx, float dy)
{
    m_scene.camera().rotate(dx, dy);
    update();
}

void RDomainWidget::panCamera(float dx, float dy)
{
    m_scene.camera().pan(dx, dy);
    update();
}

void RDomainWidget::zoomCamera(float steps)
{
    m_scene.camera().zoom(steps);
    update();
}

void RDomainWidget::pinchCamera(float factor)
{
    m_scene.camera().pinch(factor);
    update();
}

void RDomainWidget::mousePressEvent(QMouseEvent * event)
{
    m_last_mouse_pos = event->position().toPoint();
    m_panning = (event->button() != Qt::LeftButton);
}

void RDomainWidget::mouseMoveEvent(QMouseEvent * event)
{
    if (event->buttons() == Qt::NoButton)
    {
        return;
    }
    QPoint const pos = event->position().toPoint();
    float const dx = static_cast<float>(pos.x() - m_last_mouse_pos.x());
    float const dy = static_cast<float>(pos.y() - m_last_mouse_pos.y());
    m_last_mouse_pos = pos;
    if (m_panning)
    {
        m_scene.camera().pan(dx, dy);
    }
    else
    {
        m_scene.camera().rotate(dx, dy);
    }
    update();
}

void RDomainWidget::mouseReleaseEvent(QMouseEvent *)
{
    m_panning = false;
}

void RDomainWidget::wheelEvent(QWheelEvent * event)
{
    // One wheel notch is 120 eighths of a degree.
    float const steps = static_cast<float>(event->angleDelta().y()) / 120.0f;
    m_scene.camera().zoom(steps);
    update();
}

bool RDomainWidget::event(QEvent * event)
{
    // A trackpad pinch arrives as a native zoom gesture whose value() is the
    // incremental magnification (positive spreads, negative pinches); a
    // touchscreen pinch arrives as a QPinchGesture whose scaleFactor() is the
    // incremental multiplier. Feed both to pinchCamera as a scale around 1.
    if (QEvent::NativeGesture == event->type())
    {
        auto * gesture = static_cast<QNativeGestureEvent *>(event);
        if (Qt::ZoomNativeGesture == gesture->gestureType())
        {
            pinchCamera(1.0f + static_cast<float>(gesture->value()));
            return true;
        }
    }
    else if (QEvent::Gesture == event->type())
    {
        auto * gesture = static_cast<QGestureEvent *>(event);
        if (auto * pinch = static_cast<QPinchGesture *>(gesture->gesture(Qt::PinchGesture)))
        {
            pinchCamera(static_cast<float>(pinch->scaleFactor()));
            return true;
        }
    }
    return QRhiWidget::event(event);
}

void RDomainWidget::keyPressEvent(QKeyEvent * event)
{
    // First-person movement; a step is a tenth of the scene size.
    constexpr float step = 0.1f;
    switch (event->key())
    {
    case Qt::Key_W:
    case Qt::Key_Up:
        m_scene.camera().moveForward(step);
        break;
    case Qt::Key_S:
    case Qt::Key_Down:
        m_scene.camera().moveForward(-step);
        break;
    case Qt::Key_D:
    case Qt::Key_Right:
        m_scene.camera().moveRight(step);
        break;
    case Qt::Key_A:
    case Qt::Key_Left:
        m_scene.camera().moveRight(-step);
        break;
    default:
        QRhiWidget::keyPressEvent(event);
        return;
    }
    update();
}

void RDomainWidget::showAxis(bool show)
{
    m_gizmo.setVisible(show);
    update();
}

void RDomainWidget::initialize(QRhiCommandBuffer *)
{
    QRhiRenderPassDescriptor * const rpdesc = renderTarget()->renderPassDescriptor();
    if (m_rhi != rhi() || m_rpdesc != rpdesc || m_sample_count != sampleCount())
    {
        // The graphics device, render target, or sample count changed; drop
        // every device resource so the drawables rebuild against the new one
        // (the pipelines are tied to the render-pass descriptor).
        m_scene.releaseAll();
        m_gizmo.release();
        m_rhi = rhi();
        m_rpdesc = rpdesc;
        m_sample_count = sampleCount();
    }
}

void RDomainWidget::render(QRhiCommandBuffer * cb)
{
    QRhiResourceUpdateBatch * batch = m_rhi->nextResourceUpdateBatch();

    QSize const pixel_size = renderTarget()->pixelSize();
    QMatrix4x4 const view_proj = m_scene.viewProjection(pixel_size, m_rhi);
    QRhiRenderPassDescriptor * const rpdesc = renderTarget()->renderPassDescriptor();

    // A camera-following headlight: the lit surface faces toward the eye read
    // brightest. Non-lit drawables ignore the direction.
    QVector3D light_dir = m_scene.camera().position() - m_scene.camera().target();
    if (light_dir.lengthSquared() <= 0.0f)
    {
        light_dir = QVector3D(0.0f, 0.0f, 1.0f);
    }
    light_dir.normalize();

    for (std::unique_ptr<RDrawable> const & drawable : m_scene.drawables())
    {
        drawable->prepare(m_rhi, rpdesc, sampleCount(), batch);
        drawable->setLightDir(light_dir);
        drawable->updateUniform(batch, view_proj);
    }

    // The orientation guide shows two axes for a 2D domain and three for 3D,
    // oriented by the main camera. Its resources update before the pass.
    m_gizmo.setAxisCount((2 == m_scene.dimension()) ? 2 : 3);
    QVector3D const camera_forward = m_scene.camera().target() - m_scene.camera().position();
    m_gizmo.update(
        m_rhi, rpdesc, sampleCount(), pixel_size, camera_forward, m_scene.camera().up(), batch);

    QColor const clear_color = QColor::fromRgbF(1.0f, 1.0f, 1.0f, 1.0f);
    QRhiDepthStencilClearValue const ds_clear(1.0f, 0);

    cb->beginPass(renderTarget(), clear_color, ds_clear, batch);
    cb->setViewport(QRhiViewport(
        0, 0, float(pixel_size.width()), float(pixel_size.height())));
    for (std::unique_ptr<RDrawable> const & drawable : m_scene.drawables())
    {
        drawable->draw(cb);
    }
    m_gizmo.draw(cb);
    cb->endPass();
}

void RDomainWidget::releaseResources()
{
    m_scene.releaseAll();
    m_gizmo.release();
    m_rhi = nullptr;
    m_rpdesc = nullptr;
    m_sample_count = 0;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
