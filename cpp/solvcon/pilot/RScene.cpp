/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RScene.hpp> // Must be the first include.

#include <algorithm>
#include <cmath>

namespace solvcon
{

namespace
{

// Vertical field of view used for the 3D perspective projection.
constexpr float FOV_DEGREES = 45.0f;

} /* end namespace */

void RScene::addDrawable(std::unique_ptr<RDrawable> drawable)
{
    m_drawables.push_back(std::move(drawable));
}

void RScene::removeDrawable(RDrawable const * drawable)
{
    if (nullptr == drawable)
    {
        return;
    }
    std::erase_if(
        m_drawables,
        [drawable](std::unique_ptr<RDrawable> const & d)
        { return d.get() == drawable; });
}

void RScene::removeDrawableIf(std::function<bool(RDrawable const *)> const & pred)
{
    std::erase_if(
        m_drawables,
        [&pred](std::unique_ptr<RDrawable> const & d)
        { return pred(d.get()); });
}

void RScene::releaseAll()
{
    for (std::unique_ptr<RDrawable> const & drawable : m_drawables)
    {
        drawable->release();
    }
}

void RScene::resetBoundingBox()
{
    m_has_bbox = false;
}

void RScene::extendBoundingBox(QVector3D const & lo, QVector3D const & hi)
{
    if (!m_has_bbox)
    {
        m_bbox_lo = lo;
        m_bbox_hi = hi;
        m_has_bbox = true;
        return;
    }
    m_bbox_lo = QVector3D(
        std::min(m_bbox_lo.x(), lo.x()),
        std::min(m_bbox_lo.y(), lo.y()),
        std::min(m_bbox_lo.z(), lo.z()));
    m_bbox_hi = QVector3D(
        std::max(m_bbox_hi.x(), hi.x()),
        std::max(m_bbox_hi.y(), hi.y()),
        std::max(m_bbox_hi.z(), hi.z()));
}

float RScene::boundingRadius() const
{
    float const radius = (m_bbox_hi - m_bbox_lo).length() * 0.5f;
    return (radius > 0.0f) ? radius : 1.0f;
}

float RScene::framingRadius() const
{
    if (m_has_frame_box)
    {
        float const radius = (m_frame_hi - m_frame_lo).length() * 0.5f;
        return (radius > 0.0f) ? radius : 1.0f;
    }
    return boundingRadius();
}

void RScene::fitCameraToScene(float aspect)
{
    if (!m_has_bbox)
    {
        m_camera.setPosition(QVector3D(0.0f, 0.0f, 1.0f));
        m_camera.setTarget(QVector3D(0.0f, 0.0f, 0.0f));
        m_camera.setUp(QVector3D(0.0f, 1.0f, 0.0f));
        return;
    }
    m_has_frame_box = false;
    m_camera.fitToBoundingBox(m_bbox_lo, m_bbox_hi, m_ndim, aspect);
}

void RScene::frameBox(QVector3D const & lo, QVector3D const & hi, float aspect)
{
    m_frame_lo = lo;
    m_frame_hi = hi;
    m_has_frame_box = true;
    m_camera.fitToBoundingBox(lo, hi, m_ndim, aspect);
}

void RScene::setProjection(std::string const & name)
{
    if ("auto" == name)
    {
        m_projection = Projection::Auto;
    }
    else if ("parallel" == name)
    {
        m_projection = Projection::Parallel;
    }
    else if ("perspective" == name)
    {
        m_projection = Projection::Perspective;
    }
}

std::string RScene::projectionName() const
{
    switch (m_projection)
    {
    case Projection::Parallel:
        return "parallel";
    case Projection::Perspective:
        return "perspective";
    default:
        return "auto";
    }
}

void RScene::setViewPreset(std::string const & name, float aspect)
{
    if (!m_has_bbox)
    {
        return;
    }

    // The preset direction points from the target toward the eye; the up axis
    // is +z for the side views and +y when looking down or up the z axis.
    QVector3D dir;
    QVector3D up(0.0f, 0.0f, 1.0f);
    if ("front" == name || "-y" == name)
    {
        dir = QVector3D(0.0f, -1.0f, 0.0f);
    }
    else if ("back" == name || "+y" == name)
    {
        dir = QVector3D(0.0f, 1.0f, 0.0f);
    }
    else if ("right" == name || "+x" == name)
    {
        dir = QVector3D(1.0f, 0.0f, 0.0f);
    }
    else if ("left" == name || "-x" == name)
    {
        dir = QVector3D(-1.0f, 0.0f, 0.0f);
    }
    else if ("top" == name || "+z" == name)
    {
        dir = QVector3D(0.0f, 0.0f, 1.0f);
        up = QVector3D(0.0f, 1.0f, 0.0f);
    }
    else if ("bottom" == name || "-z" == name)
    {
        dir = QVector3D(0.0f, 0.0f, -1.0f);
        up = QVector3D(0.0f, 1.0f, 0.0f);
    }
    else if ("iso" == name || "isometric" == name)
    {
        dir = QVector3D(1.0f, 1.0f, 1.0f);
    }
    else
    {
        return; // Ignore an unknown preset.
    }

    QVector3D const center = (m_bbox_lo + m_bbox_hi) * 0.5f;
    float radius = boundingRadius();
    if (radius <= 0.0f)
    {
        radius = 1.0f;
    }
    float const safe_aspect = (aspect > 0.0f) ? aspect : 1.0f;
    float const deg2rad = 3.14159265358979323846f / 180.0f;
    float const half_v = FOV_DEGREES * 0.5f * deg2rad;
    float const half_h = std::atan(std::tan(half_v) * safe_aspect);
    float const distance = radius / std::tan(std::min(half_v, half_h)) * 1.1f;

    m_camera.setTarget(center);
    m_camera.setPosition(center + dir.normalized() * distance);
    m_camera.setUp(up.normalized());
}

QMatrix4x4 RScene::viewProjection(QSize pixel_size, QRhi * rhi) const
{
    QMatrix4x4 clip = (nullptr != rhi) ? rhi->clipSpaceCorrMatrix() : QMatrix4x4();
    if (!m_has_bbox || pixel_size.height() <= 0 || pixel_size.width() <= 0)
    {
        return clip;
    }

    float const aspect = static_cast<float>(pixel_size.width()) / static_cast<float>(pixel_size.height());
    float const radius = framingRadius();

    QMatrix4x4 const view = m_camera.viewMatrix();
    float distance = (m_camera.position() - m_camera.target()).length();
    if (distance <= 0.0f)
    {
        distance = 2.0f * radius;
    }

    bool const use_perspective =
        (Projection::Perspective == m_projection) ||
        (Projection::Auto == m_projection && 3 == m_ndim);

    QMatrix4x4 proj;
    if (use_perspective)
    {
        proj.perspective(FOV_DEGREES, aspect, 0.01f * radius, distance + 3.0f * radius);
    }
    else
    {
        // The orthographic box is sized around the bounding sphere for the
        // viewport aspect, scaled by the camera's zoom factor.
        float const margin = radius * 1.1f * m_camera.orthoScale();
        float half_w = margin;
        float half_h = margin;
        if (aspect >= 1.0f)
        {
            half_w = margin * aspect;
        }
        else
        {
            half_h = margin / aspect;
        }
        proj.ortho(-half_w, half_w, -half_h, half_h, 0.01f * radius, distance + 3.0f * radius);
    }

    return clip * proj * view;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
