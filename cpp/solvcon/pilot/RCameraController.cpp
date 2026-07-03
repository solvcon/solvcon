/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RCameraController.hpp> // Must be the first include.

#include <QQuaternion>
#include <QtMath>

#include <algorithm>
#include <cmath>

namespace solvcon
{

namespace
{

constexpr float FOV_DEGREES = 45.0f; // Matches RScene's perspective FOV.
constexpr float LOOK_DEGREES_PER_PIXEL = 0.3f;
constexpr float PAN_PIXELS_PER_EXTENT = 300.0f;
constexpr float ZOOM_PER_STEP = 0.12f;
constexpr float ORTHO_SCALE_MIN = 0.02f;
constexpr float ORTHO_SCALE_MAX = 50.0f;
constexpr float ORBIT_DISTANCE_MIN = 0.05f; // Times the scene radius.
constexpr float ORBIT_DISTANCE_MAX = 20.0f; // Times the scene radius.

} /* end namespace */

RCameraController::Mode RCameraController::modeFromName(std::string const & name)
{
    if ("fps" == name)
    {
        return Mode::FirstPerson;
    }
    if ("orbit" == name)
    {
        return Mode::Orbit;
    }
    return Mode::PanZoom;
}

std::string RCameraController::modeName(Mode mode)
{
    switch (mode)
    {
    case Mode::FirstPerson:
        return "fps";
    case Mode::Orbit:
        return "orbit";
    default:
        return "pan";
    }
}

RCameraController::OrbitStyle RCameraController::orbitStyleFromName(std::string const & name)
{
    if ("trackball" == name)
    {
        return OrbitStyle::Trackball;
    }
    return OrbitStyle::Turntable;
}

std::string RCameraController::orbitStyleName(OrbitStyle style)
{
    return (OrbitStyle::Trackball == style) ? "trackball" : "turntable";
}

QVector3D RCameraController::forward() const
{
    QVector3D const dir = m_target - m_position;
    float const length = dir.length();
    return (length > 0.0f) ? dir / length : QVector3D(0.0f, 0.0f, -1.0f);
}

QVector3D RCameraController::rightAxis() const
{
    QVector3D const axis = QVector3D::crossProduct(forward(), m_up);
    float const length = axis.length();
    return (length > 0.0f) ? axis / length : QVector3D(1.0f, 0.0f, 0.0f);
}

void RCameraController::fitToBoundingBox(
    QVector3D const & lo, QVector3D const & hi, uint32_t ndim, float aspect)
{
    QVector3D const center = (lo + hi) * 0.5f;
    float radius = (hi - lo).length() * 0.5f;
    if (radius <= 0.0f)
    {
        radius = 1.0f;
    }
    m_radius = radius;
    m_ortho_scale = 1.0f;
    m_up = QVector3D(0.0f, 1.0f, 0.0f);
    m_target = center;

    if (3 == ndim)
    {
        // Pull back so the bounding sphere fits the tighter of the vertical
        // and horizontal fields of view for this viewport.
        float const safe_aspect = (aspect > 0.0f) ? aspect : 1.0f;
        float const half_v = qDegreesToRadians(FOV_DEGREES) * 0.5f;
        float const half_h = std::atan(std::tan(half_v) * safe_aspect);
        float const distance = radius / std::tan(std::min(half_v, half_h)) * 1.1f;
        QVector3D const dir = QVector3D(0.6f, 0.5f, 1.0f).normalized();
        m_position = center + dir * distance;
    }
    else
    {
        // 2D: head-on, ahead along +z. The distance is immaterial to the
        // orthographic projection.
        m_position = center + QVector3D(0.0f, 0.0f, 2.0f * radius);
    }
}

QMatrix4x4 RCameraController::viewMatrix() const
{
    QMatrix4x4 view;
    view.lookAt(m_position, m_target, m_up);
    return view;
}

void RCameraController::orbitBy(float yaw_deg, float pitch_deg)
{
    if (OrbitStyle::Trackball == m_orbit_style)
    {
        // Tumble freely: yaw about the current up axis and pitch about the
        // right axis, then roll the up axis with the same rotation so the
        // horizon is free to tilt. No pole guard, so the eye can pass over the
        // top and keep going, the way a virtual-trackball drag does.
        QVector3D const offset = m_position - m_target;
        QQuaternion const yaw = QQuaternion::fromAxisAndAngle(m_up, yaw_deg);
        QQuaternion const pitch =
            QQuaternion::fromAxisAndAngle(rightAxis(), pitch_deg);
        QQuaternion const rot = yaw * pitch;
        m_position = m_target + rot.rotatedVector(offset);
        m_up = rot.rotatedVector(m_up).normalized();
        return;
    }

    // Turntable: swing the eye around the fixed target, up axis held so the
    // horizon never rolls.
    QVector3D const offset = m_position - m_target;
    QQuaternion const yaw = QQuaternion::fromAxisAndAngle(m_up, yaw_deg);
    QQuaternion const pitch = QQuaternion::fromAxisAndAngle(rightAxis(), pitch_deg);
    QVector3D rotated = (yaw * pitch).rotatedVector(offset);
    // Avoid gimbal lock: do not let the eye-to-target direction reach the up
    // axis (which would degenerate the right axis and flip the view). Drop the
    // pitch if it would push past the pole, but always ease away.
    QVector3D const up = m_up.normalized();
    float const limit = std::cos(qDegreesToRadians(1.0f));
    float const aligned = std::abs(QVector3D::dotProduct(rotated.normalized(), up));
    float const aligned0 = std::abs(QVector3D::dotProduct(offset.normalized(), up));
    if (aligned > limit && aligned > aligned0)
    {
        rotated = yaw.rotatedVector(offset);
    }
    m_position = m_target + rotated;
}

void RCameraController::rotate(float dx, float dy)
{
    float const scale = LOOK_DEGREES_PER_PIXEL * m_orbit_sensitivity;
    if (Mode::Orbit == m_mode)
    {
        orbitBy(-dx * scale, -dy * scale);
    }
    else if (Mode::FirstPerson == m_mode)
    {
        // Look around in place: yaw about the up axis, pitch about the right
        // axis, keeping the eye fixed and swinging the target.
        QVector3D const f0 = forward();
        QQuaternion const yaw = QQuaternion::fromAxisAndAngle(m_up, -dx * scale);
        QQuaternion const pitch = QQuaternion::fromAxisAndAngle(rightAxis(), -dy * scale);
        QVector3D dir = (yaw * pitch).rotatedVector(f0).normalized();
        // Avoid gimbal lock: do not let the view direction reach the up axis
        // (which would degenerate the right axis and flip the view). Drop the
        // pitch if it would push past the pole, but always allow easing away.
        QVector3D const up = m_up.normalized();
        float const limit = std::cos(qDegreesToRadians(1.0f));
        float const aligned = std::abs(QVector3D::dotProduct(dir, up));
        if (aligned > limit && aligned > std::abs(QVector3D::dotProduct(f0, up)))
        {
            dir = yaw.rotatedVector(f0).normalized();
        }
        float const distance = (m_target - m_position).length();
        m_target = m_position + dir * distance;
    }
    else
    {
        pan(dx, dy);
    }
}

void RCameraController::pan(float dx, float dy)
{
    // A drag across the viewport pans roughly the visible extent, moving in
    // the camera's own plane (the screen right and screen up axes). Screen up
    // is the world up head-on, so the 2D pan is unchanged.
    float const extent = 2.0f * m_radius * m_ortho_scale;
    float const sx = -dx / PAN_PIXELS_PER_EXTENT * extent;
    float const sy = dy / PAN_PIXELS_PER_EXTENT * extent;
    QVector3D const screen_up = QVector3D::crossProduct(rightAxis(), forward()).normalized();
    QVector3D const offset = rightAxis() * sx + screen_up * sy;
    m_position += offset;
    m_target += offset;
}

void RCameraController::zoom(float steps)
{
    if (Mode::Orbit == m_mode)
    {
        // Dolly the eye along its offset to the target: a positive step moves
        // it closer, a negative one farther. The scale is multiplicative so it
        // feels uniform at any range, and the distance is clamped so the eye
        // never reaches or crosses the pivot, nor flies arbitrarily far.
        QVector3D const offset = m_position - m_target;
        float const distance = offset.length();
        if (distance > 0.0f)
        {
            float const scaled = std::clamp(
                distance * std::exp(-steps * ZOOM_PER_STEP),
                ORBIT_DISTANCE_MIN * m_radius,
                ORBIT_DISTANCE_MAX * m_radius);
            m_position = m_target + offset * (scaled / distance);
        }
    }
    else if (Mode::FirstPerson == m_mode)
    {
        moveForward(steps * 0.1f);
    }
    else
    {
        m_ortho_scale = std::clamp(
            m_ortho_scale * std::exp(-steps * ZOOM_PER_STEP),
            ORTHO_SCALE_MIN,
            ORTHO_SCALE_MAX);
    }
}

void RCameraController::pinch(float factor)
{
    // A pinch reports a multiplicative scale; route it through the wheel zoom
    // so both share one path. zoom() applies exp(-steps * ZOOM_PER_STEP), so
    // steps = log(factor) / ZOOM_PER_STEP makes the zoom track the pinch
    // exactly: a 2x spread zooms in 2x. A non-positive factor is ignored.
    if (factor > 0.0f)
    {
        zoom(std::log(factor) / ZOOM_PER_STEP);
    }
}

void RCameraController::moveForward(float amount)
{
    QVector3D const step = forward() * (amount * m_radius);
    m_position += step;
    m_target += step;
}

void RCameraController::moveRight(float amount)
{
    QVector3D const step = rightAxis() * (amount * m_radius);
    m_position += step;
    m_target += step;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
