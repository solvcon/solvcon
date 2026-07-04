#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The camera controller for the pilot domain viewer, with pan-zoom,
 * first-person, and orbit interaction modes.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <QMatrix4x4>
#include <QVector3D>

#include <string>

namespace solvcon
{

/**
 * @brief A single camera controller with three interaction modes.
 *
 * PanZoom frames a 2D domain head-on: dragging pans and the wheel zooms an
 * orthographic view. FirstPerson flies through a 3D domain: dragging looks
 * around and the wheel dollies along the view direction. Orbit inspects a 3D
 * domain from outside: dragging swings the eye around the fixed target and the
 * wheel dollies toward it, the up axis held fixed so the view never rolls (a
 * turntable orbit). The controller holds one pose (position, target, up) plus
 * an orthographic zoom factor, and feeds a QMatrix4x4 view matrix.
 *
 * The widget drives the controller from mouse and key events, and Python
 * drives the same primitives (rotate / zoom / pan) and reads or sets the pose
 * directly, so the domain navigates the same way from code as from the mouse.
 *
 * @ingroup group_domain
 */
class RCameraController
{

public:

    /**
     * The interaction mode that selects how drag, wheel, and key input
     * move the camera.
     *
     * @ingroup group_domain
     */
    enum class Mode
    {
        PanZoom, ///< 2D: drag pans, wheel zooms the orthographic box.
        FirstPerson, ///< 3D: drag looks around, wheel dollies forward.
        Orbit, ///< 3D: drag swings the eye around the target, wheel dollies in.
    };

    void setMode(Mode mode) { m_mode = mode; }
    Mode mode() const { return m_mode; }

    static Mode modeFromName(std::string const & name);
    static std::string modeName(Mode mode);

    /**
     * The orbit style. Turntable holds the up axis fixed so the horizon never
     * rolls; Trackball tumbles freely, rolling the up axis with the drag.
     *
     * @ingroup group_domain
     */
    enum class OrbitStyle
    {
        Turntable,
        Trackball,
    };

    void setOrbitStyle(OrbitStyle style) { m_orbit_style = style; }
    OrbitStyle orbitStyle() const { return m_orbit_style; }
    static OrbitStyle orbitStyleFromName(std::string const & name);
    static std::string orbitStyleName(OrbitStyle style);

    /// Set the orbit pivot: the point the Orbit mode swings the eye around.
    /// The eye stays put, so the next drag rotates about the new center.
    void setPivot(QVector3D const & pivot) { m_target = pivot; }

    /// Scale the rotate/look speed. A positive factor only; ignored otherwise.
    void setOrbitSensitivity(float factor) { m_orbit_sensitivity = (factor > 0.0f) ? factor : m_orbit_sensitivity; }
    float orbitSensitivity() const { return m_orbit_sensitivity; }

    /// Orbit by a fixed number of degrees, a discrete step independent of the
    /// per-pixel drag scaling (what a numpad-style fixed rotation drives).
    void orbitStep(float yaw_deg, float pitch_deg) { orbitBy(yaw_deg, pitch_deg); }

    /// Frame the camera onto the bounding box [lo, hi] for an @p ndim domain
    /// at the given viewport @p aspect (width / height).
    void fitToBoundingBox(QVector3D const & lo, QVector3D const & hi, uint32_t ndim, float aspect);

    /// The view matrix for the current pose.
    QMatrix4x4 viewMatrix() const;

    /// The orthographic zoom factor (smaller means zoomed in).
    float orthoScale() const { return m_ortho_scale; }

    /// Drag interaction: pan in PanZoom, look around in FirstPerson, swing the
    /// eye around the target in Orbit. @p dx and @p dy are pixel deltas.
    void rotate(float dx, float dy);

    /// Pan the view in its own plane by the pixel deltas (both modes).
    void pan(float dx, float dy);

    /// Wheel interaction: change the orthographic zoom in PanZoom, dolly along
    /// the view direction in FirstPerson, dolly toward the target in Orbit.
    /// @p steps is the wheel notch count.
    void zoom(float steps);

    /// Pinch interaction: zoom by a multiplicative @p factor (greater than 1
    /// zooms in, less than 1 zooms out), as a trackpad or touch pinch reports.
    /// Maps onto the same zoom path as the wheel.
    void pinch(float factor);

    /// Move along the view direction (forward, +) or sideways (strafe, +right)
    /// by a fraction of the scene size; used by the first-person keys.
    void moveForward(float amount);
    void moveRight(float amount);

    QVector3D position() const { return m_position; }
    void setPosition(QVector3D const & position) { m_position = position; }
    QVector3D target() const { return m_target; }
    void setTarget(QVector3D const & target) { m_target = target; }
    QVector3D up() const { return m_up; }
    void setUp(QVector3D const & up) { m_up = up; }

private:

    QVector3D forward() const;
    QVector3D rightAxis() const;
    /// Rotate the eye around the target by explicit degrees, honoring the
    /// turntable/trackball style. Shared by the drag rotate and the discrete
    /// step.
    void orbitBy(float yaw_deg, float pitch_deg);

    Mode m_mode = Mode::Orbit; ///< Default; a 2D domain switches to PanZoom.
    OrbitStyle m_orbit_style = OrbitStyle::Turntable; ///< Default orbit style.
    float m_orbit_sensitivity = 1.0f; ///< Rotate/look speed multiplier.

    QVector3D m_position{0.0f, 0.0f, 1.0f};
    QVector3D m_target{0.0f, 0.0f, 0.0f};
    QVector3D m_up{0.0f, 1.0f, 0.0f};

    float m_ortho_scale = 1.0f; ///< PanZoom zoom factor.
    float m_radius = 1.0f; ///< Scene scale, sets pan and dolly speeds.

}; /* end class RCameraController */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
