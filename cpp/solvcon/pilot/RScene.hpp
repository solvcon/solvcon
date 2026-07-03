#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Defines the RScene class that owns the drawables, the domain bounding
 * box, and the framing camera for the pilot viewer.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/RCameraController.hpp>
#include <solvcon/pilot/RDrawable.hpp>

#include <rhi/qrhi.h>

#include <QMatrix4x4>
#include <QSize>
#include <QVector3D>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace solvcon
{

/**
 * @brief The renderable scene: the drawables, the domain bounding box, and
 * the framing camera with a per-dimension projection.
 *
 * The scene owns the list of RDrawable objects and the domain bounding box,
 * picks an orthographic projection for 2D domains and a perspective
 * projection for 3D ones, and frames the camera onto the box. RDomainWidget
 * holds one scene and drives it; the scene itself is free of Qt widget and
 * device-loop concerns.
 *
 * @ingroup group_domain
 */
class RScene
{

public:

    RScene() = default;

    RScene(RScene const &) = delete;
    RScene & operator=(RScene const &) = delete;

    /// Add a drawable; ownership transfers to the scene.
    void addDrawable(std::unique_ptr<RDrawable> drawable);

    /// Remove a specific drawable (no-op for nullptr or a foreign pointer).
    void removeDrawable(RDrawable const * drawable);

    /// Remove every drawable the predicate accepts.
    void removeDrawableIf(std::function<bool(RDrawable const *)> const & pred);

    std::vector<std::unique_ptr<RDrawable>> & drawables() { return m_drawables; }

    /// Release the device resources of every drawable.
    void releaseAll();

    /// Forget the bounding box so the next extendBoundingBox sets it afresh.
    void resetBoundingBox();

    /// Grow the bounding box to include the point range [lo, hi].
    void extendBoundingBox(QVector3D const & lo, QVector3D const & hi);

    bool hasBoundingBox() const { return m_has_bbox; }

    void setDimension(uint32_t ndim) { m_ndim = ndim; }
    uint32_t dimension() const { return m_ndim; }

    /// The projection selector. Auto picks orthographic for a 2D domain and
    /// perspective for a 3D one; the others force the choice.
    enum class Projection
    {
        Auto,
        Parallel,
        Perspective,
    };

    /// Override the projection: "auto", "parallel", or "perspective". An
    /// unknown name is ignored.
    void setProjection(std::string const & name);
    std::string projectionName() const;

    /// Point the camera along an axis-aligned or isometric preset and frame
    /// the scene: "front", "back", "left", "right", "top", "bottom", the axis
    /// names "+x".."-z", or "iso". An unknown name is ignored.
    void setViewPreset(std::string const & name, float aspect);

    RCameraController & camera() { return m_camera; }
    RCameraController const & camera() const { return m_camera; }

    /// Frame the camera so the whole bounding box is in view at the given
    /// viewport @p aspect (width / height).
    void fitCameraToScene(float aspect);

    /// The model-view-projection for the framed scene at the given viewport.
    /// 2D domains use an orthographic projection, 3D domains a perspective
    /// one. @p rhi supplies the backend clip-space correction.
    QMatrix4x4 viewProjection(QSize pixel_size, QRhi * rhi) const;

private:

    float boundingRadius() const;

    std::vector<std::unique_ptr<RDrawable>> m_drawables;

    QVector3D m_bbox_lo;
    QVector3D m_bbox_hi;
    bool m_has_bbox = false;
    uint32_t m_ndim = 0;
    Projection m_projection = Projection::Auto;

    RCameraController m_camera;

}; /* end class RScene */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
