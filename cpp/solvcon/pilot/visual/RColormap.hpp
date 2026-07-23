#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Named one-dimensional color lookup tables for scalar-field rendering.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <QImage>
#include <QVector3D>

#include <string>

namespace solvcon
{

/**
 * @brief A named one-dimensional color lookup table.
 *
 * A colormap is a piecewise-linear ramp over a static table of RGB anchor
 * points spaced uniformly on [0, 1]. It bakes into a one-row QImage that a
 * drawable uploads as the LUT texture sampled by the scalar-color material,
 * and the same table drives the scalar-bar strip so the on-screen report
 * always matches the mapping on the GPU.
 *
 * @ingroup group_domain
 */
class RColormap
{

public:

    /// The available names: "viridis", "coolwarm", "jet", "grayscale".
    /// Throws std::invalid_argument for anything else.
    static RColormap named(std::string const & name);

    /// A qualitative map for categorical coloring: a fixed set of visually
    /// distinct colors. Used with an integer scalar over [0, ncat - 1].
    static RColormap categorical();

    std::string const & name() const { return m_name; }

    /// Piecewise-linear sample of the ramp at @p t clamped to [0, 1].
    QVector3D sample(float t) const;

    /// Bake the ramp into a @p width x 1 RGBA image for the LUT texture.
    QImage image(int width = 256) const;

private:

    RColormap(std::string name, float const * anchors, size_t count);

    std::string m_name;
    float const * m_anchors; ///< Static table of RGB triples.
    size_t m_count;

}; /* end class RColormap */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
