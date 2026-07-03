#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * A flat-colored line-segment drawable, used for the measurement ruler.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/RDrawable.hpp>

#include <QVector3D>

#include <vector>

namespace solvcon
{

/**
 * @brief A set of independent line segments in world space, flat-colored.
 *
 * The constructor takes points in consecutive pairs (p0, p1, p2, p3, ...),
 * each pair drawn as one line. It backs the measurement ruler, which draws the
 * measured distance segment or the two angle arms over the scene.
 *
 * @ingroup group_domain
 */
class RSegments
    : public RDrawable
{

public:

    explicit RSegments(std::vector<QVector3D> const & points);

    bool hasGeometry() const { return m_vertices.size() >= 6; }

protected:

    RMaterial::Kind materialKind() const override { return RMaterial::Kind::FlatColor; }

    QRhiGraphicsPipeline::Topology topology() const override
    {
        return QRhiGraphicsPipeline::Lines;
    }

    QRhiVertexInputLayout vertexInputLayout() const override;

    void createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch) override;

private:

    std::vector<float> m_vertices; ///< Three floats (x, y, z) per vertex.

}; /* end class RSegments */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
