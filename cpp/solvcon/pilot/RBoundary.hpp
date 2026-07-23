#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Draw a single mesh boundary set as thick colored ribbons over the
 * wireframe.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/RDrawable.hpp>

#include <solvcon/solvcon.hpp>

#include <memory>

namespace solvcon
{

/**
 * @brief A single boundary set drawn as thick colored ribbons over the
 * wireframe so the set stands out.
 *
 * Each boundary edge is widened into a flat quad (two triangles) lifted
 * slightly toward the viewer, because line width is clamped to one pixel in
 * the core profile. @ref ibc identifies which boundary set this draws.
 *
 * @ingroup group_domain
 */
class RBoundary
    : public RDrawable
{

public:

    RBoundary(std::shared_ptr<StaticMesh> const & mesh, int ibc);

    int ibc() const { return m_ibc; }

    bool hasGeometry() const { return m_indices.size() > 0; }

protected:

    RMaterial::Kind materialKind() const override { return RMaterial::Kind::VertexColor; }

    QRhiGraphicsPipeline::Topology topology() const override { return QRhiGraphicsPipeline::Triangles; }

    QRhiVertexInputLayout vertexInputLayout() const override;

    void createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch) override;

private:

    void build(StaticMesh const & mh, int ibc);

    int m_ibc = -1;

    // Interleaved [x, y, z, r, g, b] per ribbon vertex.
    SimpleCollector<float> m_interleaved;
    SimpleCollector<uint32_t> m_indices;

}; /* end class RBoundary */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
