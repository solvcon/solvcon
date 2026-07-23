#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Draw the domain's boundary (feature) edges as one bold colored overlay.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/visual/RDrawable.hpp>

#include <solvcon/solvcon.hpp>

#include <memory>

namespace solvcon
{

/**
 * @brief The domain's boundary (feature) edges drawn as one bold colored
 * overlay over the hairline wireframe.
 *
 * The edges are the rim edges of every boundary face, gathered across all
 * boundary sets and widened into flat ribbons in a single color, so the whole
 * domain outline reads as one distinct layer. This complements the per-set
 * highlight (RBoundary), which colors one set at a time.
 *
 * @ingroup group_domain
 */
class RFeatureEdges
    : public RDrawable
{

public:

    explicit RFeatureEdges(std::shared_ptr<StaticMesh> const & mesh);

    bool hasGeometry() const { return m_indices.size() > 0; }

protected:

    RMaterial::Kind materialKind() const override { return RMaterial::Kind::VertexColor; }

    QRhiGraphicsPipeline::Topology topology() const override { return QRhiGraphicsPipeline::Triangles; }

    QRhiVertexInputLayout vertexInputLayout() const override;

    void createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch) override;

private:

    void build(StaticMesh const & mh);

    // Interleaved [x, y, z, r, g, b] per ribbon vertex.
    SimpleCollector<float> m_interleaved;
    SimpleCollector<uint32_t> m_indices;

}; /* end class RFeatureEdges */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
