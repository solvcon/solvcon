#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Draw a short arrow at every face center along the face normal.
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
 * @brief The face normals drawn as short line arrows, one per face.
 *
 * Each arrow starts at the face center (StaticMesh::fccnd) and points along
 * the unit face normal (StaticMesh::fcnml); its length is a small fraction of
 * the mesh extent, so the field reads as short quills. A two-segment head
 * marks the direction. Both are precomputed by StaticMesh, so no geometry is
 * derived here beyond scaling.
 *
 * @ingroup group_domain
 */
class RNormals
    : public RDrawable
{

public:

    explicit RNormals(std::shared_ptr<StaticMesh> const & mesh);

    bool hasGeometry() const { return m_positions.size() > 0; }

protected:

    RMaterial::Kind materialKind() const override { return RMaterial::Kind::FlatColor; }

    QRhiGraphicsPipeline::Topology topology() const override { return QRhiGraphicsPipeline::Lines; }

    QRhiVertexInputLayout vertexInputLayout() const override;

    void createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch) override;

private:

    void build(StaticMesh const & mh);

    SimpleCollector<float> m_positions; ///< Line-segment endpoints (x, y, z).

}; /* end class RNormals */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
