#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Renders the unstructured-mesh domain as a wireframe, a point cloud, or a
 * lit surface.
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
 * @brief The unstructured-mesh domain rendered as an edge wireframe, a node
 * point cloud, or a lit, shaded surface.
 *
 * The wireframe draws the mesh edge list (StaticMesh::ednds) as lines and the
 * point cloud draws the nodes themselves, both over the shared node-coordinate
 * buffer and flat-colored. The surface instead fills the domain: a 2D mesh
 * fills its cells in the z = 0 plane facing +z, a 3D mesh draws its boundary
 * shell with each face's outward normal, every face fan-triangulated into
 * flat-shaded triangles that the lit material shades two-sided over an ambient
 * floor. Works for both 2D and 3D meshes.
 *
 * @ingroup group_domain
 */
class RMeshFrame
    : public RDrawable
{

public:

    /// Which primitive the geometry is assembled into.
    enum class Style
    {
        Wireframe, ///< Mesh edges as lines.
        Points, ///< Mesh nodes as points.
        Surface, ///< Cell or boundary faces as a lit, shaded surface.
    };

    explicit RMeshFrame(
        std::shared_ptr<StaticMesh> const & mesh, Style style = Style::Wireframe);

protected:

    RMaterial::Kind materialKind() const override
    {
        return (Style::Surface == m_style) ? RMaterial::Kind::Lit
                                           : RMaterial::Kind::FlatColor;
    }

    QRhiGraphicsPipeline::Topology topology() const override;

    QRhiVertexInputLayout vertexInputLayout() const override;

    void createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch) override;

    // Sink the surface a touch in depth so a wireframe or point cloud sharing
    // its plane draws on top instead of z-fighting into it.
    int depthBias() const override { return (Style::Surface == m_style) ? 2 : 0; }
    float slopeScaledDepthBias() const override
    {
        return (Style::Surface == m_style) ? 1.0f : 0.0f;
    }

private:

    /// Node positions plus the edge index list, for the wireframe and points.
    void buildFrame(StaticMesh const & mh);
    /// Fan-triangulated, per-vertex-normal triangles, for the lit surface.
    void buildSurface(StaticMesh const & mh);

    Style m_style;

    // CPU-side geometry captured at construction; the rhi is not available
    // until prepare() runs, so the buffers are uploaded then. m_vertices holds
    // 3 floats per vertex (x, y, z) for the wireframe and points, or 6
    // interleaved (x, y, z, nx, ny, nz) for the lit surface.
    SimpleCollector<float> m_vertices;
    SimpleCollector<uint32_t> m_indices;

}; /* end class RMeshFrame */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
