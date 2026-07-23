#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * A drawable field of per-vertex-colored triangles for the pilot viewer.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/visual/RDrawable.hpp>

#include <solvcon/solvcon.hpp>

#include <QVector3D>

namespace solvcon
{

/**
 * @brief A field drawn as per-vertex-colored triangles.
 *
 * Takes a vertex table (nvert, 3), a matching color table (nvert, 3) in the
 * [0, 1] range, and a triangle index table (ntri, 3). The colors are uploaded
 * interleaved with the positions and read by the per-vertex-color material,
 * so the field is swappable at runtime by replacing the drawable.
 *
 * @ingroup group_domain
 */
class RField
    : public RDrawable
{

public:

    RField(
        SimpleArray<float> const & vertices,
        SimpleArray<float> const & colors,
        SimpleArray<uint32_t> const & indices);

    bool hasGeometry() const { return m_indices.size() > 0; }

    QVector3D bboxLo() const { return m_lo; }
    QVector3D bboxHi() const { return m_hi; }

protected:

    RMaterial::Kind materialKind() const override { return RMaterial::Kind::VertexColor; }

    QRhiGraphicsPipeline::Topology topology() const override { return QRhiGraphicsPipeline::Triangles; }

    QRhiVertexInputLayout vertexInputLayout() const override;

    void createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch) override;

private:

    /*
     * The geometry is kept as one interleaved [x, y, z, r, g, b] vertex buffer
     * plus a shared-vertex index buffer, the layout QRhi vertex input reads
     * directly, so createGeometry() uploads it with no repack. This is
     * deliberately not the universe TrianglePad: that container stores corners
     * de-indexed in a structure-of-arrays layout and carries no color, so it
     * cannot express the per-vertex color or the vertex sharing kept here, and
     * it would force a gather into this buffer on every upload while pulling the
     * geometry/boolean-ops module into the renderer. A TrianglePad is adapted
     * into the (vertices, colors, indices) constructor arrays instead.
     */
    SimpleCollector<float> m_interleaved;
    SimpleCollector<uint32_t> m_indices;

    QVector3D m_lo;
    QVector3D m_hi;

}; /* end class RField */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
