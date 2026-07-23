#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * A drawable field of scalar-valued triangles colored through a GPU lookup
 * table.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/visual/RColormap.hpp>
#include <solvcon/pilot/visual/RDrawable.hpp>

#include <solvcon/solvcon.hpp>

#include <QVector3D>

namespace solvcon
{

/**
 * @brief A field of triangles colored by a per-vertex scalar on the GPU.
 *
 * Takes a vertex table (nvert, 3), a matching scalar table (nvert,), and a
 * triangle index table (ntri, 3). The scalar is uploaded interleaved with the
 * positions as a vertex attribute; the fragment stage normalizes it by the
 * mapping range and samples the colormap LUT texture, so recoloring by range
 * or by map never touches the vertex data.
 *
 * The mapping range rides the color slot of the shared uniform block as
 * (vmin, 1/(vmax - vmin)); see shaders/scalar.frag.
 *
 * @ingroup group_domain
 */
class RScalarField
    : public RDrawable
{

public:

    RScalarField(
        SimpleArray<float> const & vertices,
        SimpleArray<float> const & scalars,
        SimpleArray<uint32_t> const & indices,
        RColormap colormap);

    bool hasGeometry() const { return m_indices.size() > 0; }

    QVector3D bboxLo() const { return m_lo; }
    QVector3D bboxHi() const { return m_hi; }

    RColormap const & colormap() const { return m_colormap; }
    /// Swap the lookup table; takes effect on the next frame.
    void setColormap(RColormap colormap);

    /// Set the value-to-[0, 1] mapping range for the LUT sampling.
    void setScalarRange(float lo, float hi);
    float rangeLo() const { return m_range_lo; }
    float rangeHi() const { return m_range_hi; }

    void updateUniform(QRhiResourceUpdateBatch * batch, QMatrix4x4 const & view_proj) override;

    void release() override;

protected:

    RMaterial::Kind materialKind() const override { return RMaterial::Kind::ScalarColor; }

    QRhiGraphicsPipeline::Topology topology() const override { return QRhiGraphicsPipeline::Triangles; }

    QRhiVertexInputLayout vertexInputLayout() const override;

    void createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch) override;

    void extendBindings(
        QRhi * rhi,
        QRhiResourceUpdateBatch * batch,
        std::vector<QRhiShaderResourceBinding> & bindings) override;

private:

    static constexpr int LUT_WIDTH = 256;

    /// Pack (vmin, 1/span) into the color slot of the shared uniform block.
    void packRange();

    RColormap m_colormap;

    // Interleaved [x, y, z, s] per vertex, captured at construction.
    SimpleCollector<float> m_interleaved;
    SimpleCollector<uint32_t> m_indices;

    QVector3D m_lo;
    QVector3D m_hi;

    float m_range_lo = 0.0f;
    float m_range_hi = 1.0f;
    bool m_lut_dirty = false;

    std::unique_ptr<QRhiTexture> m_lut;
    std::unique_ptr<QRhiSampler> m_sampler;

}; /* end class RScalarField */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
