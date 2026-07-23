/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/visual/RScalarField.hpp> // Must be the first include.

#include <limits>
#include <stdexcept>
#include <utility>

namespace solvcon
{

RScalarField::RScalarField(
    SimpleArray<float> const & vertices,
    SimpleArray<float> const & scalars,
    SimpleArray<uint32_t> const & indices,
    RColormap colormap)
    : m_colormap(std::move(colormap))
{
    // Require (nvert, 3) vertices, a matching (nvert,) scalar table, and
    // (ntri, 3) triangle indices; mismatches would feed malformed buffers.
    if (vertices.ndim() != 2 || vertices.shape(1) != 3)
    {
        throw std::invalid_argument("RScalarField: vertices must have shape (nvert, 3)");
    }
    if (scalars.ndim() != 1 || scalars.shape(0) != vertices.shape(0))
    {
        throw std::invalid_argument("RScalarField: scalars must have shape (nvert,) matching vertices");
    }
    if (indices.ndim() != 2 || indices.shape(1) != 3)
    {
        throw std::invalid_argument("RScalarField: indices must have shape (ntri, 3)");
    }

    size_t const nvert = vertices.shape(0);
    size_t const ntri = indices.shape(0);
    if (0 == nvert || 0 == ntri)
    {
        return;
    }

    float lo[3] = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    float hi[3] = {
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    float smin = std::numeric_limits<float>::max();
    float smax = std::numeric_limits<float>::lowest();

    m_interleaved.reserve(nvert * 4);
    for (size_t i = 0; i < nvert; ++i)
    {
        for (size_t d = 0; d < 3; ++d)
        {
            float const v = vertices(i, d);
            m_interleaved.push_back(v);
            lo[d] = std::min(lo[d], v);
            hi[d] = std::max(hi[d], v);
        }
        float const s = scalars(i);
        m_interleaved.push_back(s);
        smin = std::min(smin, s);
        smax = std::max(smax, s);
    }
    m_lo = QVector3D(lo[0], lo[1], lo[2]);
    m_hi = QVector3D(hi[0], hi[1], hi[2]);

    m_indices.reserve(ntri * 3);
    for (size_t i = 0; i < ntri; ++i)
    {
        for (size_t k = 0; k < 3; ++k)
        {
            uint32_t const idx = indices(i, k);
            if (idx >= nvert)
            {
                throw std::invalid_argument("RScalarField: triangle index out of range [0, nvert)");
            }
            m_indices.push_back(idx);
        }
    }

    m_range_lo = smin;
    m_range_hi = smax;
    packRange();
}

void RScalarField::setColormap(RColormap colormap)
{
    m_colormap = std::move(colormap);
    m_lut_dirty = true;
}

void RScalarField::setScalarRange(float lo, float hi)
{
    if (hi < lo)
    {
        throw std::invalid_argument("RScalarField: scalar range must have hi >= lo");
    }
    m_range_lo = lo;
    m_range_hi = hi;
    packRange();
}

void RScalarField::packRange()
{
    float const span = m_range_hi - m_range_lo;
    m_color = QVector4D(m_range_lo, (span > 0.0f) ? 1.0f / span : 0.0f, 0.0f, 0.0f);
}

QRhiVertexInputLayout RScalarField::vertexInputLayout() const
{
    QRhiVertexInputLayout layout;
    layout.setBindings({{4 * sizeof(float)}});
    layout.setAttributes({
        {0, 0, QRhiVertexInputAttribute::Float3, 0},
        {0, 1, QRhiVertexInputAttribute::Float, 3 * sizeof(float)},
    });
    return layout;
}

void RScalarField::createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch)
{
    if (0 == m_interleaved.size() || 0 == m_indices.size())
    {
        return;
    }

    quint32 const vbytes = static_cast<quint32>(m_interleaved.size() * sizeof(float));
    m_vbuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, vbytes));
    m_vbuf->create();
    batch->uploadStaticBuffer(m_vbuf.get(), m_interleaved.data());
    m_vertex_count = static_cast<quint32>(m_interleaved.size() / 4);

    quint32 const ibytes = static_cast<quint32>(m_indices.size() * sizeof(uint32_t));
    m_ibuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::IndexBuffer, ibytes));
    m_ibuf->create();
    batch->uploadStaticBuffer(m_ibuf.get(), m_indices.data());
    m_index_count = static_cast<quint32>(m_indices.size());
}

void RScalarField::extendBindings(
    QRhi * rhi, QRhiResourceUpdateBatch * batch, std::vector<QRhiShaderResourceBinding> & bindings)
{
    m_sampler.reset(rhi->newSampler(
        QRhiSampler::Linear,
        QRhiSampler::Linear,
        QRhiSampler::None,
        QRhiSampler::ClampToEdge,
        QRhiSampler::ClampToEdge));
    m_sampler->create();

    m_lut.reset(rhi->newTexture(QRhiTexture::RGBA8, QSize(LUT_WIDTH, 1)));
    m_lut->create();
    batch->uploadTexture(m_lut.get(), m_colormap.image(LUT_WIDTH));
    m_lut_dirty = false;

    bindings.push_back(QRhiShaderResourceBinding::sampledTexture(
        1, QRhiShaderResourceBinding::FragmentStage, m_lut.get(), m_sampler.get()));
}

void RScalarField::updateUniform(QRhiResourceUpdateBatch * batch, QMatrix4x4 const & view_proj)
{
    RDrawable::updateUniform(batch, view_proj);
    if (m_lut_dirty && m_lut)
    {
        batch->uploadTexture(m_lut.get(), m_colormap.image(LUT_WIDTH));
        m_lut_dirty = false;
    }
}

void RScalarField::release()
{
    m_lut.reset();
    m_sampler.reset();
    RDrawable::release();
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
