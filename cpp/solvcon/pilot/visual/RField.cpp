/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/visual/RField.hpp> // Must be the first include.

#include <limits>
#include <stdexcept>

namespace solvcon
{

RField::RField(
    SimpleArray<float> const & vertices,
    SimpleArray<float> const & colors,
    SimpleArray<uint32_t> const & indices)
{
    // Require (nvert, 3) vertices, a matching (nvert, 3) color table, and
    // (ntri, 3) triangle indices; mismatches would feed malformed buffers.
    if (vertices.ndim() != 2 || vertices.shape(1) != 3)
    {
        throw std::invalid_argument("RField: vertices must have shape (nvert, 3)");
    }
    if (colors.ndim() != 2 || colors.shape(0) != vertices.shape(0) || colors.shape(1) != 3)
    {
        throw std::invalid_argument("RField: colors must have shape (nvert, 3) matching vertices");
    }
    if (indices.ndim() != 2 || indices.shape(1) != 3)
    {
        throw std::invalid_argument("RField: indices must have shape (ntri, 3)");
    }

    size_t const nvert = static_cast<size_t>(vertices.shape(0));
    size_t const ntri = static_cast<size_t>(indices.shape(0));
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

    m_interleaved.reserve(nvert * 6);
    for (size_t i = 0; i < nvert; ++i)
    {
        for (size_t d = 0; d < 3; ++d)
        {
            float const v = vertices(i, d);
            m_interleaved.push_back(v);
            lo[d] = std::min(lo[d], v);
            hi[d] = std::max(hi[d], v);
        }
        for (size_t d = 0; d < 3; ++d)
        {
            m_interleaved.push_back(colors(i, d));
        }
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
                throw std::invalid_argument("RField: triangle index out of range [0, nvert)");
            }
            m_indices.push_back(idx);
        }
    }
}

QRhiVertexInputLayout RField::vertexInputLayout() const
{
    QRhiVertexInputLayout layout;
    layout.setBindings({{6 * sizeof(float)}});
    layout.setAttributes({
        {0, 0, QRhiVertexInputAttribute::Float3, 0},
        {0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)},
    });
    return layout;
}

void RField::createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch)
{
    if (0 == m_interleaved.size() || 0 == m_indices.size())
    {
        return;
    }

    quint32 const vbytes = static_cast<quint32>(m_interleaved.size() * sizeof(float));
    m_vbuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, vbytes));
    m_vbuf->create();
    batch->uploadStaticBuffer(m_vbuf.get(), m_interleaved.data());
    m_vertex_count = static_cast<quint32>(m_interleaved.size() / 6);

    quint32 const ibytes = static_cast<quint32>(m_indices.size() * sizeof(uint32_t));
    m_ibuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::IndexBuffer, ibytes));
    m_ibuf->create();
    batch->uploadStaticBuffer(m_ibuf.get(), m_indices.data());
    m_index_count = static_cast<quint32>(m_indices.size());
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
