/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/visual/RBoundary.hpp> // Must be the first include.

#include <solvcon/pilot/common/render_misc.hpp>

#include <array>

namespace solvcon
{

namespace
{

/// Pick a distinct, saturated color for a boundary set so neighboring sets
/// stay tellable apart; the palette repeats for meshes with many sets.
std::array<float, 3> boundary_color(int ibc)
{
    static const std::array<std::array<float, 3>, 6> palette{{
        {1.0f, 0.20f, 0.20f}, // red
        {0.20f, 0.60f, 1.0f}, // blue
        {0.20f, 0.80f, 0.30f}, // green
        {1.0f, 0.70f, 0.10f}, // amber
        {0.80f, 0.30f, 0.90f}, // purple
        {0.10f, 0.80f, 0.80f}, // teal
    }};
    return palette.at(static_cast<size_t>(ibc < 0 ? 0 : ibc) % palette.size());
}

} /* end namespace */

RBoundary::RBoundary(std::shared_ptr<StaticMesh> const & mesh, int ibc)
    : m_ibc(ibc)
{
    build(*mesh, ibc);
}

void RBoundary::build(StaticMesh const & mh, int ibc)
{
    // Gather the boundary-set edges as node-index pairs, then widen them into
    // the colored ribbon.
    SimpleCollector<uint32_t> ends;
    SimpleArray<int32_t> const & bndfcs = mh.bndfcs();
    for (size_t ibnd = 0; ibnd < static_cast<size_t>(bndfcs.shape(0)); ++ibnd)
    {
        if (bndfcs(ibnd, 1) == ibc)
        {
            append_face_edges(mh, bndfcs(ibnd, 0), ends);
        }
    }

    std::array<float, 3> const color = boundary_color(ibc);
    append_edge_ribbons(mh, ends, color, m_interleaved, m_indices);
    setColor(QVector4D(color[0], color[1], color[2], 1.0f));
}

QRhiVertexInputLayout RBoundary::vertexInputLayout() const
{
    QRhiVertexInputLayout layout;
    layout.setBindings({{6 * sizeof(float)}});
    layout.setAttributes({
        {0, 0, QRhiVertexInputAttribute::Float3, 0},
        {0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)},
    });
    return layout;
}

void RBoundary::createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch)
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
