/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RFeatureEdges.hpp> // Must be the first include.

#include <solvcon/pilot/common/render_misc.hpp>

#include <array>

namespace solvcon
{

namespace
{

// A bold orange, distinct from the black wireframe, the white background, and
// the per-set boundary palette.
constexpr std::array<float, 3> FEATURE_COLOR{0.95f, 0.45f, 0.05f};

} /* end namespace */

RFeatureEdges::RFeatureEdges(std::shared_ptr<StaticMesh> const & mesh)
{
    build(*mesh);
}

void RFeatureEdges::build(StaticMesh const & mh)
{
    // Every boundary face, across all sets, contributes its rim edges.
    SimpleCollector<uint32_t> ends;
    SimpleArray<int32_t> const & bndfcs = mh.bndfcs();
    for (size_t ibnd = 0; ibnd < bndfcs.shape(0); ++ibnd)
    {
        append_face_edges(mh, bndfcs(ibnd, 0), ends);
    }

    append_edge_ribbons(mh, ends, FEATURE_COLOR, m_interleaved, m_indices);
    setColor(QVector4D(FEATURE_COLOR[0], FEATURE_COLOR[1], FEATURE_COLOR[2], 1.0f));
}

QRhiVertexInputLayout RFeatureEdges::vertexInputLayout() const
{
    QRhiVertexInputLayout layout;
    layout.setBindings({{6 * sizeof(float)}});
    layout.setAttributes({
        {0, 0, QRhiVertexInputAttribute::Float3, 0},
        {0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)},
    });
    return layout;
}

void RFeatureEdges::createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch)
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
