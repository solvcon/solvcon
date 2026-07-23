/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/visual/RSegments.hpp> // Must be the first include.

namespace solvcon
{

RSegments::RSegments(std::vector<QVector3D> const & points)
{
    m_vertices.reserve(points.size() * 3);
    for (QVector3D const & p : points)
    {
        m_vertices.push_back(p.x());
        m_vertices.push_back(p.y());
        m_vertices.push_back(p.z());
    }
    // A magenta ruler stands out over the mesh and the white background.
    setColor(QVector4D(0.85f, 0.10f, 0.65f, 1.0f));
}

QRhiVertexInputLayout RSegments::vertexInputLayout() const
{
    QRhiVertexInputLayout layout;
    layout.setBindings({{3 * sizeof(float)}});
    layout.setAttributes({{0, 0, QRhiVertexInputAttribute::Float3, 0}});
    return layout;
}

void RSegments::createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch)
{
    if (m_vertices.empty())
    {
        return;
    }
    quint32 const vbytes = static_cast<quint32>(m_vertices.size() * sizeof(float));
    m_vbuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, vbytes));
    m_vbuf->create();
    batch->uploadStaticBuffer(m_vbuf.get(), m_vertices.data());
    m_vertex_count = static_cast<quint32>(m_vertices.size() / 3);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
