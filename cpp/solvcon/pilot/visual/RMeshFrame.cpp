/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/visual/RMeshFrame.hpp> // Must be the first include.

namespace solvcon
{

RMeshFrame::RMeshFrame(std::shared_ptr<StaticMesh> const & mesh, Style style)
    : m_style(style)
{
    StaticMesh const & mh = *mesh;
    if (Style::Surface == m_style)
    {
        buildSurface(mh);
        // A muted steel-blue surface: clearly a solid, and distinct from both
        // the white background and the black wireframe it can pair with.
        setColor(QVector4D(0.60f, 0.70f, 0.85f, 1.0f));
    }
    else
    {
        buildFrame(mh);
        // A black hairline/point over the white background.
        setColor(QVector4D(0.0f, 0.0f, 0.0f, 1.0f));
    }
}

void RMeshFrame::buildFrame(StaticMesh const & mh)
{
    uint32_t const nnode = mh.nnode();
    uint32_t const ndim = mh.ndim();

    m_vertices.reserve(static_cast<size_t>(nnode) * 3);
    for (uint32_t ind = 0; ind < nnode; ++ind)
    {
        m_vertices.push_back(static_cast<float>(mh.ndcrd(ind, 0)));
        m_vertices.push_back(static_cast<float>(mh.ndcrd(ind, 1)));
        m_vertices.push_back((3 == ndim) ? static_cast<float>(mh.ndcrd(ind, 2)) : 0.0f);
    }

    // The point cloud draws the nodes directly; only the wireframe needs edges.
    if (Style::Wireframe == m_style)
    {
        uint32_t const nedge = mh.nedge();
        m_indices.reserve(static_cast<size_t>(nedge) * 2);
        for (uint32_t ie = 0; ie < nedge; ++ie)
        {
            m_indices.push_back(static_cast<uint32_t>(mh.ednds(ie, 0)));
            m_indices.push_back(static_cast<uint32_t>(mh.ednds(ie, 1)));
        }
    }
}

void RMeshFrame::buildSurface(StaticMesh const & mh)
{
    uint32_t const ndim = mh.ndim();

    auto node_coord = [&mh, ndim](int32_t ind, uint32_t dim) -> float
    { return (dim < ndim) ? static_cast<float>(mh.ndcrd(ind, dim)) : 0.0f; };

    // Fan-triangulate a polygon of @p nnd nodes (indexed through @p node),
    // emitting flat-shaded triangles that all carry @p normal.
    auto add_polygon =
        [this, &node_coord](auto node, int32_t nnd, float const normal[3])
    {
        for (int32_t k = 1; k + 1 < nnd; ++k)
        {
            int32_t const tri[3] = {node(0), node(k), node(k + 1)};
            uint32_t const base = static_cast<uint32_t>(m_vertices.size() / 6);
            for (int32_t const ind : tri)
            {
                m_vertices.push_back(node_coord(ind, 0));
                m_vertices.push_back(node_coord(ind, 1));
                m_vertices.push_back(node_coord(ind, 2));
                m_vertices.push_back(normal[0]);
                m_vertices.push_back(normal[1]);
                m_vertices.push_back(normal[2]);
            }
            m_indices.push_back(base + 0);
            m_indices.push_back(base + 1);
            m_indices.push_back(base + 2);
        }
    };

    if (3 == ndim)
    {
        // The visible surface of a 3D domain is its boundary shell; each
        // boundary face carries an outward normal from the metric.
        SimpleArray<int32_t> const & bndfcs = mh.bndfcs();
        for (size_t ibnd = 0; ibnd < bndfcs.shape(0); ++ibnd)
        {
            int32_t const ifc = bndfcs(ibnd, 0);
            int32_t const nnd = mh.fcnds(ifc, 0);
            float const normal[3] = {
                static_cast<float>(mh.fcnml(ifc, 0)),
                static_cast<float>(mh.fcnml(ifc, 1)),
                static_cast<float>(mh.fcnml(ifc, 2))};
            add_polygon(
                [&mh, ifc](int32_t k)
                { return mh.fcnds(ifc, k + 1); },
                nnd,
                normal);
        }
    }
    else
    {
        // A 2D domain is a flat sheet of cells in the z = 0 plane, facing +z.
        float const normal[3] = {0.0f, 0.0f, 1.0f};
        for (uint32_t icl = 0; icl < mh.ncell(); ++icl)
        {
            int32_t const nnd = mh.clnds(icl, 0);
            add_polygon(
                [&mh, icl](int32_t k)
                { return mh.clnds(icl, k + 1); },
                nnd,
                normal);
        }
    }
}

QRhiGraphicsPipeline::Topology RMeshFrame::topology() const
{
    switch (m_style)
    {
    case Style::Points:
        return QRhiGraphicsPipeline::Points;
    case Style::Surface:
        return QRhiGraphicsPipeline::Triangles;
    default:
        return QRhiGraphicsPipeline::Lines;
    }
}

QRhiVertexInputLayout RMeshFrame::vertexInputLayout() const
{
    QRhiVertexInputLayout layout;
    if (Style::Surface == m_style)
    {
        layout.setBindings({{6 * sizeof(float)}});
        layout.setAttributes({
            {0, 0, QRhiVertexInputAttribute::Float3, 0},
            {0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)},
        });
    }
    else
    {
        layout.setBindings({{3 * sizeof(float)}});
        layout.setAttributes({{0, 0, QRhiVertexInputAttribute::Float3, 0}});
    }
    return layout;
}

void RMeshFrame::createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch)
{
    if (0 == m_vertices.size())
    {
        return;
    }

    // The surface interleaves a normal after each position; the frame stores
    // bare positions.
    size_t const stride = (Style::Surface == m_style) ? 6 : 3;
    quint32 const vbytes = static_cast<quint32>(m_vertices.size() * sizeof(float));
    m_vbuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, vbytes));
    m_vbuf->create();
    batch->uploadStaticBuffer(m_vbuf.get(), m_vertices.data());
    m_vertex_count = static_cast<quint32>(m_vertices.size() / stride);

    // The point cloud is a non-indexed draw over every node; the wireframe and
    // the surface index into their vertices.
    if (0 == m_indices.size())
    {
        return;
    }
    quint32 const ibytes = static_cast<quint32>(m_indices.size() * sizeof(uint32_t));
    m_ibuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::IndexBuffer, ibytes));
    m_ibuf->create();
    batch->uploadStaticBuffer(m_ibuf.get(), m_indices.data());
    m_index_count = static_cast<quint32>(m_indices.size());
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
