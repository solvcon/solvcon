/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RNormals.hpp> // Must be the first include.

#include <QVector3D>

#include <algorithm>
#include <cmath>
#include <limits>

namespace solvcon
{

namespace
{

// A saturated green, distinct from the black wireframe, the white background,
// and the orange feature-edge overlay.
constexpr float NORMAL_COLOR[3] = {0.15f, 0.65f, 0.20f};

// Arrow shaft length as a fraction of the mesh diagonal, plus the head size as
// a fraction of the shaft.
constexpr float SHAFT_FRACTION = 0.04f;
constexpr float HEAD_LENGTH = 0.32f;
constexpr float HEAD_WIDTH = 0.18f;

// An in-plane unit vector perpendicular to @p n, so the arrowhead of a 2D
// mesh (drawn top-down in z = 0) stays visible; a 3D mesh takes any
// perpendicular.
QVector3D head_perpendicular(QVector3D const & n, uint32_t ndim)
{
    QVector3D u = (2 == ndim)
                      ? QVector3D(-n.y(), n.x(), 0.0f)
                      : QVector3D::crossProduct(n, (std::abs(n.x()) < 0.9f) ? QVector3D(1.0f, 0.0f, 0.0f) : QVector3D(0.0f, 1.0f, 0.0f));
    float const len = u.length();
    return (len > std::numeric_limits<float>::epsilon()) ? u / len : QVector3D(1.0f, 0.0f, 0.0f);
}

} /* end namespace */

RNormals::RNormals(std::shared_ptr<StaticMesh> const & mesh)
{
    build(*mesh);
    setColor(QVector4D(NORMAL_COLOR[0], NORMAL_COLOR[1], NORMAL_COLOR[2], 1.0f));
}

void RNormals::build(StaticMesh const & mh)
{
    uint32_t const ndim = mh.ndim();
    uint32_t const nface = mh.nface();
    if (0 == nface)
    {
        return;
    }

    // Size the arrows from the mesh diagonal so they read short at any scale.
    QVector3D lo(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    QVector3D hi(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest());
    for (uint32_t ind = 0; ind < mh.nnode(); ++ind)
    {
        float const z = (3 == ndim) ? static_cast<float>(mh.ndcrd(ind, 2)) : 0.0f;
        QVector3D const p(static_cast<float>(mh.ndcrd(ind, 0)), static_cast<float>(mh.ndcrd(ind, 1)), z);
        lo = QVector3D(std::min(lo.x(), p.x()), std::min(lo.y(), p.y()), std::min(lo.z(), p.z()));
        hi = QVector3D(std::max(hi.x(), p.x()), std::max(hi.y(), p.y()), std::max(hi.z(), p.z()));
    }
    float const diag = (hi - lo).length();
    float const shaft = (diag > 0.0f ? diag : 1.0f) * SHAFT_FRACTION;

    auto push = [this](QVector3D const & a, QVector3D const & b)
    {
        m_positions.push_back(a.x());
        m_positions.push_back(a.y());
        m_positions.push_back(a.z());
        m_positions.push_back(b.x());
        m_positions.push_back(b.y());
        m_positions.push_back(b.z());
    };

    m_positions.reserve(static_cast<size_t>(nface) * 6 * 3);
    for (uint32_t ifc = 0; ifc < nface; ++ifc)
    {
        float const cz = (3 == ndim) ? static_cast<float>(mh.fccnd(ifc, 2)) : 0.0f;
        QVector3D const center(
            static_cast<float>(mh.fccnd(ifc, 0)), static_cast<float>(mh.fccnd(ifc, 1)), cz);
        float const nz = (3 == ndim) ? static_cast<float>(mh.fcnml(ifc, 2)) : 0.0f;
        QVector3D normal(
            static_cast<float>(mh.fcnml(ifc, 0)), static_cast<float>(mh.fcnml(ifc, 1)), nz);
        if (normal.lengthSquared() <= std::numeric_limits<float>::epsilon())
        {
            continue; // A degenerate face has no direction to point.
        }
        normal.normalize();

        QVector3D const tip = center + normal * shaft;
        QVector3D const u = head_perpendicular(normal, ndim);
        QVector3D const back = tip - normal * (shaft * HEAD_LENGTH);
        QVector3D const barb = u * (shaft * HEAD_WIDTH);
        push(center, tip); // shaft
        push(tip, back + barb); // head, one side
        push(tip, back - barb); // head, other side
    }
}

QRhiVertexInputLayout RNormals::vertexInputLayout() const
{
    QRhiVertexInputLayout layout;
    layout.setBindings({{3 * sizeof(float)}});
    layout.setAttributes({{0, 0, QRhiVertexInputAttribute::Float3, 0}});
    return layout;
}

void RNormals::createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch)
{
    if (0 == m_positions.size())
    {
        return;
    }

    quint32 const vbytes = static_cast<quint32>(m_positions.size() * sizeof(float));
    m_vbuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, vbytes));
    m_vbuf->create();
    batch->uploadStaticBuffer(m_vbuf.get(), m_positions.data());
    m_vertex_count = static_cast<quint32>(m_positions.size() / 3);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
