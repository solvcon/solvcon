/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common/render_misc.hpp> // Must be the first include.

#include <algorithm>
#include <cmath>
#include <limits>

namespace solvcon
{

void append_face_edges(StaticMesh const & mh, int32_t ifc, SimpleCollector<uint32_t> & ends)
{
    int32_t const nnd = mh.fcnds(ifc, 0);
    for (int32_t ind = 1; ind <= nnd; ++ind)
    {
        ends.push_back(static_cast<uint32_t>(mh.fcnds(ifc, ind)));
        int32_t const next = (ind == nnd) ? 1 : ind + 1;
        ends.push_back(static_cast<uint32_t>(mh.fcnds(ifc, next)));
        if (2 == nnd)
        {
            break; // A 2D face is one edge; avoid drawing it twice.
        }
    }
}

void append_edge_ribbons(
    StaticMesh const & mh,
    SimpleCollector<uint32_t> const & ends,
    std::array<float, 3> const & color,
    SimpleCollector<float> & interleaved,
    SimpleCollector<uint32_t> & indices)
{
    size_t const nedge = ends.size() / 2;
    if (0 == nedge)
    {
        return;
    }

    auto node = [&mh](uint32_t ind, size_t dim) -> float
    { return (dim < mh.ndim()) ? static_cast<float>(mh.ndcrd(ind, dim)) : 0.0f; };

    // The ribbon is widened in the xy plane and lifted along +z, which is
    // exact for a z-planar (2D) boundary: the case the overlay targets. A
    // genuine out-of-plane 3D edge still draws but is not oriented to its face.
    //
    // The ribbon half-width is a small fraction of the mesh extent, so it reads
    // about twice the hairline wireframe at any zoom.
    float lo[3] = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    float hi[3] = {
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    for (uint32_t ind = 0; ind < mh.nnode(); ++ind)
    {
        for (size_t dim = 0; dim < 3; ++dim)
        {
            lo[dim] = std::min(lo[dim], node(ind, dim));
            hi[dim] = std::max(hi[dim], node(ind, dim));
        }
    }
    float const diag = std::sqrt(
        (hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) + (hi[2] - lo[2]) * (hi[2] - lo[2]));
    float const half = (diag > 0.0f ? diag : 1.0f) * 0.0035f;
    // Lift the ribbon toward the viewer (the meshes are 2D, so +z) so it wins
    // the depth test against the coplanar wireframe instead of z-fighting it.
    float const lift = half;

    interleaved.reserve(interleaved.size() + nedge * 4 * 6);
    indices.reserve(indices.size() + nedge * 6);
    for (size_t ie = 0; ie < nedge; ++ie)
    {
        uint32_t const i0 = ends[ie * 2];
        uint32_t const i1 = ends[ie * 2 + 1];
        float const dx = node(i1, 0) - node(i0, 0);
        float const dy = node(i1, 1) - node(i0, 1);
        float nx = dy;
        float ny = -dx;
        float len = std::sqrt(nx * nx + ny * ny);
        if (len < std::numeric_limits<float>::epsilon())
        {
            nx = 1.0f;
            ny = 0.0f;
            len = 1.0f;
        }
        float const ox = nx / len * half;
        float const oy = ny / len * half;
        // Corner order p0+o, p0-o, p1-o, p1+o makes the two triangles below.
        float const px[4] = {node(i0, 0) + ox, node(i0, 0) - ox, node(i1, 0) - ox, node(i1, 0) + ox};
        float const py[4] = {node(i0, 1) + oy, node(i0, 1) - oy, node(i1, 1) - oy, node(i1, 1) + oy};
        float const pz[4] = {node(i0, 2) + lift, node(i0, 2) + lift, node(i1, 2) + lift, node(i1, 2) + lift};
        uint32_t const base = static_cast<uint32_t>(interleaved.size() / 6);
        for (size_t ic = 0; ic < 4; ++ic)
        {
            interleaved.push_back(px[ic]);
            interleaved.push_back(py[ic]);
            interleaved.push_back(pz[ic]);
            interleaved.push_back(color[0]);
            interleaved.push_back(color[1]);
            interleaved.push_back(color[2]);
        }
        indices.push_back(base + 0);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 0);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
