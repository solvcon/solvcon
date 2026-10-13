/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/mesh/DelaunayBuilder.hpp>

#include <algorithm>
#include <format>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace solvcon
{

/**
 * Append the triangle nodes (ia, ib, ic) whose circumcircle has the squared
 * radius rr and the center (ux, uy, uz). For points in the xy plane, uz
 * is zero by default.
 */
void DelaunayBuilder::add(int_type ia, int_type ib, int_type ic, real_type rr, real_type ux, real_type uy, real_type uz)
{
    m_vertex.push_back(ia);
    m_vertex.push_back(ib);
    m_vertex.push_back(ic);
    m_circle.push_back(ux);
    m_circle.push_back(uy);
    m_circle.push_back(uz);
    m_circle.push_back(rr);
}

/**
 * Remove every triangle whose circumcircle strictly contains the point
 * (px, py, pz), and return the removed triangles in their stored order. A
 * point within the relative tolerance of the circle counts as
 * outside, so cocircular points do not make the result depend on
 * rounding.
 */
std::vector<DelaunayBuilder::triangle_type> DelaunayBuilder::remove_containing(real_type px, real_type py, real_type pz)
{
    std::vector<triangle_type> removed;
    size_t const ntri = size();
    size_t kept = 0;
    // One pass: collect the bad triangles and compact the kept ones in place.
    for (size_t it = 0; it < ntri; ++it)
    {
        real_type const dx = px - m_circle[4 * it];
        real_type const dy = py - m_circle[4 * it + 1];
        real_type const dz = pz - m_circle[4 * it + 2];
        real_type const rr = m_circle[4 * it + 3];
        // The shrunk radius makes a point on the circle count as outside.
        if (dx * dx + dy * dy + dz * dz < rr * (1.0 - m_relative_tolerance))
        {
            removed.push_back({m_vertex[3 * it], m_vertex[3 * it + 1], m_vertex[3 * it + 2]});
            continue;
        }
        if (kept != it)
        {
            for (size_t ic = 0; ic < 3; ++ic)
            {
                m_vertex[3 * kept + ic] = m_vertex[3 * it + ic];
            }
            for (size_t ic = 0; ic < 4; ++ic)
            {
                m_circle[4 * kept + ic] = m_circle[4 * it + ic];
            }
        }
        ++kept;
    }
    for (size_t it = kept; it < ntri; ++it)
    {
        for (size_t ic = 0; ic < 3; ++ic)
        {
            m_vertex.pop_back();
        }
        for (size_t ic = 0; ic < 4; ++ic)
        {
            m_circle.pop_back();
        }
    }
    return removed;
}

/**
 * Insert the points first to last-1 of the pad one at a time, the
 * Bowyer-Watson step. For each point, remove the triangles whose
 * circumcircle strictly contains it (see remove_containing), then fan the
 * rim of the hole to the point. The rim is the removed directed edges
 * whose reverse was not removed. They run counter-clockwise, so the new
 * triangles are counter-clockwise too. The new triangles follow the order
 * of the removed ones.
 *
 * The set must hold a triangle that contains every inserted point, for
 * example a super triangle.
 */
void DelaunayBuilder::populate_from_points(PointPad<real_type> const & pad, size_t first, size_t last)
{
    if (first > last || last > pad.size())
    {
        throw std::out_of_range(std::format("DelaunayBuilder::populate_from_points: range [{}, {}) out of range of {}", first, last, pad.size()));
    }
    auto const key = [](int_type a, int_type b)
    { return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b); };
    for (size_t ip = first; ip < last; ++ip)
    {
        // Carve out the triangles whose circumcircle contains the point.
        std::vector<triangle_type> const bad = remove_containing(pad.x_at(ip), pad.y_at(ip), pad.ndim() == 3 ? pad.z_at(ip) : 0.0);

        // Collect the directed edges of the hole. A shared edge shows up in
        // both directions, so an edge without its reverse lies on the rim.
        std::vector<std::pair<int_type, int_type>> edges;
        std::unordered_set<uint64_t> directed;
        edges.reserve(3 * bad.size());
        for (triangle_type const & tri : bad)
        {
            for (size_t ie = 0; ie < 3; ++ie)
            {
                int_type const ea = tri[ie];
                int_type const eb = tri[(ie + 1) % 3];
                edges.emplace_back(ea, eb);
                directed.insert(key(ea, eb));
            }
        }

        // Fan the rim to the point. The rim runs counter-clockwise around the
        // hole, so each new triangle is counter-clockwise.
        for (auto const & [ea, eb] : edges)
        {
            if (directed.count(key(eb, ea)) == 0)
            {
                auto const [rr, ux, uy, uz] = pad.circumcircle(ea, eb, ip);
                add(ea, eb, static_cast<int_type>(ip), rr, ux, uy, uz);
            }
        }
    }
}

/**
 * The vertex indices of the triangles whose vertices are all smaller than
 * nnode, in creation order, as an n-by-3 array. This drops the triangles
 * that touch the super triangle. Each triangle is counter-clockwise in
 * the xy plane.
 */
SimpleArray<DelaunayBuilder::int_type> DelaunayBuilder::compute_triangle_nodes(PointPad<real_type> const & pad, size_t nnode) const
{
    // A triangle is dropped when it uses a vertex outside the first nnode points.
    auto const inside = [&](size_t it)
    {
        return static_cast<size_t>(std::max({m_vertex[3 * it], m_vertex[3 * it + 1], m_vertex[3 * it + 2]})) < nnode;
    };
    size_t nkept = 0;
    for (size_t it = 0; it < size(); ++it)
    {
        nkept += inside(it) ? 1 : 0;
    }

    SimpleArray<int_type> ret(small_vector<ssize_t>{static_cast<ssize_t>(nkept), 3});
    size_t ik = 0;
    for (size_t it = 0; it < size(); ++it)
    {
        if (!inside(it))
        {
            continue;
        }
        int_type const ia = m_vertex[3 * it];
        int_type ib = m_vertex[3 * it + 1];
        int_type ic = m_vertex[3 * it + 2];
        // Guard against a clockwise triangle by swapping two vertices.
        real_type const ax = pad.x_at(ia);
        real_type const ay = pad.y_at(ia);
        real_type const area = (pad.x_at(ib) - ax) * (pad.y_at(ic) - ay) - (pad.x_at(ic) - ax) * (pad.y_at(ib) - ay);
        if (area < 0.0)
        {
            std::swap(ib, ic);
        }
        ret(ik, 0) = ia;
        ret(ik, 1) = ib;
        ret(ik, 2) = ic;
        ++ik;
    }
    return ret;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
