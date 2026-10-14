/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/mesh/DelaunayBuilder.hpp>

#include <algorithm>
#include <format>
#include <stdexcept>
#include <unordered_map>
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

/**
 * Bind the builder to the pad of points. The pad is shared, not copied, so
 * the Steiner points added later show up in it.
 */
DelaunayBuilder::DelaunayBuilder(std::shared_ptr<PointPad<real_type>> pad)
    : m_pad(std::move(pad))
{
    if (!m_pad)
    {
        throw std::invalid_argument("DelaunayBuilder::DelaunayBuilder: the pad is null");
    }
    if (m_pad->ndim() != 2)
    {
        throw std::invalid_argument(std::format("DelaunayBuilder::DelaunayBuilder: the pad is {}D but must be 2D", static_cast<int>(m_pad->ndim())));
    }
}

/**
 * Triangulate every point of the bound pad by the Bowyer-Watson algorithm,
 * dropping whatever triangles the builder held. A super triangle far outside
 * the bounding box seeds the set, and the triangles that touch it are
 * dropped at the end. The super triangle goes into a copy, so the bound pad
 * is not modified. A point on a circumcircle counts as outside, which keeps
 * the cavity well defined for the cocircular points of a regular layout. The
 * cost is O(n^2) for n points.
 */
SimpleArray<DelaunayBuilder::int_type> DelaunayBuilder::triangulate()
{
    if (!m_pad)
    {
        throw std::logic_error("DelaunayBuilder::triangulate: no pad is bound");
    }
    PointPad<real_type> const & pad = *m_pad;
    size_t const npt = pad.size();
    if (npt < 3)
    {
        throw std::invalid_argument(std::format("DelaunayBuilder::triangulate: {} points are fewer than 3", npt));
    }
    real_type xmin = pad.x_at(0);
    real_type xmax = xmin;
    real_type ymin = pad.y_at(0);
    real_type ymax = ymin;
    for (size_t ip = 1; ip < npt; ++ip)
    {
        xmin = std::min(xmin, pad.x_at(ip));
        xmax = std::max(xmax, pad.x_at(ip));
        ymin = std::min(ymin, pad.y_at(ip));
        ymax = std::max(ymax, pad.y_at(ip));
    }
    real_type const xmid = (xmin + xmax) / 2.0;
    real_type const ymid = (ymin + ymax) / 2.0;
    real_type const span = std::max({xmax - xmin, ymax - ymin, 1.0});

    auto const work = PointPad<real_type>::construct(2);
    for (size_t ip = 0; ip < npt; ++ip)
    {
        work->append(pad.x_at(ip), pad.y_at(ip));
    }
    work->append(xmid - 64.0 * span, ymid - 32.0 * span);
    work->append(xmid + 64.0 * span, ymid - 32.0 * span);
    work->append(xmid, ymid + 64.0 * span);

    m_vertex.clear();
    m_circle.clear();
    auto const [rr, ux, uy, uz] = work->circumcircle(npt, npt + 1, npt + 2);
    add(static_cast<int_type>(npt), static_cast<int_type>(npt + 1), static_cast<int_type>(npt + 2), rr, ux, uy, uz);
    populate_from_points(*work, 0, npt);
    return compute_triangle_nodes(*work, npt);
}

/**
 * The centroids of the ears, in triangle order, as an n-by-2 array. An ear is
 * a triangle that has two or more boundary edges. An edge is a boundary edge
 * when only one triangle uses it. Finite-volume solvers do not produce good
 * results in an ear.
 */
SimpleArray<DelaunayBuilder::real_type> DelaunayBuilder::calc_ear_centroids(SimpleArray<int_type> const & tris) const
{
    if (!m_pad)
    {
        throw std::logic_error("DelaunayBuilder::calc_ear_centroids: no pad is bound");
    }
    PointPad<real_type> const & pad = *m_pad;
    ssize_t const ntri = tris.shape(0);
    auto const key = [](int_type a, int_type b)
    {
        auto const lo = static_cast<uint32_t>(std::min(a, b));
        auto const hi = static_cast<uint32_t>(std::max(a, b));
        return (static_cast<uint64_t>(lo) << 32) | hi;
    };
    std::unordered_map<uint64_t, int_type> shared;
    shared.reserve(static_cast<size_t>(ntri) * 2);
    for (ssize_t it = 0; it < ntri; ++it)
    {
        for (ssize_t ie = 0; ie < 3; ++ie)
        {
            ++shared[key(tris(it, ie), tris(it, (ie + 1) % 3))];
        }
    }
    std::vector<ssize_t> ears;
    for (ssize_t it = 0; it < ntri; ++it)
    {
        int_type nbound = 0;
        for (ssize_t ie = 0; ie < 3; ++ie)
        {
            nbound += shared[key(tris(it, ie), tris(it, (ie + 1) % 3))] == 1 ? 1 : 0;
        }
        if (nbound >= 2)
        {
            ears.push_back(it);
        }
    }
    SimpleArray<real_type> ret(small_vector<ssize_t>{static_cast<ssize_t>(ears.size()), 2});
    for (size_t ie = 0; ie < ears.size(); ++ie)
    {
        real_type xc = 0.0;
        real_type yc = 0.0;
        for (ssize_t iv = 0; iv < 3; ++iv)
        {
            xc += pad.x_at(tris(ears[ie], iv));
            yc += pad.y_at(tris(ears[ie], iv));
        }
        ret(ie, 0) = xc / 3.0;
        ret(ie, 1) = yc / 3.0;
    }
    return ret;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
