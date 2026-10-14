#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Incremental Delaunay triangulation by the Bowyer-Watson algorithm.
 *
 * @ingroup group_mesh
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/buffer.hpp>
#include <solvcon/universe/coord.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace solvcon
{

/**
 * Builds a Delaunay triangulation by inserting points one at a time. The
 * triangles keep their creation order, so the triangle numbering of a mesh
 * built from them does not depend on how the bad triangles were found. Each
 * triangle is stored with its circumcircle, which the point test reuses.
 *
 * @see A. Bowyer, "Computing Dirichlet tessellations", The Computer Journal
 *      24(2):162-166, 1981. https://doi.org/10.1093/comjnl/24.2.162
 * @see D. F. Watson, "Computing the n-dimensional Delaunay tessellation with
 *      application to Voronoi polytopes", The Computer Journal 24(2):167-172,
 *      1981. https://doi.org/10.1093/comjnl/24.2.167
 * @see https://en.wikipedia.org/wiki/Bowyer%E2%80%93Watson_algorithm
 */
class DelaunayBuilder
{

public:

    using int_type = int32_t;
    using real_type = double;
    using triangle_type = std::array<int_type, 3>;

    DelaunayBuilder() = default;

    size_t size() const { return m_vertex.size() / 3; }

    real_type relative_tolerance() const { return m_relative_tolerance; }
    void set_relative_tolerance(real_type value) { m_relative_tolerance = value; }

    void add(int_type ia, int_type ib, int_type ic, real_type rr, real_type ux, real_type uy, real_type uz = 0.0);

    void populate_from_points(PointPad<real_type> const & pad, size_t first, size_t last);

    SimpleArray<int_type> compute_triangle_nodes(PointPad<real_type> const & pad, size_t nnode) const;

    explicit DelaunayBuilder(std::shared_ptr<PointPad<real_type>> pad);

    std::shared_ptr<PointPad<real_type>> const & pad() const { return m_pad; }

    SimpleArray<int_type> triangulate();

    SimpleArray<real_type> calc_ear_centroids(SimpleArray<int_type> const & tris) const;

private:

    std::vector<triangle_type> remove_containing(real_type px, real_type py, real_type pz = 0.0);

    std::shared_ptr<PointPad<real_type>> m_pad; ///< The points to triangulate, when bound.
    SimpleCollector<int_type> m_vertex; ///< Three vertex indices per triangle.
    SimpleCollector<real_type> m_circle; ///< Center x, y, z and squared radius per triangle.
    real_type m_relative_tolerance = 1.e-12; ///< Shrinks every circle so that a point on it counts as outside.

}; /* end class DelaunayBuilder */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
