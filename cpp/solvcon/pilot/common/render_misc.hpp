#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Miscellaneous helpers for rendering 3D objects.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/solvcon.hpp>

#include <array>

namespace solvcon
{

/// Append the rim edges of face @p ifc as node-index pairs to @p ends. A 2D
/// face is a single edge; a 3D face is a polygon whose rim edges close the
/// loop back to the first node.
void append_face_edges(StaticMesh const & mh, int32_t ifc, SimpleCollector<uint32_t> & ends);

/// Widen the edges in @p ends (node-index pairs) into flat @p color quads
/// lifted toward the viewer, so they read about twice the hairline wireframe
/// and win the depth test against it. Appends interleaved [x, y, z, r, g, b]
/// vertices to @p interleaved and their triangle indices to @p indices.
void append_edge_ribbons(
    StaticMesh const & mh,
    SimpleCollector<uint32_t> const & ends,
    std::array<float, 3> const & color,
    SimpleCollector<float> & interleaved,
    SimpleCollector<uint32_t> & indices);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
