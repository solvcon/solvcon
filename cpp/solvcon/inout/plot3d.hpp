#pragma once

/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Reader that turns Plot3D structured grid data into a solvcon mesh.
 *
 * @ingroup group_inout
 */

#include <queue>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <solvcon/base.hpp>
#include <solvcon/mesh/mesh.hpp>
#include <solvcon/buffer/buffer.hpp>
#include <solvcon/inout/inout_util.hpp>

namespace solvcon
{

namespace inout
{

/**
 * Parser that converts Plot3D structured grid data into a StaticMesh.
 *
 * The grid is read from an in-memory string holding one or more blocks.
 * Node coordinates are parsed per block, indexed by the per-block x, y,
 * and z shapes, and assembled into hexahedron elements to build the mesh.
 *
 * @ingroup group_inout
 */
class Plot3d
    : public NumberBase<int32_t, double>
{
    using number_base = NumberBase<int32_t, double>;
    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

public:
    explicit Plot3d(const std::string & data);

    ~Plot3d() = default;

    Plot3d() = delete;
    Plot3d(Plot3d const & other) = delete;
    Plot3d(Plot3d && other) = delete;
    Plot3d & operator=(Plot3d const & other) = delete;
    Plot3d & operator=(Plot3d && other) = delete;

    std::shared_ptr<StaticMesh> to_block();
    void build_interior(const std::shared_ptr<StaticMesh> & blk);

private:
    inline uint_type coordinate_to_node_id(uint_type blk, uint_type x, uint_type y, uint_type z)
    {
        return blk > 0 ? m_blk_sizes(blk - 1) : 0 + x + y * m_x_shape(blk) + z * m_x_shape(blk) * m_y_shape(blk);
    }

    void parseCoordinates(uint_type nblocks);
    void buildHexahedronElements(uint_type nblocks);

    std::stringstream stream;

    SimpleArray<real_type> m_nds;
    SimpleArray<uint_type> m_x_shape;
    SimpleArray<uint_type> m_y_shape;
    SimpleArray<uint_type> m_z_shape;
    SimpleArray<uint_type> m_blk_sizes;

    std::unordered_map<uint_type, small_vector<uint_type>> m_elems;
}; /* end class Plot3d */

} /* end namespace inout */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
