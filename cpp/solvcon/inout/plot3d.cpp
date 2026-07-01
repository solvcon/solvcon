/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/inout/plot3d.hpp>

namespace solvcon
{

namespace inout
{

Plot3d::Plot3d(const std::string & data)
{
    std::string line;
    size_t total_blk_size = 0;
    uint_type nblocks = 0;
    const std::regex plot3d_block_delim(R"(\d+)");

    stream << data;
    // getting the first line for total blocks number of the p3d mesh file
    std::getline(stream, line);
    nblocks = stoul(line);

    m_x_shape.remake(small_vector<ssize_t>{static_cast<ssize_t>(nblocks)}, 0);
    m_y_shape.remake(small_vector<ssize_t>{static_cast<ssize_t>(nblocks)}, 0);
    m_z_shape.remake(small_vector<ssize_t>{static_cast<ssize_t>(nblocks)}, 0);
    m_blk_sizes.remake(small_vector<ssize_t>{static_cast<ssize_t>(nblocks)}, 0);

    // parsing xyz dimension of each block
    for (auto i = 0; i < nblocks; ++i)
    {
        std::getline(stream, line);
        auto tokens = tokenize(line, plot3d_block_delim);
        m_x_shape(i) = std::stoul(tokens[0]);
        m_y_shape(i) = std::stoul(tokens[1]);
        m_z_shape(i) = std::stoul(tokens[2]);
        m_blk_sizes(i) = m_x_shape(i) * m_y_shape(i) * m_z_shape(i);
        total_blk_size += m_blk_sizes(i);
    }

    m_nds.remake(small_vector<ssize_t>{static_cast<ssize_t>(total_blk_size), 3}, 0);

    parseCoordinates(nblocks);
    buildHexahedronElements(nblocks);
}

void Plot3d::parseCoordinates(const uint_type nblocks)
{
    uint_type base_idx = 0;
    std::string line;
    std::queue<real_type> parsing_q;
    const std::regex plot3d_coord_delim(R"(-?\d+(\.\d+)?([eE][+-]?\d+)?|\b\d+\b)");

    // TODO: The nested for-loop needs to be enhanced
    for (auto blk = 0; blk < nblocks; ++blk)
    {
        for (auto k = 0; k < 3; ++k)
        {
            for (auto j = 0; j < m_blk_sizes(blk); ++j)
            {
                if (parsing_q.empty())
                {
                    std::getline(stream, line);
                    // using regex to parsing the string and split them into float number
                    // also need to consider the scientific notation.
                    auto tokens = tokenize(line, plot3d_coord_delim);
                    for (auto const & token : tokens) { parsing_q.push(std::stod(token)); }
                }
                m_nds(j + base_idx, k) = parsing_q.front();
                parsing_q.pop();
            }
        }
        base_idx += m_blk_sizes(blk);
    }
}

void Plot3d::buildHexahedronElements(const uint_type nblocks)
{
    uint_type idx = 0;
    // hexahedron has 8 nodes + 1 for store nodes number
    small_vector<uint_type> nds_temp(9, 0U);
    // These vector of x, y, z coordinate shift is used to build the hexahedron element
    // the nodes order of the hexahedron element is clock of the xy face
    // Referece: https://github.com/nasa/Plot3D_utilities/blob/main/colab/Plot3D_SplitBlocksExample.ipynb
    const std::vector<std::vector<int_type>> shift = {
        {-1, -1, -1},
        {-1, 0, -1},
        {-1, 0, 0},
        {-1, -1, 0},
        {0, -1, -1},
        {0, 0, -1},
        {0, 0, 0},
        {0, -1, 0}};

    // TODO: The nested for-loop needs to be enhanced
    // build the hexahedron element from nodes
    for (auto blk = 0; blk < nblocks; ++blk)
    {
        for (auto k = 1; k < m_z_shape(blk); ++k)
        {
            for (auto j = 1; j < m_y_shape(blk); ++j)
            {
                for (auto i = 1; i < m_x_shape(blk); ++i)
                {
                    nds_temp[0] = 8;
                    for (auto ii = 1; ii <= 8; ++ii)
                    {
                        nds_temp[ii] = coordinate_to_node_id(blk, i + shift[ii - 1][0], j + shift[ii - 1][1], k + shift[ii - 1][2]);
                    }
                    m_elems.insert(std::pair{idx, nds_temp});
                    idx++;
                }
            }
        }
    }
}

std::shared_ptr<StaticMesh> Plot3d::to_block()
{
    std::shared_ptr<StaticMesh> block = StaticMesh::construct(
        3,
        static_cast<StaticMesh::uint_type>(m_nds.shape(0)),
        0,
        static_cast<StaticMesh::uint_type>(m_elems.size()));
    build_interior(block);
    return block;
}

void Plot3d::build_interior(const std::shared_ptr<StaticMesh> & blk)
{
    SimpleArray<int_type> m_cltpn;
    m_cltpn.remake(small_vector<ssize_t>{static_cast<ssize_t>(m_elems.size())}, 5);
    blk->cltpn().swap(m_cltpn);
    blk->ndcrd().swap(m_nds);

    for (size_t i = 0; i < m_elems.size(); ++i)
    {
        blk->clnds()(i, 0) = m_elems[i][0];
        for (size_t j = 1; j <= m_elems[i][0]; ++j)
        {
            blk->clnds()(i, j) = m_elems[i][j];
        }
    }
    blk->build_interior(true);
    blk->build_boundary();
    blk->build_ghost();
}

} // namespace inout

} // namespace solvcon

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
