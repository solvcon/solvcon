/*
 * Copyright (c) 2021, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/mesh/StaticMesh.hpp>

#include <set>

namespace solvcon
{

namespace detail
{

template <typename N>
struct FaceBuilder
    : public StaticMeshConstant
{

    using number_base = N;
    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

    size_t add_point(int_type icl, int_type ifc)
    {
        constexpr const size_t NFACE = 1;
        clfcs(icl, 0) = NFACE;
        for (size_t i = 0; i < NFACE; ++i)
        {
            fctpn(ifc + i) = CellType::NONCELLTYPE;
            fcnds(ifc + i, 0) = 0; // number of nodes per face.
        }
        // face 1.
        clfcs(icl, 1) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        return NFACE;
    }

    size_t add_line(int_type icl, int_type ifc)
    {
        constexpr const size_t NFACE = 2;
        // extract 2 points from a line.
        clfcs(icl, 0) = NFACE;
        for (size_t i = 0; i < NFACE; ++i)
        {
            fctpn(ifc + i) = CellType::POINT;
            fcnds(ifc + i, 0) = 1; // number of nodes per face.
        }
        // face 1.
        clfcs(icl, 1) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        ++ifc;
        // face 2.
        clfcs(icl, 2) = ifc;
        fcnds(ifc, 1) = clnds(icl, 2);
        return NFACE;
    }

    size_t add_quadrilateral(int_type icl, int_type ifc)
    {
        constexpr const size_t NFACE = 4;
        // extract 4 lines from a quadrilateral.
        clfcs(icl, 0) = NFACE;
        for (size_t i = 0; i < NFACE; ++i)
        {
            fctpn(ifc + i) = CellType::LINE;
            fcnds(ifc + i, 0) = 2; // number of nodes per face.
        }
        // face 1.
        clfcs(icl, 1) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        fcnds(ifc, 2) = clnds(icl, 2);
        ++ifc;
        // face 2.
        clfcs(icl, 2) = ifc;
        fcnds(ifc, 1) = clnds(icl, 2);
        fcnds(ifc, 2) = clnds(icl, 3);
        ++ifc;
        // face 3.
        clfcs(icl, 3) = ifc;
        fcnds(ifc, 1) = clnds(icl, 3);
        fcnds(ifc, 2) = clnds(icl, 4);
        ++ifc;
        // face 4.
        clfcs(icl, 4) = ifc;
        fcnds(ifc, 1) = clnds(icl, 4);
        fcnds(ifc, 2) = clnds(icl, 1);
        return NFACE;
    }

    size_t add_triangle(int_type icl, int_type ifc)
    {
        constexpr const size_t NFACE = 3;
        // extract 3 lines from a triangle.
        clfcs(icl, 0) = NFACE;
        for (size_t i = 0; i < NFACE; ++i)
        {
            fctpn(ifc + i) = CellType::LINE;
            fcnds(ifc + i, 0) = 2; // number of nodes per face.
        }
        // face 1.
        clfcs(icl, 1) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        fcnds(ifc, 2) = clnds(icl, 2);
        ++ifc;
        // face 2.
        clfcs(icl, 2) = ifc;
        fcnds(ifc, 1) = clnds(icl, 2);
        fcnds(ifc, 2) = clnds(icl, 3);
        ++ifc;
        // face 3.
        clfcs(icl, 3) = ifc;
        fcnds(ifc, 1) = clnds(icl, 3);
        fcnds(ifc, 2) = clnds(icl, 1);
        return NFACE;
    }

    size_t add_hexahedron(int_type icl, int_type ifc)
    {
        constexpr const size_t NFACE = 6;
        // extract 6 quadrilaterals from a hexahedron.
        clfcs(icl, 0) = NFACE;
        for (size_t i = 0; i < NFACE; ++i)
        {
            fctpn(ifc + i) = CellType::QUADRILATERAL;
            fcnds(ifc + i, 0) = 4; // number of nodes per face.
        }
        // face 1.
        clfcs(icl, 1) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        fcnds(ifc, 2) = clnds(icl, 4);
        fcnds(ifc, 3) = clnds(icl, 3);
        fcnds(ifc, 4) = clnds(icl, 2);
        ++ifc;
        // face 2.
        clfcs(icl, 2) = ifc;
        fcnds(ifc, 1) = clnds(icl, 2);
        fcnds(ifc, 2) = clnds(icl, 3);
        fcnds(ifc, 3) = clnds(icl, 7);
        fcnds(ifc, 4) = clnds(icl, 6);
        ++ifc;
        // face 3.
        clfcs(icl, 3) = ifc;
        fcnds(ifc, 1) = clnds(icl, 5);
        fcnds(ifc, 2) = clnds(icl, 6);
        fcnds(ifc, 3) = clnds(icl, 7);
        fcnds(ifc, 4) = clnds(icl, 8);
        ++ifc;
        // face 4.
        clfcs(icl, 4) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        fcnds(ifc, 2) = clnds(icl, 5);
        fcnds(ifc, 3) = clnds(icl, 8);
        fcnds(ifc, 4) = clnds(icl, 4);
        ++ifc;
        // face 5.
        clfcs(icl, 5) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        fcnds(ifc, 2) = clnds(icl, 2);
        fcnds(ifc, 3) = clnds(icl, 6);
        fcnds(ifc, 4) = clnds(icl, 5);
        ++ifc;
        // face 6.
        clfcs(icl, 6) = ifc;
        fcnds(ifc, 1) = clnds(icl, 3);
        fcnds(ifc, 2) = clnds(icl, 4);
        fcnds(ifc, 3) = clnds(icl, 8);
        fcnds(ifc, 4) = clnds(icl, 7);
        return NFACE;
    }

    size_t add_tetrahedron(int_type icl, int_type ifc)
    {
        constexpr const size_t NFACE = 4;
        // extract 4 triangles from a tetrahedron.
        clfcs(icl, 0) = NFACE;
        for (size_t i = 0; i < NFACE; ++i)
        {
            fctpn(ifc + i) = CellType::TRIANGLE;
            fcnds(ifc + i, 0) = 3; // number of nodes per face.
        }
        // face 1.
        clfcs(icl, 1) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        fcnds(ifc, 2) = clnds(icl, 3);
        fcnds(ifc, 3) = clnds(icl, 2);
        ++ifc;
        // face 2.
        clfcs(icl, 2) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        fcnds(ifc, 2) = clnds(icl, 2);
        fcnds(ifc, 3) = clnds(icl, 4);
        ++ifc;
        // face 3.
        clfcs(icl, 3) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        fcnds(ifc, 2) = clnds(icl, 4);
        fcnds(ifc, 3) = clnds(icl, 3);
        ++ifc;
        // face 4.
        clfcs(icl, 4) = ifc;
        fcnds(ifc, 1) = clnds(icl, 2);
        fcnds(ifc, 2) = clnds(icl, 3);
        fcnds(ifc, 3) = clnds(icl, 4);
        return NFACE;
    }

    size_t add_prism(int_type icl, int_type ifc)
    {
        constexpr const size_t NFACE = 5;
        // extract 2 triangles and 3 quadrilaterals from a prism.
        clfcs(icl, 0) = NFACE;
        for (size_t i = 0; i < 2; ++i)
        {
            fctpn(ifc + i) = CellType::TRIANGLE;
            fcnds(ifc + i, 0) = 3; // number of nodes per face.
        }
        for (size_t i = 2; i < NFACE; ++i)
        {
            fctpn(ifc + i) = CellType::QUADRILATERAL;
            fcnds(ifc + i, 0) = 4; // number of nodes per face.
        }
        // face 1.
        clfcs(icl, 1) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        fcnds(ifc, 2) = clnds(icl, 2);
        fcnds(ifc, 3) = clnds(icl, 3);
        ++ifc;
        // face 2.
        clfcs(icl, 2) = ifc;
        fcnds(ifc, 1) = clnds(icl, 4);
        fcnds(ifc, 2) = clnds(icl, 6);
        fcnds(ifc, 3) = clnds(icl, 5);
        ++ifc;
        // face 3.
        clfcs(icl, 3) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        fcnds(ifc, 2) = clnds(icl, 4);
        fcnds(ifc, 3) = clnds(icl, 5);
        fcnds(ifc, 4) = clnds(icl, 2);
        ++ifc;
        // face 4.
        clfcs(icl, 4) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        fcnds(ifc, 2) = clnds(icl, 3);
        fcnds(ifc, 3) = clnds(icl, 6);
        fcnds(ifc, 4) = clnds(icl, 4);
        ++ifc;
        // face 5.
        clfcs(icl, 5) = ifc;
        fcnds(ifc, 1) = clnds(icl, 2);
        fcnds(ifc, 2) = clnds(icl, 5);
        fcnds(ifc, 3) = clnds(icl, 6);
        fcnds(ifc, 4) = clnds(icl, 3);
        ++ifc;
        return NFACE;
    }

    size_t add_pyramid(int_type icl, int_type ifc)
    {
        constexpr const size_t NFACE = 5;
        // extract 4 triangles and 1 quadrilaterals from a pyramid.
        clfcs(icl, 0) = NFACE;
        for (size_t i = 0; i < 4; ++i)
        {
            fctpn(ifc + i) = CellType::TRIANGLE;
            fcnds(ifc + i, 0) = 3; // number of nodes per face.
        }
        for (size_t i = 4; i < NFACE; ++i)
        {
            fctpn(ifc + i) = CellType::QUADRILATERAL;
            fcnds(ifc + i, 0) = 4; // number of nodes per face.
        }
        // face 1.
        clfcs(icl, 1) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        fcnds(ifc, 2) = clnds(icl, 5);
        fcnds(ifc, 3) = clnds(icl, 4);
        ++ifc;
        // face 2.
        clfcs(icl, 2) = ifc;
        fcnds(ifc, 1) = clnds(icl, 2);
        fcnds(ifc, 2) = clnds(icl, 5);
        fcnds(ifc, 3) = clnds(icl, 1);
        ++ifc;
        // face 3.
        clfcs(icl, 3) = ifc;
        fcnds(ifc, 1) = clnds(icl, 3);
        fcnds(ifc, 2) = clnds(icl, 5);
        fcnds(ifc, 3) = clnds(icl, 2);
        ++ifc;
        // face 4.
        clfcs(icl, 4) = ifc;
        fcnds(ifc, 1) = clnds(icl, 4);
        fcnds(ifc, 2) = clnds(icl, 5);
        fcnds(ifc, 3) = clnds(icl, 3);
        ++ifc;
        // face 5.
        clfcs(icl, 5) = ifc;
        fcnds(ifc, 1) = clnds(icl, 1);
        fcnds(ifc, 2) = clnds(icl, 4);
        fcnds(ifc, 3) = clnds(icl, 3);
        fcnds(ifc, 4) = clnds(icl, 2);
        ++ifc;
        return NFACE;
    }

    // Data members.
    size_t nnode = 0;
    size_t nface = 0;
    size_t mface = 0;
    SimpleArray<int_type> const & cltpn;
    SimpleArray<int_type> const & clnds;
    SimpleArray<int_type> clfcs{};
    SimpleArray<int_type> fctpn{};
    SimpleArray<int_type> fcnds{};
    SimpleArray<int_type> fccls{};
    std::vector<uint_type> dedupmap{};

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    FaceBuilder(size_t nnode_in, SimpleArray<int_type> const & cltpn_in, SimpleArray<int_type> const & clnds_in)
        : nnode(nnode_in)
        , nface(count_mface(cltpn_in))
        , mface(nface)
        , cltpn(cltpn_in)
        , clnds(clnds_in)
        , clfcs(shape_type{cltpn.nbody(), CLMFC + 1}, -1)
        , fctpn(shape_type{static_cast<ssize_t>(mface)}, -1)
        , fcnds(shape_type{static_cast<ssize_t>(mface), FCMND + 1}, -1)
        , fccls(shape_type{static_cast<ssize_t>(mface), FCREL}, -1)
        , dedupmap(mface)
    {
        populate();
        make_dedupmap();
        remap_face();
    }

    static size_t count_mface(SimpleArray<int_type> const & tcltpn)
    {
        // clang-format off
        return std::accumulate
        (
            tcltpn.body(), tcltpn.body()+tcltpn.nbody(), static_cast<size_t>(0)
          , [](size_t a, uint8_t b){ return a + static_cast<size_t>(CellType::by_id(b).nface()); }
        );
        // clang-format on
    }

    void populate()
    {
        int_type ifc = 0;
        for (int icl = 0; icl < static_cast<int_type>(cltpn.nbody()); ++icl)
        {
            switch (cltpn(icl))
            {
            case CellType::POINT:
                ifc += static_cast<int_type>(add_point(icl, ifc));
                break;
            case CellType::LINE:
                ifc += static_cast<int_type>(add_line(icl, ifc));
                break;
            case CellType::QUADRILATERAL:
                ifc += static_cast<int_type>(add_quadrilateral(icl, ifc));
                break;
            case CellType::TRIANGLE:
                ifc += static_cast<int_type>(add_triangle(icl, ifc));
                break;
            case CellType::HEXAHEDRON:
                ifc += static_cast<int_type>(add_hexahedron(icl, ifc));
                break;
            case CellType::TETRAHEDRON:
                ifc += static_cast<int_type>(add_tetrahedron(icl, ifc));
                break;
            case CellType::PRISM:
                ifc += static_cast<int_type>(add_prism(icl, ifc));
                break;
            case CellType::PYRAMID:
                ifc += static_cast<int_type>(add_pyramid(icl, ifc));
                break;
            default:
                break;
            }
        }
    }

    SimpleArray<int_type> make_ndfcs()
    {
        // first pass: get the maximum number of faces.
        std::vector<size_t> ndnfc(nnode, 0);
        for (size_t ifc = 0; ifc < mface; ++ifc)
        {
            int_type const fcnnd = fcnds(ifc, 0);
            for (int_type const * p = fcnds.vptr(ifc, 1); p != fcnds.vptr(ifc, fcnnd + 1); ++p)
            {
                ndnfc[*p] += 1;
            }
        }

        // second pass: scan again to build hash table.
        size_t const ndmfc = *std::max_element(ndnfc.begin(), ndnfc.end());
        SimpleArray<int_type> ndfcs(shape_type{static_cast<ssize_t>(nnode), static_cast<ssize_t>(ndmfc) + 1});
        for (size_t ind = 0; ind < nnode; ++ind)
        {
            ndfcs(ind, 0) = 0;
            for (size_t it = 1; it <= ndmfc; ++it)
            {
                ndfcs(ind, it) = -1;
            }
        }
        for (size_t ifc = 0; ifc < mface; ++ifc)
        {
            int_type const fcnnd = fcnds(ifc, 0);
            for (int_type const * p = fcnds.vptr(ifc, 1); p != fcnds.vptr(ifc, fcnnd + 1); ++p)
            {
                ndfcs(*p, 0) += 1;
                ndfcs(*p, ndfcs(*p, 0)) = static_cast<int_type>(ifc);
            }
        }

        return ndfcs;
    }

    /**
     * @brief Build the (hash) table to know what faces connect to each node.
     */
    void make_dedupmap()
    {
        auto const & ndfcs = make_ndfcs();

        auto check_match = [this](int_type const ifc, int_type const jfc)
        {
            int_type cond = 0;
            // scan all nodes in ifc and jfc to see if all the same.
            for (int_type jnf = 1; jnf <= fcnds(jfc, 0); ++jnf)
            {
                // clang-format off
                auto result = std::find
                (
                    fcnds.vptr(ifc, 1)
                  , fcnds.vptr(ifc, fcnds(ifc, 0)+1)
                  , fcnds(jfc, jnf)
                );
                // clang-format off
                if (result != fcnds.vptr(ifc, fcnds(ifc, 0)+1))
                {
                    ++cond;
                }
            }
            return cond;
        };

        // scan for duplicated faces and build duplication map.
        std::iota(dedupmap.begin(), dedupmap.end(), 0);
        for (size_t ifc = 0 ; ifc < mface ; ++ifc)
        {
            if (dedupmap[ifc] == ifc)
            {
                int_type const ind = fcnds(ifc, 1); // take only the FIRST node of a face.
                for (int_type it = 1 ; it <= ndfcs(ind, 0) ; ++it)
                {
                    size_t const jfc = ndfcs(ind, it);
                    // test for duplication.
                    if
                    (
                        (jfc != ifc)
                     && (fctpn(jfc) == fctpn(ifc))
                     && (check_match(static_cast<int_type>(ifc), static_cast<int_type>(jfc)) == fcnds(static_cast<int_type>(jfc), 0))
                    )
                    {
                        dedupmap[jfc] = static_cast<uint_type>(ifc);  // record duplication.
                    }
                }
            }
        }
    }

    void remap_face()
    {
        // use the duplication map to remap nodes in faces, and build renewed map.
        std::vector<int_type> map2(mface);
        {
            size_t jfc = 0;
            for (size_t ifc = 0 ; ifc < mface ; ++ifc)
            {
                if (dedupmap[ifc] == ifc)
                {
                    //std::copy(fcnds.vptr(ifc, 0), fcnds.vptr(ifc, CellType::FCMND), fcnds.vptr(jfc));
                    for (int_type inf = 0 ; inf <= FCMND ; ++inf)
                    {
                        fcnds(jfc, inf) = fcnds(ifc, inf);
                    }
                    fctpn(jfc) = fctpn(ifc);
                    map2[ifc] = static_cast<int_type>(jfc);
                    // increment j-face.
                    jfc += 1;
                }
                else
                {
                    map2[ifc] = map2[dedupmap[ifc]];
                }
            }
            nface = jfc; // record deduplicated number of face.
        }

        // rebuild faces in cells and build face neighboring, according to the
        // renewed face map.
        size_t const ncell = cltpn.nbody();
        for (size_t icl = 0 ; icl < ncell ; ++icl)
        {
            for (int_type ifl = 1 ; ifl <= clfcs(icl, 0) ; ++ifl)
            {
                int_type const ifc = clfcs(icl, ifl);
                int_type const jfc = map2[ifc];
                // rebuild faces in cells.
                clfcs(icl, ifl) = jfc;
                // build face neighboring.
                if      (-1 == fccls(jfc, 0)) { fccls(jfc, 0) = static_cast<int_type>(icl); }
                else if (-1 == fccls(jfc, 1)) { fccls(jfc, 1) = static_cast<int_type>(icl); }
            }
        }
    }

    void rebuild_fctpn(decltype(fctpn) & ofctpn)
    {
        ofctpn.remake(shape_type{static_cast<ssize_t>(nface)});
        std::copy(fctpn.vptr(0), fctpn.vptr(nface), ofctpn.vptr(0));
    }

    void rebuild_fcnds(decltype(fcnds) & ofcnds)
    {
        ofcnds.remake(shape_type{static_cast<ssize_t>(nface), FCMND+1});
        std::copy(fcnds.vptr(0, 0), fcnds.vptr(nface, 0), ofcnds.vptr(0, 0));
    }

    void rebuild_fccls(decltype(fccls) & ofccls)
    {
        ofccls.remake(shape_type{static_cast<ssize_t>(nface), FCREL});
        std::copy(fccls.vptr(0, 0), fccls.vptr(nface, 0), ofccls.vptr(0, 0));
    }

}; /* end struct FaceBuilder */

} /* end namespace detail */

/**
 * Extract interier faces from node list of cells.  Subroutine is designed to
 * handle all types of cells.
 */
void StaticMesh::build_faces_from_cells()
{
    detail::FaceBuilder<number_base> fb(m_nnode, m_cltpn, m_clnds);
    m_nface = static_cast<uint_type>(fb.nface);

    // recreate member tables.
    fb.rebuild_fctpn(m_fctpn);
    fb.rebuild_fcnds(m_fcnds);
    fb.rebuild_fccls(m_fccls);
    m_fccnd.remake(small_vector<ssize_t>{static_cast<ssize_t>(nface()), m_ndim}, 0);
    m_fcnml.remake(small_vector<ssize_t>{static_cast<ssize_t>(nface()), m_ndim}, 0);
    m_fcara.remake(small_vector<ssize_t>{static_cast<ssize_t>(nface())}, 0);
    std::copy(fb.clfcs.vptr(0, 0), fb.clfcs.vptr(m_ncell, 0), m_clfcs.vptr(0, 0));
}

void StaticMesh::build_edge()
{
    using etype = std::pair<int32_t, int32_t>;
    std::set<etype> known_edges;
    std::vector<etype> edges;

    size_t nedge_max = 0;
    for (uint32_t ifc = 0; ifc < nface(); ++ifc)
    {
        nedge_max += fcnds(ifc, 0);
    }
    edges.reserve(nedge_max);
    // Can we optimize the memory allocation for the known_edges (std::set) too?

    for (uint32_t ifc = 0; ifc < nface(); ++ifc)
    {
        for (int32_t inf = 1; inf <= fcnds(ifc, 0); ++inf)
        {
            // Determine the edge of this node index.
            unsigned int const idx1 = (fcnds(ifc, 0) == inf) ? 1 : inf+1;
            int32_t nd0 = fcnds(ifc, inf);
            int32_t nd1 = fcnds(ifc, idx1);
            if (nd0 > nd1) // Lower node index goes first.
            {
                std::swap(nd0, nd1);
            }
            // Add the edge to the container array.
            etype const edge(nd0, nd1);
            if (0 == known_edges.count(edge))
            {
                edges.push_back(edge);
                known_edges.insert(edge);
            }
        }
    }

    // Build the edge node array and populate.
    m_ednds.remake(small_vector<ssize_t>{static_cast<ssize_t>(edges.size()), 2}, 0);
    for (size_t ied = 0; ied < edges.size(); ++ied)
    {
        etype const edge = edges[ied];
        m_ednds(ied, 0) = edge.first;
        m_ednds(ied, 1) = edge.second;
    }
}

/**
 * Calculate all metric information, including:
 *
 *  1. center of faces.
 *  2. unit normal and area of faces.
 *  3. center of cells.
 *  4. volume of cells.
 *
 * And fcnds could be reordered.
 */
/* NOLINTNEXTLINE(readability-function-cognitive-complexity) */
void StaticMesh::calc_metric()
{
    // compute face centroids.
    if (m_ndim == 2)
    {
        // 2D faces must be edge.
        for (size_t ifc = 0 ; ifc < m_nface ; ++ifc)
        {
            // point 1.
            {
                int_type const ind = m_fcnds(ifc, 1);
                m_fccnd(ifc, 0) = m_ndcrd(ind, 0);
                m_fccnd(ifc, 1) = m_ndcrd(ind, 1);
            }
            // point 2.
            {
                int_type const ind = m_fcnds(ifc, 2);
                m_fccnd(ifc, 0) += m_ndcrd(ind, 0);
                m_fccnd(ifc, 1) += m_ndcrd(ind, 1);
            }
            // average.
            m_fccnd(ifc, 0) /= 2;
            m_fccnd(ifc, 1) /= 2;
        }
    }
    else if (m_ndim == 3)
    {
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            std::array<real_type, 3> crd; // NOLINT(cppcoreguidelines-pro-type-member-init)
            std::array<std::array<real_type, 3>, FCMND+2> cfd; // NOLINT(cppcoreguidelines-pro-type-member-init)
            // find averaged point.
            cfd[0][0] = cfd[0][1] = cfd[0][2] = 0.0;
            size_t const nnd = m_fcnds(ifc, 0);
            for (size_t inf = 1 ; inf <= nnd ; ++inf)
            {
                int_type const ind = m_fcnds(ifc, inf);
                cfd[inf][0]  = m_ndcrd(ind, 0);
                cfd[0  ][0] += m_ndcrd(ind, 0);
                cfd[inf][1]  = m_ndcrd(ind, 1);
                cfd[0  ][1] += m_ndcrd(ind, 1);
                cfd[inf][2]  = m_ndcrd(ind, 2);
                cfd[0  ][2] += m_ndcrd(ind, 2);
            }
            cfd[nnd+1][0] = cfd[1][0];
            cfd[nnd+1][1] = cfd[1][1];
            cfd[nnd+1][2] = cfd[1][2];
            cfd[0][0] /= nnd;
            cfd[0][1] /= nnd;
            cfd[0][2] /= nnd;
            // calculate area.
            real_type voc = 0.0;
            m_fccnd(ifc, 0) = m_fccnd(ifc, 1) = m_fccnd(ifc, 2) = 0.0;
            for (size_t inf = 1 ; inf <= nnd ; ++inf)
            {
                // NOLINTBEGIN(readability-math-missing-parentheses)
                crd[0] = (cfd[0][0] + cfd[inf][0] + cfd[inf+1][0])/3;
                crd[1] = (cfd[0][1] + cfd[inf][1] + cfd[inf+1][1])/3;
                crd[2] = (cfd[0][2] + cfd[inf][2] + cfd[inf+1][2])/3;
                real_type const du0 = cfd[inf][0] - cfd[0][0];
                real_type const du1 = cfd[inf][1] - cfd[0][1];
                real_type const du2 = cfd[inf][2] - cfd[0][2];
                real_type const dv0 = cfd[inf+1][0] - cfd[0][0];
                real_type const dv1 = cfd[inf+1][1] - cfd[0][1];
                real_type const dv2 = cfd[inf+1][2] - cfd[0][2];
                real_type const dw0 = du1*dv2 - du2*dv1;
                real_type const dw1 = du2*dv0 - du0*dv2;
                real_type const dw2 = du0*dv1 - du1*dv0;
                real_type const vob = std::sqrt(dw0*dw0 + dw1*dw1 + dw2*dw2);
                m_fccnd(ifc, 0) += crd[0] * vob;
                m_fccnd(ifc, 1) += crd[1] * vob;
                m_fccnd(ifc, 2) += crd[2] * vob;
                voc += vob;
                // NOLINTEND(readability-math-missing-parentheses)
            }
            m_fccnd(ifc, 0) /= voc;
            m_fccnd(ifc, 1) /= voc;
            m_fccnd(ifc, 2) /= voc;
        }
    }

    // compute face normal vector and area.
    if (m_ndim == 2)
    {
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            // 2D faces are always lines.
            int_type const ind1 = m_fcnds(ifc, 1);
            int_type const ind2 = m_fcnds(ifc, 2);
            // face normal.
            m_fcnml(ifc, 0) = m_ndcrd(ind2, 1) - m_ndcrd(ind1, 1);
            m_fcnml(ifc, 1) = m_ndcrd(ind1, 0) - m_ndcrd(ind2, 0);
            // face area. NOLINTNEXTLINE(readability-math-missing-parentheses)
            m_fcara(ifc) = std::sqrt(m_fcnml(ifc, 0)*m_fcnml(ifc, 0) + m_fcnml(ifc, 1)*m_fcnml(ifc, 1));
            // normalize face normal.
            m_fcnml(ifc, 0) /= m_fcara(ifc);
            m_fcnml(ifc, 1) /= m_fcara(ifc);
        }
    }
    else if (m_ndim == 3)
    {
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            // compute radial vector.
            std::array<std::array<real_type, 3>, FCMND> radvec; // NOLINT(cppcoreguidelines-pro-type-member-init)
            size_t const nnd = m_fcnds(ifc, 0);
            for (size_t inf = 0 ; inf < nnd ; ++inf)
            {
                int_type const ind = m_fcnds(ifc, inf+1);
                radvec[inf][0] = m_ndcrd(ind, 0) - m_fccnd(ifc, 0);
                radvec[inf][1] = m_ndcrd(ind, 1) - m_fccnd(ifc, 1);
                radvec[inf][2] = m_ndcrd(ind, 2) - m_fccnd(ifc, 2);
            }
            // compute cross product.
            // NOLINTBEGIN(readability-math-missing-parentheses)
            m_fcnml(ifc, 0) = radvec[nnd-1][1]*radvec[0][2]
                            - radvec[nnd-1][2]*radvec[0][1];
            m_fcnml(ifc, 1) = radvec[nnd-1][2]*radvec[0][0]
                            - radvec[nnd-1][0]*radvec[0][2];
            m_fcnml(ifc, 2) = radvec[nnd-1][0]*radvec[0][1]
                            - radvec[nnd-1][1]*radvec[0][0];
            for (size_t ind = 1 ; ind < nnd ; ++ind)
            {
                m_fcnml(ifc, 0) += radvec[ind-1][1]*radvec[ind][2]
                                 - radvec[ind-1][2]*radvec[ind][1];
                m_fcnml(ifc, 1) += radvec[ind-1][2]*radvec[ind][0]
                                 - radvec[ind-1][0]*radvec[ind][2];
                m_fcnml(ifc, 2) += radvec[ind-1][0]*radvec[ind][1]
                                 - radvec[ind-1][1]*radvec[ind][0];
            }
            // compute face area.
            m_fcara(ifc) = std::sqrt
            (
                m_fcnml(ifc, 0)*m_fcnml(ifc, 0)
              + m_fcnml(ifc, 1)*m_fcnml(ifc, 1)
              + m_fcnml(ifc, 2)*m_fcnml(ifc, 2)
            );
            // NOLINTEND(readability-math-missing-parentheses)
            // normalize normal vector.
            m_fcnml(ifc, 0) /= m_fcara(ifc);
            m_fcnml(ifc, 1) /= m_fcara(ifc);
            m_fcnml(ifc, 2) /= m_fcara(ifc);
            // get real face area.
            m_fcara(ifc) /= 2.0;
        }
    }

    // compute cell centers.
    if (m_ndim == 2)
    {
        for (size_t icl = 0 ; icl < ncell() ; ++icl)
        {
            if ((use_incenter()) && (CellType::TRIANGLE == m_cltpn(icl)))
            {
                real_type voc = 0.0;
                {
                    int_type const ind = m_clnds(icl, 1);
                    real_type const vob = m_fcara(m_clfcs(icl, 2));
                    voc += vob;
                    m_clcnd(icl, 0) = vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) = vob*m_ndcrd(ind, 1);
                }
                {
                    int_type const ind = m_clnds(icl, 2);
                    real_type const vob = m_fcara(m_clfcs(icl, 3));
                    voc += vob;
                    m_clcnd(icl, 0) += vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) += vob*m_ndcrd(ind, 1);
                }
                {
                    int_type const ind = m_clnds(icl, 3);
                    real_type const vob = m_fcara(m_clfcs(icl, 1));
                    voc += vob;
                    m_clcnd(icl, 0) += vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) += vob*m_ndcrd(ind, 1);
                }
                m_clcnd(icl, 0) /= voc;
                m_clcnd(icl, 1) /= voc;
            }
            else // centroids.
            {
                // averaged point.
                std::array<real_type, 2> crd; // NOLINT(cppcoreguidelines-pro-type-member-init)
                crd[0] = crd[1] = 0.0;
                size_t const nnd = m_clnds(icl, 0);
                for (size_t inc = 1 ; inc <= nnd ; ++inc)
                {
                    int_type const ind = m_clnds(icl, inc);
                    crd[0] += m_ndcrd(ind, 0);
                    crd[1] += m_ndcrd(ind, 1);
                }
                crd[0] /= nnd;
                crd[1] /= nnd;
                // weight centroid.
                real_type voc = 0.0;
                m_clcnd(icl, 0) = m_clcnd(icl, 1) = 0.0;
                size_t const nfc = m_clfcs(icl, 0);
                for (size_t ifl = 1 ; ifl <= nfc ; ++ifl)
                {
                    // NOLINTBEGIN(readability-math-missing-parentheses)
                    int_type const ifc = m_clfcs(icl, ifl);
                    real_type const du0 = crd[0] - m_fccnd(ifc, 0);
                    real_type const du1 = crd[1] - m_fccnd(ifc, 1);
                    real_type const vob = fabs(du0*m_fcnml(ifc, 0) + du1*m_fcnml(ifc, 1)) * m_fcara(ifc);
                    voc += vob;
                    real_type const dv0 = m_fccnd(ifc, 0) + du0/3;
                    real_type const dv1 = m_fccnd(ifc, 1) + du1/3;
                    m_clcnd(icl, 0) += dv0 * vob;
                    m_clcnd(icl, 1) += dv1 * vob;
                    // NOLINTEND(readability-math-missing-parentheses)
                }
                m_clcnd(icl, 0) /= voc;
                m_clcnd(icl, 1) /= voc;
            }
        }
    }
    else if (m_ndim == 3)
    {
        for (size_t icl = 0 ; icl < ncell() ; ++icl)
        {
            if ((use_incenter()) && (CellType::TETRAHEDRON == m_cltpn(icl)))
            {
                real_type voc = 0.0;
                {
                    int_type const ind = m_clnds(icl, 1);
                    real_type const vob = m_fcara(m_clfcs(icl, 4));
                    voc += vob;
                    m_clcnd(icl, 0) = vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) = vob*m_ndcrd(ind, 1);
                    m_clcnd(icl, 2) = vob*m_ndcrd(ind, 2);
                }
                {
                    int_type const ind = m_clnds(icl, 2);
                    real_type const vob = m_fcara(m_clfcs(icl, 3));
                    voc += vob;
                    m_clcnd(icl, 0) = vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) = vob*m_ndcrd(ind, 1);
                    m_clcnd(icl, 2) = vob*m_ndcrd(ind, 2);
                }
                {
                    int_type const ind = m_clnds(icl, 3);
                    real_type const vob = m_fcara(m_clfcs(icl, 2));
                    voc += vob;
                    m_clcnd(icl, 0) = vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) = vob*m_ndcrd(ind, 1);
                    m_clcnd(icl, 2) = vob*m_ndcrd(ind, 2);
                }
                {
                    int_type const ind = m_clnds(icl, 4);
                    real_type const vob = m_fcara(m_clfcs(icl, 1));
                    voc += vob;
                    m_clcnd(icl, 0) = vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) = vob*m_ndcrd(ind, 1);
                    m_clcnd(icl, 2) = vob*m_ndcrd(ind, 2);
                }
                m_clcnd(icl, 0) /= voc;
                m_clcnd(icl, 1) /= voc;
                m_clcnd(icl, 2) /= voc;
            }
            else // centroids.
            {
                // averaged point.
                std::array<real_type, 3> crd; // NOLINT(cppcoreguidelines-pro-type-member-init)
                crd[0] = crd[1] = crd[2] = 0.0;
                size_t const nnd = m_clnds(icl, 0);
                for (size_t inc = 1 ; inc <= nnd ; ++inc)
                {
                    int_type const ind = m_clnds(icl, inc);
                    crd[0] += m_ndcrd(ind, 0);
                    crd[1] += m_ndcrd(ind, 1);
                    crd[2] += m_ndcrd(ind, 2);
                }
                crd[0] /= nnd;
                crd[1] /= nnd;
                crd[2] /= nnd;
                // weight centroid.
                real_type voc = 0.0;
                m_clcnd(icl, 0) = m_clcnd(icl, 1) = m_clcnd(icl, 2) = 0.0;
                size_t const nfc = m_clfcs(icl, 0);
                for (size_t ifl = 1 ; ifl <= nfc ; ++ifl)
                {
                    // NOLINTBEGIN(readability-math-missing-parentheses)
                    int_type const ifc = m_clfcs(icl, ifl);
                    real_type const du0 = crd[0] - m_fccnd(ifc, 0);
                    real_type const du1 = crd[1] - m_fccnd(ifc, 1);
                    real_type const du2 = crd[2] - m_fccnd(ifc, 2);
                    real_type const vob = fabs(du0*m_fcnml(ifc, 0) + du1*m_fcnml(ifc, 1) + du2*m_fcnml(ifc, 2)) * m_fcara(ifc);
                    voc += vob;
                    real_type const dv0 = m_fccnd(ifc, 0) + du0/4;
                    real_type const dv1 = m_fccnd(ifc, 1) + du1/4;
                    real_type const dv2 = m_fccnd(ifc, 2) + du2/4;
                    m_clcnd(icl, 0) += dv0 * vob;
                    m_clcnd(icl, 1) += dv1 * vob;
                    m_clcnd(icl, 2) += dv2 * vob;
                    // NOLINTEND(readability-math-missing-parentheses)
                }
                m_clcnd(icl, 0) /= voc;
                m_clcnd(icl, 1) /= voc;
                m_clcnd(icl, 2) /= voc;
            }
        }
    }

    // compute volume for each cell.
    auto reorder_face = [this](int_type ifc)
    {
        size_t const nnd = m_fcnds(ifc, 0);
        std::array<int_type, FCMND> ndstf; // NOLINT(cppcoreguidelines-pro-type-member-init)
        for (size_t jt = 0 ; jt < nnd ; ++jt)
        {
            ndstf[jt] = m_fcnds(ifc, nnd-jt);
        }
        for (size_t jt = 0 ; jt < nnd ; ++jt)
        {
            m_fcnds(ifc, jt+1) = ndstf[jt];
        }
        for (size_t idm = 0 ; idm < m_ndim ; ++idm)
        {
            m_fcnml(ifc, idm) = -m_fcnml(ifc, idm);
        }
    };
    for (size_t icl = 0 ; icl < ncell() ; ++icl)
    {
        m_clvol(icl) = 0.0;
        size_t const nfc = m_clfcs(icl, 0);
        for (size_t it = 1 ; it <= nfc ; ++it)
        {
            int_type const ifc = m_clfcs(icl, it);
            // calculate volume associated with each face.
            real_type vol = 0.0;
            for (size_t idm = 0 ; idm < m_ndim ; ++idm)
            {
                vol += (m_fccnd(ifc, idm) - m_clcnd(icl, idm)) * m_fcnml(ifc, idm);
            }
            vol *= m_fcara(ifc);
            // check if need to reorder node definition and connecting cell
            // list for the face.
            size_t const this_fcl = m_fccls(ifc, 0);
            if (vol < 0.0)
            {
                if (this_fcl == icl) { reorder_face(ifc); }
                vol = -vol;
            }
            else
            {
                if (this_fcl != icl) { reorder_face(ifc); }
            }
            // accumulate the volume for the cell.
            m_clvol(icl) += vol;
        }
        // calculate the real volume.
        m_clvol(icl) /= m_ndim;
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
