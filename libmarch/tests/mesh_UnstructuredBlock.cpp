/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <gtest/gtest.h>

#include <numeric>

#include "march/mesh/UnstructuredBlock/UnstructuredBlock.hpp"
#include "march/mesh/BasicCE.hpp"
#include "march/mesh/CompoundCE.hpp"

using namespace march;

/*
 * begin UnstructuredBlockBasicTest
 */

TEST(UnstructuredBlockBasicTest, Construction) {
    auto blk = UnstructuredBlock<2>::construct(/* nnode */4, /* nface */6, /* ncell */3, /* use_incenter */false);
    EXPECT_EQ(blk->nnode(), 4);
    EXPECT_EQ(blk->nface(), 6);
    EXPECT_EQ(blk->ncell(), 3);
    EXPECT_EQ(blk->nbound(), 0);
    EXPECT_EQ(blk->ngstnode(), 0);
    EXPECT_EQ(blk->ngstface(), 0);
    EXPECT_EQ(blk->ngstcell(), 0);
}

/*
 * end UnstructuredBlockBasicTest
 */

/*
 * begin TriangleDataTest
 */

class TriangleDataTest : public ::testing::Test {

public:

    /**
     * This is a block composes of only triangles.  "N" denotes node, and "C"
     * denotes cell.
     *
     *                                (0,1)N3
     *                              *
     *                             /|\
     *                            / | \
     *                           /  |  \
     *                          /   |   \
     *                         /    |    \
     *                        /     |     \
     *                       /  C2  *  C1  \
     *                      /      /^\      \
     *                     /     / N0  \     \
     *                    /    /  (0,0)  \    \
     *                   /   /             \   \
     *                  /  /                 \  \
     *                 / /         C0          \ \
     *                //                         \\
     *               *-----------------------------*
     *     (-1,-1)N1                                 (1,-1)N2
     */
    static std::shared_ptr<UnstructuredBlock<2>> make_triangles() {
        auto blk = UnstructuredBlock<2>::construct(/* nnode */4, /* nface */6, /* ncell */3, /* use_incenter */false);
        blk->ndcrd().set_at(0,  0,  0);
        blk->ndcrd().set_at(1, -1, -1);
        blk->ndcrd().set_at(2,  1, -1);
        blk->ndcrd().set_at(3,  0,  1);
        blk->cltpn().fill(3);
        blk->clnds().set_at(0, 3, 0, 1, 2);
        blk->clnds().set_at(1, 3, 0, 2, 3);
        blk->clnds().set_at(2, 3, 0, 3, 1);
        return blk;
    }

protected:

    virtual void SetUp() {
        m_triangles = make_triangles();
    }

    std::shared_ptr<UnstructuredBlock<2>> m_triangles;

}; /* end class TriangleDataTest */

TEST_F(TriangleDataTest, build_faces_from_cells) {
    using ivtype = std::vector<index_type>;

    auto & blk = *m_triangles;
    blk.build_interior();

    // shape
    EXPECT_EQ(blk.ndim(), 2);
    EXPECT_EQ(blk.nnode(), 4);
    EXPECT_EQ(blk.ngstnode(), 0);
    EXPECT_EQ(blk.nface(), 6);
    EXPECT_EQ(blk.ngstface(), 0);
    EXPECT_EQ(blk.ncell(), 3);
    EXPECT_EQ(blk.ngstcell(), 0);

    // clfcs
    EXPECT_EQ(blk.clfcs().nbody(), blk.ncell());
    EXPECT_EQ(blk.clfcs().nghost(), blk.ngstcell());
    EXPECT_EQ(blk.clfcs().vat(0), (ivtype{3, 0, 1, 2, -1, -1, -1}));
    EXPECT_EQ(blk.clfcs().vat(1), (ivtype{3, 2, 3, 4, -1, -1, -1}));
    EXPECT_EQ(blk.clfcs().vat(2), (ivtype{3, 4, 5, 0, -1, -1, -1}));

    // fctpn
    EXPECT_EQ(blk.fctpn().nbody(), blk.nface());
    EXPECT_EQ(blk.fctpn().nghost(), blk.ngstface());
    for (index_type ifc=0; ifc < blk.nface(); ++ifc) {
        EXPECT_EQ(blk.fctpn().at(ifc), 1);
    }

    // fcnds
    EXPECT_EQ(blk.fcnds().nbody(), 6);
    EXPECT_EQ(blk.fcnds().nbody(), blk.nface());
    EXPECT_EQ(blk.fcnds().nghost(), blk.ngstface());
    EXPECT_EQ(blk.fcnds().vat(0), (ivtype{2, 0, 1, -1, -1}));
    EXPECT_EQ(blk.fcnds().vat(1), (ivtype{2, 1, 2, -1, -1}));
    EXPECT_EQ(blk.fcnds().vat(2), (ivtype{2, 2, 0, -1, -1}));
    EXPECT_EQ(blk.fcnds().vat(3), (ivtype{2, 2, 3, -1, -1}));
    EXPECT_EQ(blk.fcnds().vat(4), (ivtype{2, 3, 0, -1, -1}));
    EXPECT_EQ(blk.fcnds().vat(5), (ivtype{2, 3, 1, -1, -1}));

    // fccls
    EXPECT_EQ(blk.fccls().nbody(), 6);
    EXPECT_EQ(blk.fccls().nbody(), blk.nface());
    EXPECT_EQ(blk.fccls().nghost(), blk.ngstface());
    EXPECT_EQ(blk.fccls().vat(0), (ivtype{0,  2, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(1), (ivtype{0, -1, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(2), (ivtype{0,  1, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(3), (ivtype{1, -1, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(4), (ivtype{1,  2, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(5), (ivtype{2, -1, -1, -1}));
}

TEST_F(TriangleDataTest, calc_metric) {
    using rvtype = std::vector<real_type>;

    auto & blk = *m_triangles;
    blk.build_interior();

    // fccnd
    EXPECT_EQ(blk.fccnd().vat(0), (rvtype{-0.5, -0.5}));
    EXPECT_EQ(blk.fccnd().vat(1), (rvtype{ 0.0, -1.0}));
    EXPECT_EQ(blk.fccnd().vat(2), (rvtype{ 0.5, -0.5}));
    EXPECT_EQ(blk.fccnd().vat(3), (rvtype{ 0.5,  0.0}));
    EXPECT_EQ(blk.fccnd().vat(4), (rvtype{ 0.0,  0.5}));
    EXPECT_EQ(blk.fccnd().vat(5), (rvtype{-0.5,  0.0}));

    // fcnml
    EXPECT_DOUBLE_EQ(blk.fcnml().at(0)[0], -1.0/sqrt(2));
    EXPECT_DOUBLE_EQ(blk.fcnml().at(0)[1],  1.0/sqrt(2));
    EXPECT_DOUBLE_EQ(blk.fcnml().at(1)[0],  0.0        );
    EXPECT_DOUBLE_EQ(blk.fcnml().at(1)[1], -1.0        );
    EXPECT_DOUBLE_EQ(blk.fcnml().at(2)[0],  1.0/sqrt(2));
    EXPECT_DOUBLE_EQ(blk.fcnml().at(2)[1],  1.0/sqrt(2));
    EXPECT_DOUBLE_EQ(blk.fcnml().at(3)[0],  2.0/sqrt(5));
    EXPECT_DOUBLE_EQ(blk.fcnml().at(3)[1],  1.0/sqrt(5));
    EXPECT_DOUBLE_EQ(blk.fcnml().at(4)[0], -1.0        );
    EXPECT_DOUBLE_EQ(blk.fcnml().at(4)[1],  0.0        );
    EXPECT_DOUBLE_EQ(blk.fcnml().at(5)[0], -2.0/sqrt(5));
    EXPECT_DOUBLE_EQ(blk.fcnml().at(5)[1],  1.0/sqrt(5));

    // fcara
    EXPECT_DOUBLE_EQ(blk.fcara().at(0), sqrt(2));
    EXPECT_DOUBLE_EQ(blk.fcara().at(1), 2.0    );
    EXPECT_DOUBLE_EQ(blk.fcara().at(2), sqrt(2));
    EXPECT_DOUBLE_EQ(blk.fcara().at(3), sqrt(5));
    EXPECT_DOUBLE_EQ(blk.fcara().at(4), 1.0    );
    EXPECT_DOUBLE_EQ(blk.fcara().at(5), sqrt(5));

    // clcnd
    EXPECT_DOUBLE_EQ(blk.clcnd().at(0)[0],  0.0    );
    EXPECT_DOUBLE_EQ(blk.clcnd().at(0)[1], -2.0/3.0);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(1)[0],  1.0/3.0);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(1)[1],  0.0    );
    EXPECT_DOUBLE_EQ(blk.clcnd().at(2)[0], -1.0/3.0);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(2)[1],  0.0    );

    // clcnd
    EXPECT_DOUBLE_EQ(blk.clvol().at(0), 1.0);
    EXPECT_DOUBLE_EQ(blk.clvol().at(1), 0.5);
    EXPECT_DOUBLE_EQ(blk.clvol().at(2), 0.5);
}

TEST_F(TriangleDataTest, build_boundary_unspecified) {
    auto & blk = *m_triangles;
    blk.build_interior();
    EXPECT_EQ(0, blk.nbound());
    blk.build_boundary();
    EXPECT_EQ(3, blk.nbound());
    EXPECT_EQ(1, blk.bndvec().size());
    EXPECT_EQ(3, blk.bndfcs().nbody());
    EXPECT_EQ(1, blk.bndfcs()[0][0]);
    EXPECT_EQ(3, blk.bndfcs()[1][0]);
    EXPECT_EQ(5, blk.bndfcs()[2][0]);
}

TEST_F(TriangleDataTest, build_ghost_unspecified) {
    auto & blk = *m_triangles;
    blk.build_interior();
    blk.build_boundary();
    blk.build_ghost();
    EXPECT_EQ(3, blk.nbound());
    EXPECT_EQ(3, blk.ngstcell());
    EXPECT_EQ(6, blk.ngstface());
    EXPECT_EQ(3, blk.ngstnode());
}

TEST_F(TriangleDataTest, partition) {
    auto & blk = *m_triangles;
    blk.build_interior();
    blk.build_boundary();
    blk.build_ghost();

    index_type edgecut;
    LookupTable<index_type, 0> part;
    std::tie(edgecut, part) = blk.partition(2);
    EXPECT_EQ(2, edgecut);
    EXPECT_EQ(3, part.nbody());
    /* The following expectation isn't met with homebrewed scotch.
    EXPECT_EQ(0, part[0]);
    EXPECT_EQ(1, part[1]);
    EXPECT_EQ(1, part[2]);
    */
}

TEST_F(TriangleDataTest, CEMesh) {
    auto & blk = *m_triangles;
    blk.build_interior();
    blk.build_boundary();
    blk.build_ghost();

    UnstructuredBlock<2>::CEMesh cem(blk);
}

TEST_F(TriangleDataTest, CEConstruct) {
    auto & blk = *m_triangles;
    blk.build_interior();
    blk.build_boundary();
    blk.build_ghost();
    BasicCE<2>(blk, 0, 0);
    CompoundCE<2>(blk, 0);
}

/*
 * end TriangleDataTest
 */

/*
 * begin TetrahedralDataTest
 */

class TetrahedralDataTest : public ::testing::Test {

protected:

    /**
     * This is a block composes of only tetrahedra.  3D ascii art is beyond my
     * skill.  Use imagination.
     */
    static std::shared_ptr<UnstructuredBlock<3>> make_tetrahedra() {
        auto blk = UnstructuredBlock<3>::construct(/* nnode */5, /* nface */0, /* ncell */4, /* use_incenter */false);
        blk->ndcrd().set_at(0,   0,  0,  0);
        blk->ndcrd().set_at(1,  10,  0,  0);
        blk->ndcrd().set_at(2,   0, 10,  0);
        blk->ndcrd().set_at(3,   0,  0, 10);
        blk->ndcrd().set_at(4,   1,  1,  1);
        blk->cltpn().fill(5);
        blk->clnds().set_at(0, 4, 0, 1, 2, 4);
        blk->clnds().set_at(1, 4, 0, 2, 3, 4);
        blk->clnds().set_at(2, 4, 0, 3, 1, 4);
        blk->clnds().set_at(3, 4, 1, 2, 3, 4);
        return blk;
    }

    virtual void SetUp() {
        m_tetrahedra = make_tetrahedra();
    }

    std::shared_ptr<UnstructuredBlock<3>> m_tetrahedra;
}; /* end class TetrahedralDataTest */

TEST_F(TetrahedralDataTest, build_faces_from_cells) {
    using ivtype = std::vector<index_type>;

    auto & blk = *m_tetrahedra;
    blk.build_interior();

    // shape
    EXPECT_EQ(blk.ndim(), 3);
    EXPECT_EQ(blk.nnode(), 5);
    EXPECT_EQ(blk.ngstnode(), 0);
    EXPECT_EQ(blk.nface(), 10);
    EXPECT_EQ(blk.ngstface(), 0);
    EXPECT_EQ(blk.ncell(), 4);
    EXPECT_EQ(blk.ngstcell(), 0);

    // clfcs
    EXPECT_EQ(blk.clfcs().nbody(), blk.ncell());
    EXPECT_EQ(blk.clfcs().nghost(), blk.ngstcell());
    EXPECT_EQ(blk.clfcs().vat(0), (ivtype{4, 0, 1, 2, 3, -1, -1}));
    EXPECT_EQ(blk.clfcs().vat(1), (ivtype{4, 4, 2, 5, 6, -1, -1}));
    EXPECT_EQ(blk.clfcs().vat(2), (ivtype{4, 7, 5, 1, 8, -1, -1}));
    EXPECT_EQ(blk.clfcs().vat(3), (ivtype{4, 9, 3, 8, 6, -1, -1}));

    // fctpn
    EXPECT_EQ(blk.fctpn().nbody(), blk.nface());
    EXPECT_EQ(blk.fctpn().nghost(), blk.ngstface());
    for (index_type ifc=0; ifc < blk.nface(); ++ifc) {
        EXPECT_EQ(blk.fctpn().at(ifc), 3);
    }

    // fcnds
    EXPECT_EQ(blk.fcnds().nbody(), 10);
    EXPECT_EQ(blk.fcnds().nbody(), blk.nface());
    EXPECT_EQ(blk.fcnds().nghost(), blk.ngstface());
    EXPECT_EQ(blk.fcnds().vat(0), (ivtype{3, 0, 2, 1, -1}));
    EXPECT_EQ(blk.fcnds().vat(1), (ivtype{3, 0, 1, 4, -1}));
    EXPECT_EQ(blk.fcnds().vat(2), (ivtype{3, 0, 4, 2, -1}));
    EXPECT_EQ(blk.fcnds().vat(3), (ivtype{3, 1, 2, 4, -1}));
    EXPECT_EQ(blk.fcnds().vat(4), (ivtype{3, 0, 3, 2, -1}));
    EXPECT_EQ(blk.fcnds().vat(5), (ivtype{3, 0, 4, 3, -1}));
    EXPECT_EQ(blk.fcnds().vat(6), (ivtype{3, 2, 3, 4, -1}));
    EXPECT_EQ(blk.fcnds().vat(7), (ivtype{3, 0, 1, 3, -1}));
    EXPECT_EQ(blk.fcnds().vat(8), (ivtype{3, 3, 1, 4, -1}));
    EXPECT_EQ(blk.fcnds().vat(9), (ivtype{3, 2, 3, 1, -1}));

    // fccls
    EXPECT_EQ(blk.fcnds().nbody(), 10);
    EXPECT_EQ(blk.fccls().nbody(), blk.nface());
    EXPECT_EQ(blk.fccls().nghost(), blk.ngstface());
    EXPECT_EQ(blk.fccls().vat(0), (ivtype{0, -1, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(1), (ivtype{0,  2, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(2), (ivtype{0,  1, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(3), (ivtype{0,  3, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(4), (ivtype{1, -1, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(5), (ivtype{1,  2, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(6), (ivtype{1,  3, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(7), (ivtype{2, -1, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(8), (ivtype{2,  3, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(9), (ivtype{3, -1, -1, -1}));
}

TEST_F(TetrahedralDataTest, calc_metric) {
    auto & blk = *m_tetrahedra;
    blk.build_interior();

    // fccnd
    EXPECT_EQ(blk.fccnd().nbody(), 10);
    EXPECT_DOUBLE_EQ(blk.fccnd().at(0)[0], 10./3);
    EXPECT_DOUBLE_EQ(blk.fccnd().at(0)[1], 10./3);
    EXPECT_DOUBLE_EQ(blk.fccnd().at(0)[2], 0);
    // skip 1, 2, 3 for tedious to calculate by hand
    EXPECT_DOUBLE_EQ(blk.fccnd().at(4)[0], 0);
    EXPECT_DOUBLE_EQ(blk.fccnd().at(4)[1], 10./3);
    EXPECT_DOUBLE_EQ(blk.fccnd().at(4)[2], 10./3);
    // skip 5, 6 for tedious to calculate by hand
    EXPECT_DOUBLE_EQ(blk.fccnd().at(7)[0], 10./3);
    EXPECT_DOUBLE_EQ(blk.fccnd().at(7)[1], 0);
    EXPECT_DOUBLE_EQ(blk.fccnd().at(7)[2], 10./3);
    // skip 8 for tedious to calculate by hand
    EXPECT_DOUBLE_EQ(blk.fccnd().at(9)[0], 10./3);
    EXPECT_DOUBLE_EQ(blk.fccnd().at(9)[1], 10./3);
    EXPECT_DOUBLE_EQ(blk.fccnd().at(9)[2], 10./3);

    // fcnml
    EXPECT_EQ(blk.fcnml().nbody(), 10);
    EXPECT_DOUBLE_EQ(blk.fcnml().at(0)[0],  0);
    EXPECT_DOUBLE_EQ(blk.fcnml().at(0)[1],  0);
    EXPECT_DOUBLE_EQ(blk.fcnml().at(0)[2], -1);
    // skip 1, 2, 3 for tedious to calculate by hand
    EXPECT_DOUBLE_EQ(blk.fcnml().at(4)[0], -1);
    EXPECT_DOUBLE_EQ(blk.fcnml().at(4)[1],  0);
    EXPECT_DOUBLE_EQ(blk.fcnml().at(4)[2],  0);
    // skip 5, 6 for tedious to calculate by hand
    EXPECT_DOUBLE_EQ(blk.fcnml().at(7)[0],  0);
    EXPECT_DOUBLE_EQ(blk.fcnml().at(7)[1], -1);
    EXPECT_DOUBLE_EQ(blk.fcnml().at(7)[2],  0);
    // skip 8 for tedious to calculate by hand
    EXPECT_DOUBLE_EQ(blk.fcnml().at(9)[0], 1./sqrt(3));
    EXPECT_DOUBLE_EQ(blk.fcnml().at(9)[1], 1./sqrt(3));
    EXPECT_DOUBLE_EQ(blk.fcnml().at(9)[2], 1./sqrt(3));

    // fcara
    EXPECT_EQ(blk.fcara().nbody(), 10);
    EXPECT_DOUBLE_EQ(blk.fcara().at(0), 10.*10./2);
    EXPECT_DOUBLE_EQ(blk.fcara().at(1), 10.*sqrt(2)/2);
    EXPECT_DOUBLE_EQ(blk.fcara().at(2), 10.*sqrt(2)/2);
    EXPECT_DOUBLE_EQ(blk.fcara().at(3), sqrt(66)*10/2);
    EXPECT_DOUBLE_EQ(blk.fcara().at(4), 10.*10./2);
    EXPECT_DOUBLE_EQ(blk.fcara().at(5), 10.*sqrt(2)/2);
    EXPECT_DOUBLE_EQ(blk.fcara().at(6), sqrt(66)*10/2);
    EXPECT_DOUBLE_EQ(blk.fcara().at(7), 10.*10./2);
    EXPECT_DOUBLE_EQ(blk.fcara().at(8), sqrt(66)*10/2);
    EXPECT_DOUBLE_EQ(blk.fcara().at(9), sqrt(3)*100/2);

    // clcnd
    EXPECT_EQ(blk.clcnd().nbody(), 4);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(0)[0], 2.75);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(0)[1], 2.75);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(0)[2], 0.25);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(1)[0], 0.25);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(1)[1], 2.75);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(1)[2], 2.75);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(2)[0], 2.75);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(2)[1], 0.25);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(2)[2], 2.75);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(3)[0], 2.75);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(3)[1], 2.75);
    EXPECT_DOUBLE_EQ(blk.clcnd().at(3)[2], 2.75);

    // clcnd
    EXPECT_EQ(blk.clvol().nbody(), 4);
    EXPECT_DOUBLE_EQ(blk.clvol().at(0), 10.*10.*1. / 2 / 3);
    EXPECT_DOUBLE_EQ(blk.clvol().at(1), 10.*10.*1. / 2 / 3);
    EXPECT_DOUBLE_EQ(blk.clvol().at(2), 10.*10.*1. / 2 / 3);
    EXPECT_DOUBLE_EQ(blk.clvol().at(3), 10.*10.*10. / 2 / 3 - 10.*10.*1. / 2);
}

TEST_F(TetrahedralDataTest, build_boundary_unspecified) {
    auto & blk = *m_tetrahedra;
    blk.build_interior();
    EXPECT_EQ(0, blk.nbound());
    blk.build_boundary();
    EXPECT_EQ(4, blk.nbound());
    EXPECT_EQ(1, blk.bndvec().size());
    EXPECT_EQ(4, blk.bndfcs().nbody());
    EXPECT_EQ(0, blk.bndfcs()[0][0]);
    EXPECT_EQ(4, blk.bndfcs()[1][0]);
    EXPECT_EQ(7, blk.bndfcs()[2][0]);
    EXPECT_EQ(9, blk.bndfcs()[3][0]);
}

TEST_F(TetrahedralDataTest, build_ghost_unspecified) {
    auto & blk = *m_tetrahedra;
    blk.build_interior();
    blk.build_boundary();
    blk.build_ghost();
    EXPECT_EQ(4, blk.nbound());
    EXPECT_EQ(4, blk.ngstcell());
    EXPECT_EQ(12, blk.ngstface());
    EXPECT_EQ(4, blk.ngstnode());
}

TEST_F(TetrahedralDataTest, partition) {
    auto & blk = *m_tetrahedra;
    blk.build_interior();
    blk.build_boundary();
    blk.build_ghost();

    index_type edgecut;
    LookupTable<index_type, 0> part;
    std::tie(edgecut, part) = blk.partition(2);
    EXPECT_EQ(4, edgecut);
    EXPECT_EQ(4, part.nbody());
}

TEST_F(TetrahedralDataTest, CEMesh) {
    auto & blk = *m_tetrahedra;
    blk.build_interior();
    blk.build_boundary();
    blk.build_ghost();

    UnstructuredBlock<3>::CEMesh cem(blk);
}

TEST_F(TetrahedralDataTest, CEConstruct) {
    auto & blk = *m_tetrahedra;
    blk.build_interior();
    blk.build_boundary();
    blk.build_ghost();
    BasicCE<3>(blk, 0, 0);
    CompoundCE<3>(blk, 0);
}

/*
 * end TetrahedralDataTest
 */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
