/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <gtest/gtest.h>

#include <numeric>

#include "march/march.hpp"

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
 * begin UnstructuredBlockDataTest
 */

class UnstructuredBlockDataTest : public ::testing::Test {

protected:

    virtual void SetUp() {
        fill_triangles();
    }

    /**
     * This is how the block looks like.  It composes of only triangles.  "N"
     * denotes node, and "C" denotes cell.
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
    void fill_triangles() {
        m_triangles = UnstructuredBlock<2>::construct(/* nnode */4, /* nface */6, /* ncell */3, /* use_incenter */false);
        auto & blk = *m_triangles;
        blk.ndcrd().set_at(0,  0,  0);
        blk.ndcrd().set_at(1, -1, -1);
        blk.ndcrd().set_at(2,  1, -1);
        blk.ndcrd().set_at(3,  0,  1);
        blk.cltpn().fill(3);
        blk.clnds().set_at(0, 3, 0, 1, 2);
        blk.clnds().set_at(1, 3, 0, 2, 3);
        blk.clnds().set_at(2, 3, 0, 3, 1);
    }

    std::shared_ptr<UnstructuredBlock<2>> m_triangles;

}; /* end class UnstructuredBlockDataTest */

TEST_F(UnstructuredBlockDataTest, build_faces_from_cells) {
    typedef std::vector<index_type> ivtype;

    UnstructuredBlock<2> & blk = *m_triangles;
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
    EXPECT_EQ(blk.fcnds().nbody(), blk.nface());
    EXPECT_EQ(blk.fcnds().nghost(), blk.ngstface());
    EXPECT_EQ(blk.fcnds().vat(0), (ivtype{2, 0, 1, -1, -1}));
    EXPECT_EQ(blk.fcnds().vat(1), (ivtype{2, 1, 2, -1, -1}));
    EXPECT_EQ(blk.fcnds().vat(2), (ivtype{2, 2, 0, -1, -1}));
    EXPECT_EQ(blk.fcnds().vat(3), (ivtype{2, 2, 3, -1, -1}));
    EXPECT_EQ(blk.fcnds().vat(4), (ivtype{2, 3, 0, -1, -1}));
    EXPECT_EQ(blk.fcnds().vat(5), (ivtype{2, 3, 1, -1, -1}));

    // fccls
    EXPECT_EQ(blk.fccls().nbody(), blk.nface());
    EXPECT_EQ(blk.fccls().nghost(), blk.ngstface());
    EXPECT_EQ(blk.fccls().vat(0), (ivtype{0,  2, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(1), (ivtype{0, -1, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(2), (ivtype{0,  1, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(3), (ivtype{1, -1, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(4), (ivtype{1,  2, -1, -1}));
    EXPECT_EQ(blk.fccls().vat(5), (ivtype{2, -1, -1, -1}));
}

TEST_F(UnstructuredBlockDataTest, calc_metric) {
    typedef std::vector<real_type> rvtype;

    UnstructuredBlock<2> & blk = *m_triangles;
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

TEST_F(UnstructuredBlockDataTest, build_boundary_unspecified) {
    UnstructuredBlock<2> & blk = *m_triangles;
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

TEST_F(UnstructuredBlockDataTest, build_ghost_unspecified) {
    UnstructuredBlock<2> & blk = *m_triangles;
    blk.build_interior();
    blk.build_boundary();
    blk.build_ghost();
    EXPECT_EQ(3, blk.nbound());
    EXPECT_EQ(3, blk.ngstcell());
    EXPECT_EQ(6, blk.ngstface());
    EXPECT_EQ(3, blk.ngstnode());
}

TEST_F(UnstructuredBlockDataTest, partition) {
    UnstructuredBlock<2> & blk = *m_triangles;
    blk.build_interior();
    blk.build_boundary();
    blk.build_ghost();

    index_type edgecut;
    LookupTable<index_type, 0> part;
    std::tie(edgecut, part) = blk.partition(2);
    EXPECT_EQ(2, edgecut);
    EXPECT_EQ(3, part.nbody());
    EXPECT_EQ(0, part[0]);
    EXPECT_EQ(1, part[1]);
    EXPECT_EQ(1, part[2]);
}

/*
 * end UnstructuredBlockDataTest
 */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
