/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <gtest/gtest.h>

#include "march/gas/gas.hpp"

using namespace march;
using namespace march::gas;

class GasTestBase : public ::testing::Test {

protected:

    void SetUp() override { fill_triangles(); }

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
        blk.build_interior();
        blk.build_boundary();
        blk.build_ghost();
    }

    std::shared_ptr<UnstructuredBlock<2>> m_triangles;

}; /* end class GasTestBase */

class GasSolverTest : public GasTestBase {};

TEST_F(GasSolverTest, BlockConstructor) {
    Solver<2>::construct(m_triangles); // good as long as it doesn't crash.
}

TEST_F(GasSolverTest, Update) {
    auto svr_holder = Solver<2>::construct(m_triangles);
    auto & svr = *svr_holder;
    svr.update(0, 0); // good as long as it doesn't crash.
}

TEST_F(GasSolverTest, CalcCfl) {
    auto svr_holder = Solver<2>::construct(m_triangles);
    auto & svr = *svr_holder;
    svr.calc_cfl(); // good as long as it doesn't crash.
}

TEST_F(GasSolverTest, CalcSolt) {
    auto svr_holder = Solver<2>::construct(m_triangles);
    auto & svr = *svr_holder;
    svr.calc_so0t(); // good as long as it doesn't crash.
}

TEST_F(GasSolverTest, CalcSoln) {
    auto svr_holder = Solver<2>::construct(m_triangles);
    auto & svr = *svr_holder;
    svr.calc_so0t();
    svr.calc_so0n(); // good as long as it doesn't crash.
}

TEST_F(GasSolverTest, CalcDsoln) {
    auto svr_holder = Solver<2>::construct(m_triangles);
    auto & svr = *svr_holder;
    svr.calc_so0t();
    svr.calc_so0n();
    svr.calc_so1n(); // good as long as it doesn't crash.
}

class GasQuantityTest : public GasTestBase {};

TEST_F(GasQuantityTest, Update) {
    auto svr_holder = Solver<2>::construct(m_triangles);
    auto & svr = *svr_holder;
    svr.make_qty();
    svr.calc_so0t();
    svr.calc_so0n();
    svr.calc_so1n();
    auto & qty = svr.qty();
    qty->update(); // good as long as it doesn't crash.
}

class GasTrimTest : public GasTestBase {

protected:

    void SetUp() override {
        GasTestBase::SetUp();
        m_triangles_bound_0 = &m_triangles->bndvec().at(0);
        m_triangles_solver = Solver<2>::construct(m_triangles);
        auto & svr = *m_triangles_solver;
        svr.calc_so0t();
        svr.calc_so0n();
        svr.calc_so1n();
    }

    template< class TrimType > std::unique_ptr<TrimType> get_trim() {
        return std::unique_ptr<TrimType>(new TrimType(*m_triangles_solver, *m_triangles_bound_0));
    }

    std::shared_ptr<Solver<2>> m_triangles_solver;
    BoundaryData * m_triangles_bound_0;

}; /* end class GasTrimTest */

TEST_F(GasTrimTest, Boundary) {
    auto & bnd = *m_triangles_bound_0;
    // good as long as it doesn't crash.
    EXPECT_EQ((bnd.nbound()), 3);
}

TEST_F(GasTrimTest, NoOp) {
    // no-op trimming; this trim is for debugging
    std::unique_ptr<TrimNoOp<2>> noop = get_trim<TrimNoOp<2>>();
    noop->apply_do0();
    noop->apply_do1();
}

TEST_F(GasTrimTest, NonRefl) {
    // non-reflective trimming
    std::unique_ptr<TrimNonRefl<2>> nonrefl = get_trim<TrimNonRefl<2>>();
    nonrefl->apply_do0();
    nonrefl->apply_do1();
}

TEST_F(GasTrimTest, SlipWall) {
    // slip wall trimming.
    std::unique_ptr<TrimSlipWall<2>> slipwall = get_trim<TrimSlipWall<2>>();
    slipwall->apply_do0();
    slipwall->apply_do1();
}

TEST_F(GasTrimTest, Inlet) {
    // inlet trimming.
    BoundaryData & bnd = *m_triangles_bound_0;
    bnd.values() = LookupTableCore(0, bnd.nbound(), {bnd.nbound(), 6}, type_to<real_type>::id);
    TrimInlet<2> inlet(*m_triangles_solver, bnd);
    inlet.apply_do0();
    inlet.apply_do1();
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
