/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <iostream>

#include <gtest/gtest.h>

#include "march/mesh/ConservationElement.hpp"

#include "mesh_fixture.hpp"

using namespace march;

/*
 * begin TriangleCETest
 */

class TriangleCETest : public ::testing::Test {

public:

protected:

    virtual void SetUp() {
        m_triangles = make_triangles();
    }

    std::shared_ptr<UnstructuredBlock<2>> m_triangles;

}; /* end class TriangleCETest */

TEST_F(TriangleCETest, Construct) {
    auto & blk = *m_triangles;
    blk.build_interior();
    blk.build_boundary();
    blk.build_ghost();
    BasicCE<2>(blk, 0, 0);
    ConservationElement<2>(blk, 0);
}

/*
 * end TriangleCETest
 */

/*
 * begin TetrahedralCETest
 */

class TetrahedralCETest : public ::testing::Test {

protected:

    virtual void SetUp() {
        m_tetrahedra = make_tetrahedra();
    }

    std::shared_ptr<UnstructuredBlock<3>> m_tetrahedra;
}; /* end class TetrahedralCETest */

TEST_F(TetrahedralCETest, CEConstruct) {
    auto & blk = *m_tetrahedra;
    blk.build_interior();
    blk.build_boundary();
    blk.build_ghost();
    BasicCE<3>(blk, 0, 0);
    ConservationElement<3>(blk, 0);
}

/*
 * end TetrahedralCETest
 */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
