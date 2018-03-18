/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <gtest/gtest.h>

#include <vector>

#include "march/mesh/BoundaryData.hpp"

using namespace march;

TEST(BoundaryDataTest, DefaultConstruction) {
    BoundaryData bnd;
    EXPECT_EQ(bnd.facn().nbyte(), 0);
    EXPECT_EQ(bnd.values().nbyte(), 0);
    EXPECT_EQ(bnd.nvalue(), 0);
}

TEST(BoundaryDataTest, UsefulConstruction) {
    BoundaryData bnd(5);
    EXPECT_EQ((bnd.values<5>().nelem()), 0);
    bnd.values<5>() = LookupTable<real_type, 5>(0, 10);
    EXPECT_EQ((bnd.values<5>().nelem()), 50);
}

TEST(BoundaryDataTest, InVector) {
    std::vector<BoundaryData> bvec;
    bvec.push_back(BoundaryData(5));
    EXPECT_EQ((bvec[0].values<5>().nelem()), 0);
    EXPECT_TRUE(bvec[0].good_shape());
    bvec[0].values<5>() = LookupTable<real_type, 5>(0, 10);
    EXPECT_EQ((bvec[0].values<5>().nelem()), 50);
    EXPECT_FALSE(bvec[0].good_shape());
}

TEST(BoundaryDataTest, SetValue) {
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
