/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <cstdint>
#include <array>

#include <gtest/gtest.h>

#include "march/march.hpp"

using namespace march;

/* demonstrate std::array usage */
TEST(VectorTest, ArrayCreateDefault) {
    std::array<real_type, 2> arr{0,0};
    EXPECT_EQ(std::get<0>(arr), 0);
    EXPECT_EQ(std::get<1>(arr), 0);
}

TEST(VectorTest, CreateDefault) {
    Vector<2> vec1{0,0};
    EXPECT_EQ(vec1[0], 0);
    EXPECT_EQ(vec1[1], 0);
    Vector<3> vec2{10,20,30};
    EXPECT_EQ(vec2[0], 10);
    EXPECT_EQ(vec2[1], 20);
    EXPECT_EQ(vec2[2], 30);
}

TEST(VectorTest, AssignVector) {
    Vector<3> vec1{0,0,0};
    Vector<3> vec2{10,20,30};
    vec1 = vec2;
    EXPECT_EQ(vec1[0], 10);
    EXPECT_EQ(vec1[1], 20);
    EXPECT_EQ(vec1[2], 30);
}

TEST(VectorTest, AssignArray) {
    Vector<3> vec1{0,0,0};
    real_type arr2[3] = {10,20,30};
    vec1 = arr2;
    EXPECT_EQ(vec1[0], 10);
    EXPECT_EQ(vec1[1], 20);
    EXPECT_EQ(vec1[2], 30);
    const real_type arr3[3] = {100,200,300};
    vec1 = arr3;
    EXPECT_EQ(vec1[0], 100);
    EXPECT_EQ(vec1[1], 200);
    EXPECT_EQ(vec1[2], 300);
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
