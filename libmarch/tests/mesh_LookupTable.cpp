/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <gtest/gtest.h>

#include "march/march.hpp"

using namespace march;
using namespace march::mesh;

TEST(LookupTableTest, Size) {
    LookupTable<index_type, 1> tbl(10, 10);
    EXPECT_EQ(tbl.nbyte(), 80);
    EXPECT_EQ(tbl.nelem(), 20);
    EXPECT_EQ(tbl.nbyte(), tbl.nelem() * sizeof(index_type));
}

TEST(LookupTableTest, Construction) {
    using Type = LookupTable<index_type, 1>;
    EXPECT_NO_THROW(Type(2, 4));
    EXPECT_THROW(Type(-2, 4), std::invalid_argument);
    EXPECT_THROW(Type(2, -4), std::invalid_argument);
    EXPECT_THROW(Type(-2, -4), std::invalid_argument);
}

TEST(LookupTableTest, ConstructionNoOwn) {
    /* When a data pointer is passed to the LookupTable constructor, its Buffer
     * object doesn't manage its own memory! */
    using Type = LookupTable<index_type, 1>;
    char * data = new char[(2+4)*sizeof(index_type)];
    EXPECT_NO_THROW(Type(2, 4));
    delete[] data; // needs to explicitly free the memory.
    // If free again, it should segfault.
    //delete[] data;
}

TEST(LookupTableTest, OutOfRange) {
    LookupTable<index_type, 1> tb1(0, 10), tb2(0, 10);
    EXPECT_THROW(tb2.at(-1), std::out_of_range);
    EXPECT_THROW(tb2.at(10), std::out_of_range);
    EXPECT_NO_THROW(tb2[-1]);
    EXPECT_NO_THROW(tb1[10]);
}

TEST(LookupTableTest, WriteCheck) {
    LookupTable<index_type, 1> tbl(2, 4);
    for (index_type it=-tbl.nghost(); it<tbl.nbody(); ++it) {
        tbl.at(it)[0] = it*10;
    }
    EXPECT_EQ(tbl.at(-2)[0], -20);
    EXPECT_EQ(tbl.at(-1)[0], -10);
    EXPECT_EQ(tbl.at(0)[0], 0);
    EXPECT_EQ(tbl.at(1)[0], 10);
    EXPECT_EQ(tbl.at(2)[0], 20);
    EXPECT_EQ(tbl.at(3)[0], 30);
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
