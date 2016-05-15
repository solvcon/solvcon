#include <gtest/gtest.h>

#include "march/march.hpp"

TEST(LookupTableTest, Sizeof) {
    using namespace march;
    using namespace march::mesh;
    EXPECT_EQ(sizeof(LookupTable<index_type, 1>), 24);
    LookupTable<index_type, 1> tbl(10, 10);
    EXPECT_EQ(sizeof(LookupTable<index_type, 1>), sizeof(tbl));
    EXPECT_EQ(tbl.bytes(), 80);
    EXPECT_EQ(tbl.nelem(), 20);
    EXPECT_EQ(tbl.bytes(), tbl.nelem() * sizeof(index_type));
}

TEST(LookupTableTest, Construction) {
    using namespace march;
    using namespace march::mesh;
    using Type = LookupTable<index_type, 1>;
    EXPECT_NO_THROW(Type(2, 4));
    EXPECT_THROW(Type(-2, 4), std::invalid_argument);
    EXPECT_THROW(Type(2, -4), std::invalid_argument);
    EXPECT_THROW(Type(-2, -4), std::invalid_argument);
}

TEST(LookupTableTest, OutOfRange) {
    using namespace march;
    using namespace march::mesh;
    LookupTable<index_type, 1> tbl(0, 10);
    EXPECT_THROW(tbl.get_row(-1), std::out_of_range);
    EXPECT_THROW(tbl.get_row(10), std::out_of_range);
    EXPECT_NO_THROW(tbl.row(10));
}

TEST(LookupTableTest, WriteCheck) {
    using namespace march;
    using namespace march::mesh;
    LookupTable<index_type, 1> tbl(2, 4);
    for (index_type it=-tbl.nghost(); it<tbl.nbody(); ++it) {
        tbl.get_row(it)[0] = it*10;
    }
    EXPECT_EQ(tbl.get_row(-2)[0], -20);
    EXPECT_EQ(tbl.get_row(-1)[0], -10);
    EXPECT_EQ(tbl.get_row(0)[0], 0);
    EXPECT_EQ(tbl.get_row(1)[0], 10);
    EXPECT_EQ(tbl.get_row(2)[0], 20);
    EXPECT_EQ(tbl.get_row(3)[0], 30);
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
