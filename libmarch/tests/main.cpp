#include <memory>

#include <gtest/gtest.h>

#include "march/march.hpp"

TEST(LookupTableTest, Sizeof) {
    using namespace march;
    using namespace march::mesh;
    static_assert(16 == sizeof(LookupTable<index_type, 1>),
                  "wrong size of LookupTable");
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
    EXPECT_THROW(tbl.getRow(-1), std::out_of_range);
    EXPECT_THROW(tbl.getRow(10), std::out_of_range);
    EXPECT_NO_THROW(tbl.row(10));
}

TEST(LookupTableTest, WriteCheck) {
    using namespace march;
    using namespace march::mesh;
    LookupTable<index_type, 1> tbl(2, 4);
    for (index_type it=-tbl.nghost(); it<tbl.nbody(); ++it) {
        tbl.getRow(it)[0] = it*10;
    }
    EXPECT_EQ(tbl.getRow(-2)[0], -20);
    EXPECT_EQ(tbl.getRow(-1)[0], -10);
    EXPECT_EQ(tbl.getRow(0)[0], 0);
    EXPECT_EQ(tbl.getRow(1)[0], 10);
    EXPECT_EQ(tbl.getRow(2)[0], 20);
    EXPECT_EQ(tbl.getRow(3)[0], 30);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
