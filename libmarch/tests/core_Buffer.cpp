/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "march/march.hpp"

using namespace march;

TEST(BufferTest, Nbyte) {
    EXPECT_EQ(( Buffer::construct(16)->nbyte() ), 16);
}

TEST(BufferTest, Length) {
    EXPECT_EQ(( Buffer::construct(16)->length<int32_t>() ), 4);
    EXPECT_THROW(( Buffer::construct(17)->length<int32_t>() ), std::length_error);
}

TEST(BufferTest, Array) {
    std::shared_ptr<Buffer> buf = Buffer::construct(16 * sizeof(int32_t));
    // read
    EXPECT_THROW(( buf->array<int32_t, 15>() ), std::length_error);
    EXPECT_NO_THROW(( buf->array<int32_t, 16>() ));
    EXPECT_THROW(( buf->array<int32_t, 17>() ), std::length_error);
    // set value
    int32_t (&parr)[16] = buf->array<int32_t, 16>();
    parr[0] = 10;
    EXPECT_EQ(( buf->array<int32_t, 16>()[0] ), 10);
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
