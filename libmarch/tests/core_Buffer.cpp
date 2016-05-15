/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "march/march.hpp"

using namespace march;

TEST(BufferTest, Nbyte) {
    EXPECT_EQ(( Buffer(16).nbyte() ), 16);
}

TEST(BufferTest, Length) {
    EXPECT_EQ(( Buffer(16).length<int32_t>() ), 4);
    EXPECT_THROW(( Buffer(17).length<int32_t>() ), std::length_error);
}

TEST(BufferTest, Array) {
    Buffer buf(16 * sizeof(int32_t));
    // read
    EXPECT_THROW(( buf.array<int32_t, 15>() ), std::length_error);
    EXPECT_NO_THROW(( buf.array<int32_t, 16>() ));
    EXPECT_THROW(( buf.array<int32_t, 17>() ), std::length_error);
    // set value
    int32_t (&parr)[16] = buf.array<int32_t, 16>();
    parr[0] = 10;
    EXPECT_EQ(( buf.array<int32_t, 16>()[0] ), 10);
}

TEST(BufferTest, NoOwn) {
    /* When a data pointer is passed to the Buffer constructor, the constructed
     * Buffer object doesn't manage it's memory! */
    char * data = new char[1024*1024*4];
    Buffer * buf = new Buffer(1024*1024*4 * sizeof(char), data);
    delete buf;
    delete[] data; // needs to explicitly free the memory.
    // If free again, it should segfault.
    //delete[] data;
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
