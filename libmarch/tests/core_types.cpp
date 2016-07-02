/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

#include "march/march.hpp"

using namespace march;

TEST(typesTest, TypeId) {
    DataTypeId val;
    val = type_to<bool>::id;
    EXPECT_EQ(val, MH_BOOL);
    val = type_to<int8_t>::id;
    EXPECT_EQ(val, MH_INT8);
    val = type_to<uint8_t>::id;
    EXPECT_EQ(val, MH_UINT8);
    val = type_to<int16_t>::id;
    EXPECT_EQ(val, MH_INT16);
    val = type_to<uint16_t>::id;
    EXPECT_EQ(val, MH_UINT16);
    val = type_to<int32_t>::id;
    EXPECT_EQ(val, MH_INT32);
    val = type_to<uint32_t>::id;
    EXPECT_EQ(val, MH_UINT32);
    val = type_to<int64_t>::id;
    EXPECT_EQ(val, MH_INT64);
    val = type_to<uint64_t>::id;
    EXPECT_EQ(val, MH_UINT64);
    val = type_to<float>::id;
    EXPECT_EQ(val, MH_FLOAT);
    val = type_to<double>::id;
    EXPECT_EQ(val, MH_DOUBLE);
    val = type_to<std::complex<float>>::id;
    EXPECT_EQ(val, MH_CFLOAT);
    val = type_to<std::complex<double>>::id;
    EXPECT_EQ(val, MH_CDOUBLE);
}

TEST(typesTest, cpptype) {
    static_assert(std::is_same<id_to<MH_BOOL>::type, bool>::value, "bad bool");
    static_assert(std::is_same<id_to<MH_INT8>::type, int8_t>::value, "bad int8");
    static_assert(std::is_same<id_to<MH_UINT8>::type, uint8_t>::value, "bad uint8");
    static_assert(std::is_same<id_to<MH_INT16>::type, int16_t>::value, "bad int16");
    static_assert(std::is_same<id_to<MH_UINT16>::type, uint16_t>::value, "bad uint16");
    static_assert(std::is_same<id_to<MH_INT32>::type, int32_t>::value, "bad int32");
    static_assert(std::is_same<id_to<MH_UINT32>::type, uint32_t>::value, "bad uint32");
    static_assert(std::is_same<id_to<MH_INT64>::type, int64_t>::value, "bad int64");
    static_assert(std::is_same<id_to<MH_UINT64>::type, uint64_t>::value, "bad uint64");
    static_assert(std::is_same<id_to<MH_FLOAT>::type, float>::value, "bad float");
    static_assert(std::is_same<id_to<MH_DOUBLE>::type, double>::value, "bad double");
    static_assert(std::is_same<id_to<MH_CFLOAT>::type, std::complex<float>>::value, "bad cfloat");
    static_assert(std::is_same<id_to<MH_CDOUBLE>::type, std::complex<double>>::value, "bad cdouble");
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
