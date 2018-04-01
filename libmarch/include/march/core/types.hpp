#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * \file
 * Basic typedefs and constants for libmarch.
 */

#include <cstdint>
#include <complex>
#include <limits>

#include "march/core/utility.hpp"

namespace march
{

/**
 * The enum is compatible to numpy NPY_TYPES.
 *
 * MH_LONG, MH_ULONG, MH_LONGDOUBLE, MH_CLONGDOUBLE aren't used.
 */
enum DataTypeId {
    MH_BOOL=0,
    MH_INT8, MH_UINT8,
    MH_INT16, MH_UINT16,
    MH_INT32, MH_UINT32,
    MH_LONG, MH_ULONG,
    MH_INT64, MH_UINT64,
    MH_FLOAT, MH_DOUBLE, MH_LONGDOUBLE,
    MH_CFLOAT, MH_CDOUBLE, MH_CLONGDOUBLE
};

inline size_t data_type_size(const DataTypeId dtid) {
    size_t ret = 0;
    switch (dtid) {
    case MH_BOOL:
    case MH_INT8:
    case MH_UINT8:
        ret = 1;
        break;
    case MH_INT16:
    case MH_UINT16:
        ret = 2;
        break;
    case MH_INT32:
    case MH_UINT32:
        ret = 4;
        break;
    case MH_INT64:
    case MH_UINT64:
        ret = 4;
        break;
    case MH_FLOAT:
        ret = 4;
        break;
    case MH_DOUBLE:
        ret = 8;
        break;
    default: // error.
        ret = 0;
    }
    return ret;
}

/**
 * Convert type to ID.
 */
template <typename type, typename SFINAE = void> struct type_to { };

template <typename T> struct type_to<T, typename std::enable_if<std::is_integral<T>::value>::type> {
private:
    constexpr static DataTypeId ids[8] = {
        MH_INT8,  MH_UINT8,  MH_INT16, MH_UINT16,
        MH_INT32, MH_UINT32, MH_INT64, MH_UINT64
    };
public:
    constexpr static DataTypeId id = ids[detail::log2(sizeof(T)) * 2 + (std::is_unsigned<T>::value ? 1 : 0)];
};

#define MARCH_DECL_TYPEID(Type, ID) \
    template <> struct type_to<Type> { constexpr static DataTypeId id = ID; };
MARCH_DECL_TYPEID(bool, MH_BOOL)
MARCH_DECL_TYPEID(float, MH_FLOAT)
MARCH_DECL_TYPEID(double, MH_DOUBLE)
MARCH_DECL_TYPEID(std::complex<float>, MH_CFLOAT)
MARCH_DECL_TYPEID(std::complex<double>, MH_CDOUBLE)
#undef MARCH_DECL_TYPEID

/**
 * Convert ID to type.
 */
template <int ID> struct id_to { };
template <> struct id_to<MH_BOOL> { typedef bool type; };
template <> struct id_to<MH_INT8> { typedef int8_t type; };
template <> struct id_to<MH_UINT8> { typedef uint8_t type; };
template <> struct id_to<MH_INT16> { typedef int16_t type; };
template <> struct id_to<MH_UINT16> { typedef uint16_t type; };
template <> struct id_to<MH_INT32> { typedef int32_t type; };
template <> struct id_to<MH_UINT32> { typedef uint32_t type; };
template <> struct id_to<MH_INT64> { typedef int64_t type; };
template <> struct id_to<MH_UINT64> { typedef uint64_t type; };
template <> struct id_to<MH_FLOAT> { typedef float type; };
template <> struct id_to<MH_DOUBLE> { typedef double type; };
template <> struct id_to<MH_CFLOAT> { typedef std::complex<float> type; };
template <> struct id_to<MH_CDOUBLE> { typedef std::complex<double> type; };

/**
 * The primitive data type for lookup-table indices.
 */
typedef int32_t index_type;
static constexpr index_type MH_INDEX_SENTINEL = INT32_MAX;
static constexpr index_type INVALID_INDEX = MH_INDEX_SENTINEL;

/**
 * The primitive data type for element shape type.  May use only a single byte
 * but now take 4 bytes for legacy compatibility.
 */
typedef int32_t shape_type;

typedef double real_type;

constexpr static real_type MH_REAL_SENTINEL = -std::numeric_limits<real_type>::infinity();

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
