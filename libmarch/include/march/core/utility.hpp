#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * \file
 * Utilities.
 */

#include <utility>
#include <memory>

#include <cstdio>
#include <sstream>
#include <iostream>
#include <iomanip>

#if !defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#   if __cplusplus >= 201402L
#       define MH_CPP14
#   endif
#elif defined(_MSC_VER)
#   if _MSVC_LANG >= 201402L
#       define MH_CPP14
#   endif
#endif

namespace march {

#ifdef MH_CPP14
using std::make_unique;
#else // MH_CPP14
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif // MH_CPP14

namespace detail {

inline static constexpr size_t log2(size_t n, int k = 0) { return (n <= 1) ? k : log2(n >> 1, k + 1); }

} /* end namespace detail */

template<typename ElementType>
void fill_sentinel(ElementType *arr, size_t nelem, ElementType sentinel) {
    std::fill(arr, arr + nelem, sentinel);
}

template<typename ElementType>
void fill_sentinel(ElementType *arr, size_t nelem) {
    if (true == std::is_floating_point<ElementType>::value) {
        fill_sentinel(arr, nelem, std::numeric_limits<ElementType>::quiet_NaN());
    } else if (true == std::is_arithmetic<ElementType>::value) {
        char * carr = reinterpret_cast<char *>(arr);
        fill_sentinel(carr, nelem*sizeof(ElementType), static_cast<char>(-1));
    } else {
        throw std::runtime_error("cannot fill sentinel for unsupported type");
    }
}

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
