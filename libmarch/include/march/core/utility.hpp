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

#if !defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#   if __cplusplus >= 201402L
#       define MARCH_CPP14
#   endif
#elif defined(_MSC_VER)
#   if _MSVC_LANG >= 201402L
#       define MARCH_CPP14
#   endif
#endif

namespace march {

#ifdef MARCH_CPP14
using std::make_unique;
#else // MARCH_CPP14
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif // MARCH_CPP14

namespace detail {

inline static constexpr size_t log2(size_t n, int k = 0) { return (n <= 1) ? k : log2(n >> 1, k + 1); }

} /* end namespace detail */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
