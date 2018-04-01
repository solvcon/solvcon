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
#include <string>

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

namespace string {

// FIXME: va_list format string checking
template<typename ... Args>
std::string format(const std::string & format, Args ... args) {
    size_t size = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1;
    std::unique_ptr<char[]> buf(new char[size]); 
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1);
}

inline std::string create_indented_newline(size_t indent) {
    std::string indented_newline("\n");
    while (indent) { indented_newline += " "; indent--; }
    return indented_newline;
}

inline std::string replace_all_substrings(
    std::string subject, std::string const & source, std::string const & target
) {
    std::string::size_type n = 0;
    while ((n = subject.find(source, n)) != std::string::npos) {
        subject.replace(n, source.size(), target);
        n += target.size();
    }
    return subject;
}

inline std::string from_double(double value, size_t precision=0) {
    std::ostringstream os;
    os.setf(std::ios::left);
    if (precision) {
        os.setf(std::ios::scientific);
        os.precision(precision);
    }
    os << value;
    return os.str();
}

} /* end namespace string */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
