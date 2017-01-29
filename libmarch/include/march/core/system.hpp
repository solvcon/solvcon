#pragma once

/*
 * Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * \file
 * Operation system setup.
 */

#include <cstdint>
#include <memory>
#include <iostream>
#include <string>
#include <cstdio>

// windows
#define MH_WIN64 0x00000001
#define MH_WIN32 0x00000002
// apple
#define MH_IOS_SIM 0x00000010
#define MH_IOS 0x00000020
#define MH_OSX 0x00000040
// linux
#define MH_LINUX 0x00000100
// other unix
#define MH_UNIX 0x00001000
// other POSIX
#define MH_POSIX 0x00010000

#ifdef _WIN32
    // define something for Windows (32-bit and 64-bit, this part is common)
    #ifdef _WIN64
        // define something for Windows (64-bit only)
        #define MH_OS MH_WIN64
    #else
        // define something for Windows (32-bit only)
        #define MH_OS MH_WIN32
    #endif
#elif __APPLE__
    #include "TargetConditionals.h"
    #if TARGET_IPHONE_SIMULATOR
        // iOS Simulator
        #define MH_OS MH_IOS_SIM
    #elif TARGET_OS_IPHONE
        // iOS device
        #define MH_OS MH_IOS
    #elif TARGET_OS_MAC
        // Other kinds of Mac OS
        #define MH_OS MH_OSX
    #else
    #   error "Unknown Apple platform"
    #endif
#elif __linux__
    // linux
    #define MH_OS MH_LINUX
#elif __unix__ // all unices not caught above
    // Unix
    #define MH_OS MH_UNIX
#elif defined(_POSIX_VERSION)
    // POSIX
    #define MH_OS MH_POSIX
#else
#   error "Unknown compiler"
#endif


#if MH_OS == MH_OSX
#include <xmmintrin.h>
#endif // MH_OS

namespace march {

inline void setup_debug() {

    // stop with floating-point exception.
#if MH_OS == MH_OSX
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif // MH_OS

}

inline void setup_system() {
}

template<typename ... Args>
std::string string_format(const std::string & format, Args ... args) {
    size_t size = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1;
    std::unique_ptr<char[]> buf(new char[size]); 
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1);
}

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
