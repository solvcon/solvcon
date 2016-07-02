# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 2.8.11)
project(libmarch)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

option(MARCH_RELAX_ERROR "Relax compiler error" OFF)

include_directories("include")

set(NOWARN_MARCH "-Wno-unused-private-field")

if(MARCH_RELAX_ERROR)
    set(NOWARN_MARCH "${NOWARN_MARCH} -Wno-error=unused-variable")
endif()

set(CMAKE_CXX_FLAGS "-std=c++14 -fPIC -Werror -Wall ${NOWARN_MARCH}")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(tests)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
