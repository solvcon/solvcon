# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING
#
# Common build command::
#
#   $ cmake -DCMAKE_BUILD_TYPE=Debug build/debug; make -C build/debug run_gtest
#   $ cmake -DCMAKE_BUILD_TYPE=Release build/release; make -C build/release run_gtest

cmake_minimum_required(VERSION 2.8.11)
project(libmarch)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

option(MARCH_RELAX_ERROR "Relax compiler error" OFF)

find_package(SCOTCH)

include_directories("include" ${SCOTCH_INCLUDE_DIRS})

# FIXME: turn on -Wextra
set(MARCH_WARNOPT "-Werror -Wall")
set(MARCH_WARNOPT "${MARCH_WARNOPT} -Wno-unused-private-field")
# FIXME: remove after port all legacy C code.
set(MARCH_WARNOPT "${MARCH_WARNOPT} -Wno-error=sign-compare")

if(MARCH_RELAX_ERROR)
    set(MARCH_WARNOPT "${MARCH_WARNOPT} -Wno-error=unused-variable")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC ${MARCH_WARNOPT}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_subdirectory(tests)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
