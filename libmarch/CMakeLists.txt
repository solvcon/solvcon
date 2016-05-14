# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 2.8.11)
project(libmarch)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

include_directories("include")

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-unused-private-field -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")

add_subdirectory(tests)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
