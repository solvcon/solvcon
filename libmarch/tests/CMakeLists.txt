# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 2.8.11)
project(test_libmarch)

link_directories(${libmarch_BINARY_DIR}/tests)

include_directories(${test_libmarch_SOURCE_DIR})

add_executable(test_libmarch
    main.cpp gtest/gtest-all.cc
    core_types.cpp
    core_Buffer.cpp
    mesh_LookupTable.cpp
)
target_link_libraries(test_libmarch stdc++ pthread)

if (NOT TESTFILTER)
    set(TESTFILTER "*")
endif()

add_custom_target(debug_gtest lldb test_libmarch DEPENDS test_libmarch)
add_custom_target(run_gtest test_libmarch --gtest_filter=${TESTFILTER} DEPENDS test_libmarch)

install(TARGETS test_libmarch DESTINATION ${CMAKE_BINARY_DIR})

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
