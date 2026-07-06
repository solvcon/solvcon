# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_PROFILING_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/profile.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RadixTree.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SerializableProfiler.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_PROFILING_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/RadixTree.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_PROFILING_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/profiling_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_PROFILING_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/profiling_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_profile.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_PROFILING_FILES
    ${SOLVCON_PROFILING_HEADERS}
    ${SOLVCON_PROFILING_SOURCES}
    ${SOLVCON_PROFILING_PYMODHEADERS}
    ${SOLVCON_PROFILING_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
