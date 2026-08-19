# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_TIMESERIES_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/timeseries.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_TIMESERIES_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/timeseries_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_TIMESERIES_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/timeseries_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_timeseries.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_TIMESERIES_FILES
    ${SOLVCON_TIMESERIES_HEADERS}
    ${SOLVCON_TIMESERIES_PYMODHEADERS}
    ${SOLVCON_TIMESERIES_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
