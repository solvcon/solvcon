# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_MCAP_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/mcap.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mcap_reader.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MCAP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/mcap_reader.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MCAP_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/mcap_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MCAP_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/mcap_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_McapReader.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MCAP_FILES
    ${SOLVCON_MCAP_HEADERS}
    ${SOLVCON_MCAP_SOURCES}
    ${SOLVCON_MCAP_PYMODHEADERS}
    ${SOLVCON_MCAP_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
