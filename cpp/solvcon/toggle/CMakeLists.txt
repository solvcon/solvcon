# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_TOGGLE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/toggle.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_TOGGLE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/toggle.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_TOGGLE_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/toggle_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_TOGGLE_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/toggle_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_Toggle.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_TOGGLE_FILES
    ${SOLVCON_TOGGLE_HEADERS}
    ${SOLVCON_TOGGLE_SOURCES}
    ${SOLVCON_TOGGLE_PYMODHEADERS}
    ${SOLVCON_TOGGLE_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
