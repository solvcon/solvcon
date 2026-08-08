# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_PYTYON_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/python.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/module.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_PYTHON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
    CACHE FILEPATH "" FORCE)

# `module.cpp` is the only unit outside the pilot whose text depends on Qt: it
# registers the pilot submodule and sets `HAS_PILOT` behind `QT_CORE_LIB`. It
# compiles into a library of its own so that the rest of the core can be built
# without ever seeing a Qt flag.
set(SOLVCON_ENTRY_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/module.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_PYTHON_FILES
    ${SOLVCON_PYTHON_HEADERS}
    ${SOLVCON_PYTHON_SOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
