# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_METAL_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/metal.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_METAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/metal.mm
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
