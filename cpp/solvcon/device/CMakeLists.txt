# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_DEVICE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/BufferBackend.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BufferDevice.hpp)

set(SOLVCON_DEVICE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/BufferBackend.cpp)

if (BUILD_METAL)
    add_subdirectory(metal)
    list(APPEND SOLVCON_DEVICE_HEADERS ${SOLVCON_METAL_HEADERS})
    list(APPEND SOLVCON_DEVICE_SOURCES ${SOLVCON_METAL_SOURCES})
endif () # BUILD_METAL

set(SOLVCON_DEVICE_HEADERS
    ${SOLVCON_DEVICE_HEADERS}
    CACHE FILEPATH "" FORCE)

set(SOLVCON_DEVICE_SOURCES
    ${SOLVCON_DEVICE_SOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
