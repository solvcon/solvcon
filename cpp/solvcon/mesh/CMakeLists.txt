# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_MESH_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StaticMesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DelaunayBuilder.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MESH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/StaticMesh_boundary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StaticMesh_interior.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DelaunayBuilder.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MESH_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/mesh_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MESH_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/mesh_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_StaticGrid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_StaticMesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_DelaunayBuilder.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MESH_FILES
    ${SOLVCON_MESH_HEADERS}
    ${SOLVCON_MESH_SOURCES}
    ${SOLVCON_MESH_PYMODHEADERS}
    ${SOLVCON_MESH_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
