# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_BUFFER_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BufferBase.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/small_vector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/signed_stride_layout.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/matmul_plan.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/matmul.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/SimpleArrayExecution.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ConcreteBuffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BufferExpander.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SimpleArray.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SimpleCollector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_BUFFER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/BufferExpander.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SimpleArray.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_BUFFER_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/array_common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/buffer_pymod.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_SimpleArray.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/TypeBroadcast.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/SimpleArrayCaster.hpp
    CACHE FILEPATH "" FORCE)

# The SimpleArray and SimpleCollector wrappers are instantiated for every
# dtype. Splitting the instantiations into per-dtype-family units lets the
# heavy pybind11 template work compile in parallel instead of serially in a
# single translation unit.
set(SOLVCON_BUFFER_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/buffer_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_ConcreteBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_SimpleArray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_SimpleArray_bool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_SimpleArray_int.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_SimpleArray_uint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_SimpleArray_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_SimpleArray_complex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_SimpleArrayPlex.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_BUFFER_FILES
    ${SOLVCON_BUFFER_HEADERS}
    ${SOLVCON_BUFFER_SOURCES}
    ${SOLVCON_BUFFER_PYMODHEADERS}
    ${SOLVCON_BUFFER_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
