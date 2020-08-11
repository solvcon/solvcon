# Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 3.7)
project(march)

cmake_policy(SET CMP0069 NEW)

#add_subdirectory(pybind11)
find_package(pybind11 REQUIRED)
find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIR})
set(MARCH_PY_SOURCES
    march.cpp
    march_gas.cpp
)
pybind11_add_module(libmarch ${MARCH_PY_SOURCES})
target_link_libraries(libmarch PRIVATE ${SCOTCH_LIBRARIES})
install(TARGETS libmarch DESTINATION ${MARCH_DESTINATION})

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
