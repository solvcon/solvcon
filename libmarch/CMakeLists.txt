# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING
#
# Common build command::
#
#   $ cmake CMAKE_BUILD_TYPE=Debug build/debug; make -C build/debug run_gtest
#   $ cmake CMAKE_BUILD_TYPE=Release build/release; make -C build/release run_gtest
#
# To create a Xcode project::
#
#   $ mkdir -p build/xcode; cd build/xcode; cmake -G Xcode ../..

cmake_minimum_required(VERSION 2.8.11)
project(libmarch)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")
include(FindPythonAnaconda)

option(MARCH_RELAX_ERROR "Relax compiler error" OFF)
option(SOLVCON_ROOT "SOLVCON root path" OFF)

find_package(SCOTCH)

include_directories("include" ${SCOTCH_INCLUDE_DIRS})

set(MARCH_WARNOPT "-Werror -Wall -Wextra")
set(MARCH_WARNOPT "${MARCH_WARNOPT} -Wno-unused-function -Wno-unreachable-code -Wno-unused-private-field -Wno-missing-braces -Wno-sign-compare")
set(MARCH_WARNOPT "${MARCH_WARNOPT} -Wno-error=deprecated-declarations")
# FIXME: remove after port all legacy C code.
set(MARCH_WARNOPT "${MARCH_WARNOPT} -Wno-error=sign-compare")

if(MARCH_RELAX_ERROR)
    set(MARCH_WARNOPT "${MARCH_WARNOPT} -Wno-error=unused-variable")
endif()

if(NOT SOLVCON_ROOT)
    set(SOLVCON_ROOT ${CMAKE_INSTALL_PREFIX})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC ${MARCH_WARNOPT}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(MARCH_HEADERS
    # overall
    include/march/march.hpp
    # dependency
    include/march/depend/scotch.hpp
    # core
    include/march/core/core.hpp
    include/march/core/Buffer.hpp
    include/march/core/LookupTable.hpp
    include/march/core/Vector.hpp
    include/march/core/types.hpp
    include/march/core/utility.hpp
    # mesh
    include/march/mesh/mesh.hpp
    include/march/mesh/BasicCE.hpp
    include/march/mesh/BoundaryData.hpp
    include/march/mesh/CompoundCE.hpp
    include/march/mesh/UnstructuredBlock/UnstructuredBlock.hpp
    include/march/mesh/UnstructuredBlock/class.hpp
    include/march/mesh/UnstructuredBlock/build_csr.hpp
    include/march/mesh/UnstructuredBlock/build_faces_from_cells.hpp
    include/march/mesh/UnstructuredBlock/build_rcells.hpp
    include/march/mesh/UnstructuredBlock/calc_ce.hpp
    include/march/mesh/UnstructuredBlock/calc_sf.hpp
    include/march/mesh/UnstructuredBlock/calc_metric.hpp
    include/march/mesh/UnstructuredBlock/fill_ghost.hpp
    # gas
    include/march/gas/gas.hpp
    include/march/gas/Solution.hpp
    include/march/gas/Jacobian.hpp
    include/march/gas/Solver_decl.hpp
    include/march/gas/Solver.hpp
    include/march/gas/derivative.hpp
    include/march/gas/calc_cfl.hpp
    include/march/gas/calc_soln.hpp
    include/march/gas/calc_solt.hpp
    include/march/gas/Quantity.hpp
    include/march/gas/Trim.hpp
)
string(REPLACE "include/" "${CMAKE_CURRENT_SOURCE_DIR}/include/"
       MARCH_HEADERS "${MARCH_HEADERS}")

add_subdirectory(src)
add_subdirectory(tests)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
