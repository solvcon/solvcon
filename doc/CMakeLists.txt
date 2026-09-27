# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

# Every build tree writes to doc/build, the directory doc/Makefile uses and the
# one conf.py reads the Doxygen XML from, so there is one documentation build.
set(SOLVCON_DOC_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")

option(SOLVCON_DOC_WARNINGS_AS_ERRORS "fail the doc build on a Sphinx warning" OFF)
message(STATUS "SOLVCON_DOC_WARNINGS_AS_ERRORS: ${SOLVCON_DOC_WARNINGS_AS_ERRORS}")
set(SOLVCON_SPHINX_OPTS)
if(SOLVCON_DOC_WARNINGS_AS_ERRORS)
    # Sphinx 8.1 and later always keep going; an older one stops at the first
    # warning without the flag.
    list(APPEND SOLVCON_SPHINX_OPTS -W --keep-going)
endif()

# Sphinx runs under the build interpreter because that is the one an activated
# environment provides doc/requirements.txt for.
add_custom_target(doc
    COMMAND "${PYTHON_EXECUTABLE}" -m sphinx -b html ${SOLVCON_SPHINX_OPTS}
        "${CMAKE_CURRENT_SOURCE_DIR}/source" "${SOLVCON_DOC_BUILD_DIR}/html"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building the HTML documentation in ${SOLVCON_DOC_BUILD_DIR}/html"
    USES_TERMINAL
    VERBATIM)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(doc_doxygen
        COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${SOLVCON_DOC_BUILD_DIR}/doxygen"
        COMMAND Doxygen::doxygen Doxyfile
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Generating the C++ API XML in ${SOLVCON_DOC_BUILD_DIR}/doxygen"
        VERBATIM)
    add_dependencies(doc doc_doxygen)
else()
    message(STATUS "doxygen not found; the doc target leaves the C++ API empty")
endif()

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
