# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

# Register the Python suites with CTest so that ctest is the one test entry
# point on every platform, alongside the C++ cases that gtest_discover_tests
# registers. Each suite carries its own environment through a test property,
# which is what a per-suite value needs: a configure preset's environment map
# would have to give both suites the same answer.

# SKIP_PYTHON_EXECUTABLE leaves PYTHON_EXECUTABLE unset, and the suite has no
# interpreter to run under without it.
if(PYTHON_EXECUTABLE)
    add_test(
        NAME pytest
        COMMAND ${PYTHON_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/tests
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    set_tests_properties(pytest PROPERTIES
        LABELS "python"
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}"
    )
else()
    message(STATUS "PYTHON_EXECUTABLE is unset; not registering the pytest suite")
endif()

if(BUILD_QT)
    # The pilot runs the same suite inside its own embedded interpreter. On
    # Linux it needs the shiboken6 shared library on the loader path, which
    # the workflow used to export by hand.
    add_test(
        NAME pilot_pytest
        COMMAND $<TARGET_FILE:pilot> --mode=pytest
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    set(SOLVCON_PILOT_TEST_ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}")
    if(NOT "${SHIBOKEN6_PYTHON_PACKAGE_PATH}" STREQUAL "")
        list(APPEND SOLVCON_PILOT_TEST_ENVIRONMENT
            "LD_LIBRARY_PATH=${SHIBOKEN6_PYTHON_PACKAGE_PATH}")
    endif()
    set_tests_properties(pilot_pytest PROPERTIES
        LABELS "pilot"
        ENVIRONMENT "${SOLVCON_PILOT_TEST_ENVIRONMENT}"
    )
endif()

# SOLVCON_TEST_SHOW_WINDOWS is deliberately absent from both properties.
# tests/conftest.py keeps the GUI windows off the screen unless it is set, and
# a test property would override whatever the caller chose. Leaving it to the
# environment keeps the developer default off the screen and lets a runner,
# which has no desktop to take over, export it for the run.

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
