# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

# See https://google.github.io/googletest/quickstart-cmake.html for the
# recommended cmake configuration
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

if (APPLE)
    find_library(APPLE_FWK_ACCELERATE Accelerate REQUIRED)
endif () # APPLE

# The `test_nopython` target is only for testing the C++ interface of the non-Python part of the library.
add_executable(
    test_nopython
    test_nopython_buffer.cpp
    test_nopython_solvcon.cpp
    test_nopython_inout.cpp
    test_nopython_toggle.cpp
    test_nopython_radixtree.cpp
    test_nopython_callprofiler.cpp
    test_nopython_serializable.cpp
    test_nopython_transform.cpp
    test_nopython_rtree.cpp
    test_nopython_formatter.cpp
    test_nopython_mdspan.cpp
    test_nopython_multidim.cpp
    test_nopython_pilot_history.cpp
    test_nopython_pilot_syntax.cpp
    test_nopython_pilot_theme.cpp
    test_nopython_pilot_keymap.cpp
    ${SOLVCON_INCLUDE_DIR}/solvcon/pilot/console/RPythonConsoleHistory.cpp
    ${SOLVCON_INCLUDE_DIR}/solvcon/pilot/console/RPythonSyntaxRules.cpp
    ${SOLVCON_INCLUDE_DIR}/solvcon/pilot/theme/theme.cpp
    ${SOLVCON_INCLUDE_DIR}/solvcon/pilot/keymap.cpp
    ${SOLVCON_TOGGLE_SOURCES}
    ${SOLVCON_PROFILING_SOURCES}
    ${SOLVCON_BUFFER_SOURCES}
    ${SOLVCON_SERIALIZATION_SOURCES}
    ${SOLVCON_TRANSFORM_SOURCES}
    ${SOLVCON_SIMD_SOURCES}
    ${SOLVCON_MESH_SOURCES}
    ${SOLVCON_MULTIDIM_SOURCES}
)

target_link_libraries(
    test_nopython
    GTest::gtest_main
    GTest::gmock_main
    ${APPLE_FWK_ACCELERATE}
)

include(GoogleTest)
gtest_discover_tests(test_nopython)

add_custom_target(run_gtest
    COMMAND $<TARGET_FILE:test_nopython>
    DEPENDS test_nopython)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
