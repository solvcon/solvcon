# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

if (BUILD_QT)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    find_package(Qt6 QUIET)
    if(NOT Qt6_FOUND)
        if(MSVC)
            message(FATAL_ERROR  "Qt6 Not Found.\nTry to add `C:\\Qt\\6.x.y\\msvcYYYY_64\\lib\\cmake` into \"Path\" of \"Enviroment Variables\"?")
        endif()
        if(APPLE)
            message(FATAL_ERROR  "Qt6 Not Found.\nTry `CMAKE_PREFIX_PATH=/path/to/qt/6.x.y/macos`?")
        endif()
        if(UNIX AND NOT APPLE)
            message(FATAL_ERROR  "Qt6 Not Found.\nTry `cmake -DCMAKE_PREFIX_PATH=/path/to/Qt/6.x.y/gcc_64/lib/cmake`?")
        endif()
        message(FATAL_ERROR  "Qt6 Not Found.") # any other platforms
    endif()
    find_package(Qt6 REQUIRED COMPONENTS Core)
    find_package(Qt6 REQUIRED COMPONENTS Widgets)
    find_package(Qt6 REQUIRED COMPONENTS Gui)
    # GuiPrivate exposes <rhi/qrhi.h>. Some prebuilt Qt packages ship the
    # public Gui module without the GuiPrivate CMake config, so treat it as
    # optional and fall back to the private header layout when linking below.
    find_package(Qt6 QUIET OPTIONAL_COMPONENTS GuiPrivate)
    # ShaderTools provides qt6_add_shaders() and the qsb tool.
    find_package(Qt6 REQUIRED COMPONENTS ShaderTools)
endif () # BUILD_QT

add_subdirectory(buffer)
add_subdirectory(mesh)
add_subdirectory(toggle)
add_subdirectory(profiling)
add_subdirectory(universe)
add_subdirectory(onedim)
add_subdirectory(oasis)
add_subdirectory(multidim)
add_subdirectory(python)
add_subdirectory(spacetime)
add_subdirectory(pilot)
add_subdirectory(inout)
add_subdirectory(serialization)
add_subdirectory(transform)
add_subdirectory(math)
add_subdirectory(linalg)
add_subdirectory(simd)

set(SOLVCON_ROOT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/solvcon.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/grid.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_ROOT_FILES
    ${SOLVCON_ROOT_HEADERS}
    CACHE FILEPATH "" FORCE)

if (BUILD_METAL)
    add_subdirectory(device/metal)
    set(SOLVCON_DEVICE_HEADERS
        ${SOLVCON_DEVICE_HEADERS}
        ${SOLVCON_METAL_HEADERS}
        CACHE FILEPATH "" FORCE)
    set(SOLVCON_DEVICE_SOURCES
        ${SOLVCON_DEVICE_SOURCES}
        ${SOLVCON_METAL_SOURCES}
        CACHE FILEPATH "" FORCE)
else () # BUILD_METAL
    set(SOLVCON_DEVICE_HEADERS
        CACHE FILEPATH "" FORCE)
    set(SOLVCON_DEVICE_SOURCES
        CACHE FILEPATH "" FORCE)
endif () #BUILD_METAL

set(SOLVCON_DEVICE_FILES
    ${SOLVCON_DEVICE_HEADERS}
    ${SOLVCON_DEVICE_SOURCES}
    CACHE FILEPATH "" FORCE)

set(SOLVCON_TERMINAL_FILES
    ${SOLVCON_ROOT_SOURCES}
    ${SOLVCON_BUFFER_FILES}
    ${SOLVCON_TOGGLE_FILES}
    ${SOLVCON_PROFILING_FILES}
    ${SOLVCON_UNIVERSE_FILES}
    ${SOLVCON_MESH_FILES}
    ${SOLVCON_MULTIDIM_FILES}
    ${SOLVCON_DEVICE_FILES}
    ${SOLVCON_ONEDIM_FILES}
    ${SOLVCON_OASIS_FILES}
    ${SOLVCON_SPACETIME_FILES}
    ${SOLVCON_PYTHON_FILES}
    ${SOLVCON_INOUT_FILES}
    ${SOLVCON_SERIALIZATION_FILES}
    ${SOLVCON_TRANSFORM_FILES}
    ${SOLVCON_MATH_FILES}
    ${SOLVCON_LINALG_FILES}
    ${SOLVCON_SIMD_FILES}
    CACHE FILEPATH "" FORCE)

set(SOLVCON_GRAPHIC_FILES
    ${SOLVCON_PILOT_FILES}
    CACHE FILEPATH "" FORCE)

if (BUILD_QT)
    qt_add_library(
        solvcon_primary
        STATIC
        ${SOLVCON_TERMINAL_FILES}
        ${SOLVCON_GRAPHIC_FILES}
    )

    target_link_libraries(
        solvcon_primary PUBLIC
        Qt::Widgets
        Qt::Core
        Qt::Gui
    )

    # <rhi/qrhi.h> lives in the QtGui private headers. Prefer the GuiPrivate
    # CMake target when present; otherwise add the private include directory
    # directly (the QRhi symbols themselves are exported from Qt::Gui).
    if (TARGET Qt6::GuiPrivate)
        target_link_libraries(solvcon_primary PUBLIC Qt::GuiPrivate)
    else ()
        get_target_property(SOLVCON_QTGUI_INCS Qt6::Gui INTERFACE_INCLUDE_DIRECTORIES)
        set(SOLVCON_FOUND_QRHI FALSE)
        foreach (SOLVCON_INC IN LISTS SOLVCON_QTGUI_INCS)
            if (EXISTS "${SOLVCON_INC}/${Qt6Gui_VERSION}/QtGui/rhi/qrhi.h")
                target_include_directories(solvcon_primary PRIVATE
                    "${SOLVCON_INC}/${Qt6Gui_VERSION}/QtGui")
                set(SOLVCON_FOUND_QRHI TRUE)
            endif ()
        endforeach ()
        if (NOT SOLVCON_FOUND_QRHI)
            message(FATAL_ERROR
                "Qt6::GuiPrivate is unavailable and <rhi/qrhi.h> could not be "
                "located in the QtGui private headers.")
        endif ()
    endif ()

    # Bake the pilot GLSL shaders to .qsb and embed them in the Qt resource
    # system under :/solvcon/pilot/shaders/. Loaded by RMaterial at runtime.
    qt6_add_shaders(solvcon_primary "solvcon_pilot_shaders"
        BATCHABLE
        PREFIX "/solvcon/pilot"
        BASE "${SOLVCON_PILOT_SHADERDIR}/.."
        FILES ${SOLVCON_PILOT_SHADERS}
    )

    target_link_libraries(
        solvcon_primary PUBLIC
        ${PYSIDE6_LIBFILE}
    )
else () # BUILD_QT
    add_library(
        solvcon_primary
        STATIC
        ${SOLVCON_TERMINAL_FILES}
    )
endif () # BUILD_QT

set_target_properties(solvcon_primary PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (CLANG_TIDY_EXE AND USE_CLANG_TIDY)
    set_target_properties(
        solvcon_primary PROPERTIES
        CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
    )
endif () # CLANG_TIDY

# Hack: add a .clang-tidy file in the generated .rcc directory.
# See https://gitlab.kitware.com/cmake/cmake/-/merge_requests/777
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/.rcc/.clang-tidy" "---
Checks: '-*,llvm-twine-local'")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/solvcon_primary_autogen/.clang-tidy" "---
Checks: '-bugprone-suspicious-include,llvm-twine-local'")

if (APPLE)
    find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
    find_library(APPLE_FWK_ACCELERATE Accelerate REQUIRED)
    find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
    find_library(APPLE_FWK_METAL Metal REQUIRED)

    target_link_libraries(
        solvcon_primary PUBLIC
        ${APPLE_FWK_FOUNDATION}
        ${APPLE_FWK_ACCELERATE}
        ${APPLE_FWK_QUARTZ_CORE}
        ${APPLE_FWK_METAL}
    )
    # Accelerate/vecLib provides BLAS + LAPACK.
    target_compile_definitions(solvcon_primary PUBLIC MM_HAS_VENDOR_LAPACK)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        target_compile_definitions(solvcon_primary PUBLIC MM_HAS_CBLAS)
    endif ()
endif () # APPLE

if (NOT APPLE)
    # The CBLAS header is needed for SimpleArray::matmul_blas() to use
    # the vendor BLAS backend.
    # Linux OpenBLAS packages also ship LAPACK symbols for EigenSystem.
    set(OPENBLAS_PATH_SUFFIXES
        openblas
        openblas-pthread
        openblas-openmp
        openblas-serial)
    if (CMAKE_LIBRARY_ARCHITECTURE)
        list(APPEND OPENBLAS_PATH_SUFFIXES
            ${CMAKE_LIBRARY_ARCHITECTURE}/openblas
            ${CMAKE_LIBRARY_ARCHITECTURE}/openblas-pthread
            ${CMAKE_LIBRARY_ARCHITECTURE}/openblas-openmp
            ${CMAKE_LIBRARY_ARCHITECTURE}/openblas-serial)
    endif ()
    find_library(OPENBLAS_LIB
        NAMES openblas libopenblas
        PATH_SUFFIXES ${OPENBLAS_PATH_SUFFIXES})
    find_path(OPENBLAS_INCLUDE_DIR
        NAMES cblas.h
        PATH_SUFFIXES ${OPENBLAS_PATH_SUFFIXES} OpenBLAS)
    if (OPENBLAS_LIB AND OPENBLAS_INCLUDE_DIR)
        message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
        target_include_directories(solvcon_primary PUBLIC ${OPENBLAS_INCLUDE_DIR})
        target_link_libraries(solvcon_primary PUBLIC ${OPENBLAS_LIB})
        target_compile_definitions(solvcon_primary PUBLIC MM_HAS_CBLAS)
    else ()
        message(STATUS "OpenBLAS/CBLAS not found; CBLAS matmul will not be built.")
    endif ()

    if (WIN32)
        find_package(LAPACK QUIET)
        if (LAPACK_FOUND)
            message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
            target_link_libraries(solvcon_primary PUBLIC LAPACK::LAPACK)
            target_compile_definitions(solvcon_primary PUBLIC MM_HAS_VENDOR_LAPACK)
        else ()
            message(STATUS "LAPACK not found; EigenSystem will not be built.")
        endif ()
    elseif (OPENBLAS_LIB)
        target_compile_definitions(solvcon_primary PUBLIC MM_HAS_VENDOR_LAPACK)
    endif ()
endif () # NOT APPLE

if (MSVC)
    target_compile_options(
        solvcon_primary PRIVATE
        ${COMMON_COMPILER_OPTIONS}
        /bigobj # C1128: number of sections exceeded object file format limit
    )
else () # MSVC
    target_compile_options(
        solvcon_primary PRIVATE
        ${COMMON_COMPILER_OPTIONS}
        -Wno-unused-value # for PYBIND11_EXPAND_SIDE_EFFECTS in pybind11.h
        -Wno-noexcept-type # GCC
        -Wno-elaborated-enum-base # for apple accelerate/veclib
    )
endif () # MSVC

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
