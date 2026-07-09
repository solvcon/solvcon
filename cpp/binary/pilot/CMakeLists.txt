# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)
project(pilot LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# get the root of the modmdesh project
set(PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR}/../../../)

if(NOT DEFINED INSTALL_PILOTDIR)
    set(INSTALL_PILOTDIR "pilot")
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 COMPONENTS Core)
find_package(Qt6 COMPONENTS Widgets)
find_package(Qt6 COMPONENTS Gui)

if(APPLE)
    set(MACOSX_BUNDLE_ICON_FILE solvcon.icns)
    set(app_icon_macos ${PROJECT_ROOT_DIR}/resources/pilot/solvcon.icns)
    set_source_files_properties(${app_icon_macos} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

    qt_add_executable(
        pilot
        MACOSX_BUNDLE
        pilot.cpp
        ${app_icon_macos}
    )
else()
    qt_add_executable(
        pilot
        pilot.cpp
    )
endif()

target_link_libraries(
    pilot PUBLIC
    pybind11::embed
    solvcon_primary
)
if(WIN32)
    set_target_properties(
        pilot
        PROPERTIES
        LINK_FLAGS_DEBUG "/SUBSYSTEM:CONSOLE" # open the console for debug mode
    )
endif(WIN32)

qt_add_resources(
    pilot "app_icon"
    PREFIX "/"
    BASE ${PROJECT_ROOT_DIR}/resources/pilot
    FILES
    ${PROJECT_ROOT_DIR}/resources/pilot/icon.ico
)

if(HIDE_SYMBOL)
    set_target_properties(pilot PROPERTIES CXX_VISIBILITY_PRESET "hidden")
else()
    set_target_properties(pilot PROPERTIES CXX_VISIBILITY_PRESET "default")
endif()

target_compile_options(
    pilot PRIVATE
    ${COMMON_COMPILER_OPTIONS}
)

if(APPLE)
    # pilot.cpp pulls in apple's accelerate/veclib headers, whose CF_ENUM
    # macros trip clang-diagnostic-elaborated-enum-base, and the lint job
    # runs clang-tidy with -warnings-as-errors=*. solvcon_primary already
    # suppresses this; the pilot target needs it too. The library's other
    # -Wno flags are not needed here: pilot.cpp lints clean on Linux
    # without them, so this stays apple-only.
    target_compile_options(
        pilot PRIVATE
        -Wno-elaborated-enum-base # for apple accelerate/veclib
    )
endif()

if(CLANG_TIDY_EXE AND USE_CLANG_TIDY)
    set_target_properties(
        pilot PROPERTIES
        CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
    )
endif()

# Hack: add a .clang-tidy file in the generated .rcc directory.
# See https://gitlab.kitware.com/cmake/cmake/-/merge_requests/777
file(WRITE "${pilot_BINARY_DIR}/.rcc/.clang-tidy" "---
Checks: '-*,llvm-twine-local'")
file(WRITE "${pilot_BINARY_DIR}/pilot_autogen/.clang-tidy" "---
Checks: '-bugprone-suspicious-include,llvm-twine-local'")

set_target_properties(
    pilot PROPERTIES
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE TRUE
)

if(APPLE)
    find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
    find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
    find_library(APPLE_FWK_METAL Metal REQUIRED)

    target_link_libraries(
        pilot PUBLIC
        ${APPLE_FWK_FOUNDATION}
        ${APPLE_FWK_QUARTZ_CORE}
        ${APPLE_FWK_METAL}
    )
endif()

add_custom_target(run_pilot $<TARGET_FILE:pilot>)
add_custom_target(run_pilot_pytest $<TARGET_FILE:pilot> --mode=pytest)

if(DO_CLANG_TIDY_DIFF)
    get_target_property(
        SOLVCON_PRIMARY_FILES
        solvcon_primary
        SOURCES
    )

    string(
        CONCAT
        PILOT_CLANG_TIDY_DIFF_COMMAND
        "test -n \"$SOLVCON_DIFF_BASE\""
        " && "
        "git diff --no-color -U0 \"$SOLVCON_DIFF_BASE\" HEAD"
        " -- $<JOIN:${SOLVCON_PRIMARY_FILES};${CMAKE_CURRENT_SOURCE_DIR}, >"
        " | "
        "$<JOIN:${DO_CLANG_TIDY_DIFF}, > -path \"${CMAKE_BINARY_DIR}\""
    )

    add_custom_target(
        pilot_clang_tidy_diff
        COMMAND
            bash -e -o pipefail -c "${PILOT_CLANG_TIDY_DIFF_COMMAND}"
        WORKING_DIRECTORY ${PROJECT_ROOT_DIR}
        VERBATIM
    )
endif()

install(TARGETS pilot
    RUNTIME DESTINATION "${INSTALL_PILOTDIR}"
    BUNDLE DESTINATION "${INSTALL_PILOTDIR}"
    LIBRARY DESTINATION "${INSTALL_PILOTDIR}"
)
