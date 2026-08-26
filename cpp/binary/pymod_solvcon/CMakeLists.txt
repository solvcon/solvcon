# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)
project(solvcon_pymod)

# NO_EXTRAS under USE_SANITIZER turns off pybind11's link-time optimization
# (/GL + /LTCG on MSVC), which AddressSanitizer cannot be combined with.
if(USE_SANITIZER AND MSVC)
    pybind11_add_module(
        _solvcon
        NO_EXTRAS
        module.cpp
    )
else()
    pybind11_add_module(
        _solvcon
        module.cpp
    )
endif()

target_link_libraries(
    _solvcon PUBLIC
    pybind11::module
    solvcon_entry
)

if (HIDE_SYMBOL)
    set_target_properties(_solvcon PROPERTIES CXX_VISIBILITY_PRESET "hidden")
else () # HIDE_SYMBOL
    set_target_properties(_solvcon PROPERTIES CXX_VISIBILITY_PRESET "default")
endif () # HIDE_SYMBOL

target_compile_options(
    _solvcon PRIVATE
    ${COMMON_COMPILER_OPTIONS}
)

if (CLANG_TIDY_EXE AND USE_CLANG_TIDY)
    set_target_properties(
        _solvcon PROPERTIES
        CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
    )
endif ()

add_custom_target(_solvcon_py ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:_solvcon>
        ${SOLVCON_PY_DIR}/../$<TARGET_FILE_NAME:_solvcon>
    DEPENDS _solvcon
    VERBATIM)

add_custom_target(remove_solvcon_py
    COMMAND ${CMAKE_COMMAND} -E rm -f
        ${SOLVCON_PY_DIR}/$<TARGET_FILE_NAME:_solvcon>
        ${SOLVCON_PY_DIR}/../$<TARGET_FILE_NAME:_solvcon>
    COMMENT "removing the built _solvcon extension"
    VERBATIM)

message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY ${SOLVCON_INCLUDE_DIR}/solvcon DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
