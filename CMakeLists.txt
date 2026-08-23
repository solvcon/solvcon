# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)
project(solvcon)

enable_testing()

# The presets and the Makefile always state a build type. A configure that
# names neither, which is a bare `cmake -S . -B <dir>`, would otherwise get an
# empty one and compile with no optimization, no NDEBUG, and no debug info.
# A multi-config generator carries its types in CMAKE_CONFIGURATION_TYPES and
# ignores this one.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "build type" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# The Python, Qt6, and PySide6 that MSVC links ship release binaries only, so a
# Debug build must match their runtime and configuration or it corrupts the heap
# (issue #1058). Both default on; turn one off to link debug dependencies when
# they are available.
option(SOLVCON_MSVC_FORCE_RELEASE_RUNTIME
    "MSVC: use the release C runtime (/MD) in every config" ON)
option(SOLVCON_MSVC_FORCE_RELEASE_IMPORTS
    "MSVC: import dependencies from their release config in a Debug build" ON)

if(MSVC)
    # "MultiThreadedDLL" selects /MD for linking. If not specifying it, a
    # debug build will choose "MultiThreadedDebugDLL" for /MDd that uses a
    # different STL layout and crashes the debug build. See
    # https://cmake.org/cmake/help/latest/variable/CMAKE_MSVC_RUNTIME_LIBRARY.html
    # The debug build will use /Zi to keep it still debuggable.
    if(SOLVCON_MSVC_FORCE_RELEASE_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    endif()

    # Without this a Debug pilot links the debug Qt (Qt6Cored.dll) while PySide6
    # loads the release Qt6Core.dll; two Qt runtimes then run at once and a
    # QString freed across the /MDd and /MD split corrupts the heap.
    if(SOLVCON_MSVC_FORCE_RELEASE_IMPORTS)
        set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG "Release;")
    endif()
endif()

# ccache is detected here rather than named in the shared presets, so a host
# without it configures and builds unchanged. The guard skips every case where
# the wrap cannot pay: an explicit launcher, such as the sccache of the
# ci-win-mkl preset; a compiler that is already ccache, as a masquerade symlink
# ahead of the real one on PATH makes it, where wrapping would hash ccache in
# place of the real compiler; and MSVC, whose default debug information format
# ccache refuses, leaving every unit to pay the wrap and return no hit.
option(USE_CCACHE "use ccache as the compiler launcher when it is available" ON)
message(STATUS "USE_CCACHE: ${USE_CCACHE}")
file(REAL_PATH "${CMAKE_CXX_COMPILER}" CCACHE_CXX_COMPILER_REALPATH)
if(USE_CCACHE
        AND NOT MSVC
        AND NOT CMAKE_C_COMPILER_LAUNCHER
        AND NOT CMAKE_CXX_COMPILER_LAUNCHER
        AND NOT CCACHE_CXX_COMPILER_REALPATH MATCHES "/s?ccache(\\.exe)?$")
    # An uninstalled ccache leaves its path cached, and every compile then fails
    # on a launcher that is no longer there.
    if(CCACHE_EXE AND NOT EXISTS "${CCACHE_EXE}")
        unset(CCACHE_EXE CACHE)
    endif()
    find_program(
        CCACHE_EXE
        NAMES "ccache"
        DOC "Path to ccache executable"
    )
    if(CCACHE_EXE)
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_EXE}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_EXE}")
        message(STATUS "use ccache: ${CCACHE_EXE}")
    else()
        message(STATUS "not use ccache: ccache not found")
    endif()
endif()

if(NOT SKIP_PYTHON_EXECUTABLE)
    find_program(PYTHON_EXECUTABLE python3)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

# PYBIND11_FINDPYTHON makes pybind11 reuse this result instead of the retired
# FindPythonLibs. Development.Embed is for the interpreter the pilot embeds.
set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED
    COMPONENTS Interpreter Development.Module Development.Embed NumPy)

# The PySide6 and shiboken6 discovery below needs an interpreter even under
# SKIP_PYTHON_EXECUTABLE.
if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
endif()
message(STATUS "use PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")

option(BUILD_QT "build with QT" ON)
message(STATUS "BUILD_QT: ${BUILD_QT}")

# The pilot renderer knowingly includes <rhi/qrhi.h> from the QtGui private
# headers.
set(QT_NO_PRIVATE_MODULE_WARNING ON)

option(BUILD_METAL "build with Metal" OFF)
message(STATUS "BUILD_METAL: ${BUILD_METAL}")
if(BUILD_METAL)
    add_compile_options(-DSOLVCON_METAL)
endif()

# The mcap subsystem reads MCAP recordings, and lz4 and zstd decompress their
# chunks. No other part of solvcon needs those two libraries, so the option
# keeps them out of a build that does not read recordings. The option starts
# OFF, so an existing build picks up no new hard dependency.
option(BUILD_MCAP "build with MCAP recording support" OFF)
message(STATUS "BUILD_MCAP: ${BUILD_MCAP}")
if(BUILD_MCAP)
    set(MCAP_PREFIX_HINTS "$ENV{HOMEBREW_PREFIX}" /opt/homebrew /usr/local)
    find_library(LZ4_LIB NAMES lz4 liblz4
        HINTS ${MCAP_PREFIX_HINTS} PATH_SUFFIXES lib)
    find_path(LZ4_INCLUDE_DIR NAMES lz4frame.h
        HINTS ${MCAP_PREFIX_HINTS} PATH_SUFFIXES include)
    find_library(ZSTD_LIB NAMES zstd libzstd
        HINTS ${MCAP_PREFIX_HINTS} PATH_SUFFIXES lib)
    find_path(ZSTD_INCLUDE_DIR NAMES zstd.h
        HINTS ${MCAP_PREFIX_HINTS} PATH_SUFFIXES include)
    if(LZ4_LIB AND LZ4_INCLUDE_DIR AND ZSTD_LIB AND ZSTD_INCLUDE_DIR)
        message(STATUS "Found lz4: ${LZ4_LIB}")
        message(STATUS "Found zstd: ${ZSTD_LIB}")
    else()
        message(FATAL_ERROR
            "BUILD_MCAP is ON but lz4 and zstd were not both found.\n"
            "Run contrib/dependency/show-mcap-dependency.sh for the commands "
            "that install them, or configure with BUILD_MCAP=OFF.")
    endif()
endif()

option(USE_CLANG_TIDY "use clang-tidy" OFF)
option(LINT_AS_ERRORS "clang-tidy warnings as errors" OFF)
message(STATUS "LINT_AS_ERRORS: ${LINT_AS_ERRORS}")

# Sanitizers
# - Undefined Behavior Sanitization: Preventing the unpredictable outcomes of
#   undefined behavior. See
#   https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
# - Memory Leaks Sanitization: Prevententing memory leaks caused from inproper
#   memory management. See https://clang.llvm.org/docs/MemorySanitizer.html
# - Address Sanitization: Protecting against address-related vulnerabilities.
#   See https://clang.llvm.org/docs/AddressSanitizer.html
option(USE_SANITIZER "use sanitizer (undefined, leak, address)" OFF)


# PUI-related code has been migrated to PySide6, so it doesn't need to be installed by default.
# Yet, you can still install it by setting the INSTALL_PUI option to ON.
option(INSTALL_PUI "install PUI" OFF)
message(STATUS "INSTALL_PUI : ${INSTALL_PUI}")
if(INSTALL_PUI)
    if(NOT BUILD_QT)
        message(WARNING "INSTALL_PUI ON while BUILD_QT OFF")
    endif()
    execute_process(
        COMMAND python3 PUI_install.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/thirdparty
    )
endif()

find_program(
    CLANG_TIDY_EXE
    NAMES "clang-tidy"
    DOC "Path to clang-tidy executable"
)
if(CLANG_TIDY_EXE)
    file(REAL_PATH "${CLANG_TIDY_EXE}" CLANG_TIDY_REALPATH)
    get_filename_component(
        CLANG_TIDY_BINDIR
        "${CLANG_TIDY_REALPATH}"
        DIRECTORY
    )
    find_program(
        CLANG_TIDY_DIFF_EXE
        NAMES "clang-tidy-diff.py"
        HINTS
            "${CLANG_TIDY_BINDIR}"
            "${CLANG_TIDY_BINDIR}/../share/clang"
        DOC "Path to clang-tidy-diff.py executable"
    )
endif()

if(USE_CLANG_TIDY)
    # On macOS, SDK-provided headers and frameworks are resolved through an
    # SDK sysroot. Homebrew clang-tidy may not inherit Xcode's selected SDK
    # when replaying compile_commands.json, so ask xcrun for the macOS SDK
    # path and pass it to Clang as -isysroot.
    # References:
    #   xcrun --show-sdk-path: https://keith.github.io/xcode-man-pages/xcrun.1.html
    #   Clang -isysroot: https://clang.llvm.org/docs/ClangCommandLineReference.html
    #   clang-tidy --extra-arg: https://clang.llvm.org/extra/clang-tidy/
    if(APPLE)
        execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            OUTPUT_VARIABLE CLANG_TIDY_APPLE_SYSROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    if(CLANG_TIDY_EXE)
        set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}"
            "-header-filter=/cpp/.\*"
            "--extra-arg=-std=c++23"
            # FIXME: Remove --allow-no-checks in the future after figuring out
            # why the error message "Error: no checks enabled." appears with
            # some files, after upgrading to clang-tidy-22.
            # Related information (but not what we see in solvcon):
            # https://github.com/llvm/llvm-project/issues/192713
            "--allow-no-checks")
        if(APPLE AND CLANG_TIDY_APPLE_SYSROOT)
            list(APPEND DO_CLANG_TIDY
                "--extra-arg=-isysroot"
                "--extra-arg=${CLANG_TIDY_APPLE_SYSROOT}")
        endif()
        if(LINT_AS_ERRORS)
            set(DO_CLANG_TIDY "${DO_CLANG_TIDY}" "-warnings-as-errors=*")
        endif()
        message(STATUS "use clang-tidy with DO_CLANG_TIDY: ${DO_CLANG_TIDY}")
    else()
        message(FATAL_ERROR "USE_CLANG_TIDY is on but clang-tidy is not found")
    endif()

    if(CLANG_TIDY_DIFF_EXE)
        set(DO_CLANG_TIDY_DIFF "${CLANG_TIDY_DIFF_EXE}"
            "-p1"
            "-extra-arg=-std=c++23"
            "-allow-no-checks")
        if(APPLE AND CLANG_TIDY_APPLE_SYSROOT)
            list(APPEND DO_CLANG_TIDY_DIFF
                "-extra-arg=-isysroot"
                "-extra-arg=${CLANG_TIDY_APPLE_SYSROOT}")
        endif()
        if(LINT_AS_ERRORS)
            set(DO_CLANG_TIDY_DIFF
                "${DO_CLANG_TIDY_DIFF}"
                "-warnings-as-errors=*")
        endif()
        message(
            STATUS
            "use clang-tidy-diff.py with DO_CLANG_TIDY_DIFF: ${DO_CLANG_TIDY_DIFF}"
        )
    endif()
else()
    message(STATUS "not use clang-tidy")
endif()

if(USE_SANITIZER)
    message(STATUS "use sanitizer")
    add_compile_options(-DUSE_SANITIZER)
    if(MSVC)
        # MSVC ships only AddressSanitizer; it has no leak or undefined
        # sanitizer, and it does not accept the GCC/Clang
        # -fsanitize=undefined,leak,address spelling (it drops it with
        # warning D9002). Use its own /fsanitize=address. ASan is incompatible
        # with the /GL whole-program optimization that pybind11 turns on for
        # the module (see cpp/binary/pymod_solvcon/CMakeLists.txt) and with
        # incremental linking, so those are disabled alongside it. /Zi emits
        # the compiler debug info ASan needs to symbolize its reports; without
        # it MSVC raises warning C5072, which /WX turns into a build error.
        add_compile_options(/fsanitize=address /Zi)
        add_link_options(/INCREMENTAL:NO)
    else()
        set(SANITIZER_LIST "undefined,leak,address")
        add_compile_options(-fsanitize=${SANITIZER_LIST})
        add_link_options(-fsanitize=${SANITIZER_LIST})
    endif()
else()
    message(STATUS "not use sanitizer")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands")

option(HIDE_SYMBOL "hide the symbols of python wrapper" OFF)
message(STATUS "HIDE_SYMBOL: ${HIDE_SYMBOL}")

option(SOLVCON_PROFILE "enable profiler" OFF)
message(STATUS "SOLVCON_PROFILE: ${SOLVCON_PROFILE}")
if(SOLVCON_PROFILE)
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DSOLVCON_PROFILE")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSOLVCON_PROFILE")
    endif()
endif()

if(BUILD_METAL)
include_directories(SYSTEM "${CMAKE_CURRENT_LIST_DIR}/thirdparty/install/metal-cpp")
endif()

# Pybind11 2.12.0 is the first version that supports compiling for numpy 2.0.0.
# Therefore, it is required to use pybind11 2.12.0 or later.
# Reference: https://github.com/pybind/pybind11/releases/tag/v2.12.0

option(pybind11_path "pybind11 path")
find_package(pybind11 2.12.0 REQUIRED PATHS ${pybind11_path})
message(STATUS "pybind11_INCLUDE_DIRS: ${pybind11_INCLUDE_DIRS}")
# SYSTEM keeps the core's warning flags off pybind11's own headers, the same
# reason numpy gets it below.
include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})

set(NUMPY_INCLUDE_DIR "${Python_NumPy_INCLUDE_DIRS}")
message(STATUS "NUMPY_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR}")
# SYSTEM keeps the core's warning flags off numpy's own headers, which do not
# build clean under `/W4` and are not ours to fix.
include_directories(SYSTEM ${NUMPY_INCLUDE_DIR})

set(SOLVCON_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/cpp" CACHE INTERNAL "")

include_directories(${SOLVCON_INCLUDE_DIR})

# Discovery has to use PYTHON_EXECUTABLE rather than a bare python3.
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import os, PySide6; print(os.path.dirname(PySide6.__file__))"
    OUTPUT_VARIABLE PYSIDE6_PYTHON_PACKAGE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "PYSIDE6_PYTHON_PACKAGE_PATH: ${PYSIDE6_PYTHON_PACKAGE_PATH}")
# Addressing an issue with the PySide6/Shiboken6 6.11 homebrew package:
# homebrew exposes the PySide6 package as a tree of symlinks; the real
# libpyside6.* dylibs live in the formula's lib/ directory (three levels
# above the resolved package directory). Compute that fallback so the
# homebrew layout can be supported in addition to the bundled PyPI layout.
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import os, PySide6; print(os.path.join(os.path.dirname(os.path.realpath(PySide6.__file__)), '..', '..', '..'))"
    OUTPUT_VARIABLE PYSIDE6_FALLBACK_LIBDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "PYSIDE6_FALLBACK_LIBDIR: ${PYSIDE6_FALLBACK_LIBDIR}")

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import os, shiboken6; print(os.path.dirname(shiboken6.__file__))"
    OUTPUT_VARIABLE SHIBOKEN6_PYTHON_PACKAGE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "SHIBOKEN6_PYTHON_PACKAGE_PATH: ${SHIBOKEN6_PYTHON_PACKAGE_PATH}")
# For PySide6/Shiboken6 homebrew package.
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import os, shiboken6; print(os.path.join(os.path.dirname(os.path.realpath(shiboken6.__file__)), '..', '..', '..'))"
    OUTPUT_VARIABLE SHIBOKEN6_FALLBACK_LIBDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "SHIBOKEN6_FALLBACK_LIBDIR: ${SHIBOKEN6_FALLBACK_LIBDIR}")
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import os, shiboken6_generator; print(os.path.dirname(shiboken6_generator.__file__))"
    OUTPUT_VARIABLE SHIBOKEN6_GENERATOR_PYTHON_PACKAGE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "SHIBOKEN6_GENERATOR_PYTHON_PACKAGE_PATH: ${SHIBOKEN6_GENERATOR_PYTHON_PACKAGE_PATH}")

if(NOT "${PYSIDE6_PYTHON_PACKAGE_PATH}" STREQUAL "")
    include_directories("${PYSIDE6_PYTHON_PACKAGE_PATH}/include")
    link_directories("${PYSIDE6_PYTHON_PACKAGE_PATH}")
endif()
if(NOT "${PYSIDE6_FALLBACK_LIBDIR}" STREQUAL "" AND IS_DIRECTORY "${PYSIDE6_FALLBACK_LIBDIR}")
    link_directories("${PYSIDE6_FALLBACK_LIBDIR}")
endif()
if(NOT "${SHIBOKEN6_GENERATOR_PYTHON_PACKAGE_PATH}" STREQUAL "")
    include_directories("${SHIBOKEN6_GENERATOR_PYTHON_PACKAGE_PATH}/include")
endif()
if(NOT "${SHIBOKEN6_PYTHON_PACKAGE_PATH}" STREQUAL "")
    link_directories("${SHIBOKEN6_PYTHON_PACKAGE_PATH}")
endif()
if(NOT "${SHIBOKEN6_FALLBACK_LIBDIR}" STREQUAL "" AND IS_DIRECTORY "${SHIBOKEN6_FALLBACK_LIBDIR}")
    link_directories("${SHIBOKEN6_FALLBACK_LIBDIR}")
endif()

# PySide6 and Shiboken6 library name are different in Windows and Unix-like OS
# therefore needs sperate in different cases
if(MSVC)
    file(GLOB PYSIDE6_LIBFILE LIST_DIRECTORIES false "${PYSIDE6_PYTHON_PACKAGE_PATH}/pyside6.*.lib")
    file(GLOB SHIBOKEN6_LIBFILE LIST_DIRECTORIES false "${SHIBOKEN6_PYTHON_PACKAGE_PATH}/shiboken6.*.lib")
else()
    file(GLOB PYSIDE6_LIBFILE LIST_DIRECTORIES false "${PYSIDE6_PYTHON_PACKAGE_PATH}/libpyside6.*")
    if("${PYSIDE6_LIBFILE}" STREQUAL "" AND NOT "${PYSIDE6_FALLBACK_LIBDIR}" STREQUAL "")
        file(GLOB PYSIDE6_LIBFILE LIST_DIRECTORIES false "${PYSIDE6_FALLBACK_LIBDIR}/libpyside6.*")
    endif()
    file(GLOB SHIBOKEN6_LIBFILE LIST_DIRECTORIES false "${SHIBOKEN6_PYTHON_PACKAGE_PATH}/libshiboken6.*")
    if("${SHIBOKEN6_LIBFILE}" STREQUAL "" AND NOT "${SHIBOKEN6_FALLBACK_LIBDIR}" STREQUAL "")
        file(GLOB SHIBOKEN6_LIBFILE LIST_DIRECTORIES false "${SHIBOKEN6_FALLBACK_LIBDIR}/libshiboken6.*")
    endif()
endif()
message(STATUS "PYSIDE6_LIBFILE: ${PYSIDE6_LIBFILE}")
message(STATUS "SHIBOKEN6_LIBFILE: ${SHIBOKEN6_LIBFILE}")

# An empty PYSIDE6_LIBFILE links solvcon_graphic against nothing and defers the
# failure to an undefined PySide::getWrapperForQObject at link time.
if(BUILD_QT AND "${PYSIDE6_LIBFILE}" STREQUAL "")
    message(FATAL_ERROR
        "libpyside6 not found through ${PYTHON_EXECUTABLE}. Point "
        "PYTHON_EXECUTABLE at an interpreter that can import PySide6, or "
        "configure with BUILD_QT=OFF.")
endif()

# add_subdirectory() copies this scope's variables into the subdirectory as it
# is called, so this must stay above every add_subdirectory() that consumes it.
# Set it below one and the subdirectory sees nothing, with no diagnostic.
if(MSVC)
    set(COMMON_COMPILER_OPTIONS /W4 /bigobj /WX)
else()
    set(COMMON_COMPILER_OPTIONS -Wall -Wextra)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # gcc computes -Wmaybe-uninitialized after inlining, so at -O3 -g it
        # fires inside pybind11 on a tuple that is in fact initialized, and
        # -O3 -g is what a bare `make` configures. Demote rather than disable,
        # so a real uninitialized use still reaches the log.
        set(COMMON_COMPILER_OPTIONS ${COMMON_COMPILER_OPTIONS} -Wno-error=maybe-uninitialized)
    endif()
    if(USE_CLANG_TIDY AND LINT_AS_ERRORS)
        message(STATUS "Disable -Werror because clang-tidy (with lint as errors) is enabled")
    else()
        set(COMMON_COMPILER_OPTIONS ${COMMON_COMPILER_OPTIONS} -Werror)
        message(STATUS "Enable -Werror because clang-tidy (with lint as errors) is disabled")
    endif()
endif()

add_subdirectory(cpp/solvcon)

include(GNUInstallDirs)

set(SOLVCON_PY_DIR "${PROJECT_SOURCE_DIR}/solvcon")

add_subdirectory(cpp/binary/pymod_solvcon)
if(BUILD_QT)
add_subdirectory(cpp/binary/pilot)
endif()

set(USE_GOOGLETEST True CACHE BOOL "Build with googletest")
message(STATUS "USE_GOOGLETEST: ${USE_GOOGLETEST}")

if(USE_GOOGLETEST)
add_subdirectory(gtests)
endif() # USE_GOOGLETEST

add_subdirectory(tests)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
