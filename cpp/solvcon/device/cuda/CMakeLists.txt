# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_CUDA_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/fft.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_impl.hpp
    CACHE FILEPATH "" FORCE)

# fft.hpp declares the CUDA entry points unconditionally. A CUDA build
# defines them in fft.cpp (host-compiled adapter) over fft_kernel.cu (the
# nvcc unit); any other build compiles the stub that throws, so no other
# directory needs to know whether CUDA is on.
if (BUILD_CUDA)
    set(SOLVCON_CUDA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fft_kernel.cu
        CACHE FILEPATH "" FORCE)
else () # BUILD_CUDA
    set(SOLVCON_CUDA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/fft_stub.cpp
        CACHE FILEPATH "" FORCE)
endif () # BUILD_CUDA

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
