# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

add_library(
    solvcon_cuda_probe
    STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda.cu
)
target_link_libraries(solvcon_cuda_probe PRIVATE CUDA::cudart)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
