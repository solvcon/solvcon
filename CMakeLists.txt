# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see LICENSE.txt

cmake_minimum_required(VERSION 3.7)
project(solvcon)

option(SOLVCON_ROOT "SOLVCON root path" OFF)
if(NOT SOLVCON_ROOT)
    get_filename_component(SOLVCON_ROOT "solvcon" ABSOLUTE)
endif()

add_subdirectory(libmarch)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
