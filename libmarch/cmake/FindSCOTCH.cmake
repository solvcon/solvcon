# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING
#
# .. cmake_module:: SCOTCH
#
#    Simplistic cmake module for SCOTCH.  It sets the folllowing variables:
#
#    .. cmake_variable:: SCOTCH_INCLUDE_DIRS
#
#       The directory containing the SCOTCH header files.
#
#    .. cmake_variable:: SCOTCH_LIBRARIES
#
#       Libraries for SCOTCH.
#
#    .. cmake_variable:: SCOTCH_FOUND
#
#       True when both SCOTCH_INCLUDE_DIRS and SCOTCH_LIBRARIES are found.
#
# .. cmake_variable:: SCOTCH_ROOT
#
#    Additional search path for SCOTCH header and library files.

if(NOT DEFINED SCOTCH_ROOT AND DEFINED ENV{SCOTCH_ROOT})
    set(SCOTCH_ROOT $ENV{SCOTCH_ROOT})
endif()

message("SCOTCH_ROOT is ${SCOTCH_ROOT}")

find_path(SCOTCH_INCLUDE_DIRS NAMES scotch.h
    PATHS ${SCOTCH_ROOT} /usr/include/scotch #"$ENV{CONDA_PREFIX}"
    PATH_SUFFIXES include
    DOC "Include directory of SCOTCH."
)

macro(_scotch_search_lib lvar lname ldoc)
  find_library(${lvar} ${lname}
    PATHS ${SCOTCH_ROOT} #"$ENV{CONDA_PREFIX}"
    PATH_SUFFIXES lib
    DOC "${ldoc}")
endmacro(_scotch_search_lib)

_scotch_search_lib(SCOTCH_LIBRARY scotch "SCOTCH library.")
_scotch_search_lib(SCOTCHERR_LIBRARY scotcherr "SCOTCH error library.")
_scotch_search_lib(SCOTCHERREXIT_LIBRARY scotcherrexit "SCOTCH error exit library.")
_scotch_search_lib(SCOTCHMETIS_LIBRARY scotchmetis "SCOTCH metis library.")

set(SCOTCH_LIBRARIES ${SCOTCHMETIS_LIBRARY} ${SCOTCH_LIBRARY} ${SCOTCHERR_LIBRARY} ${SCOTCHERREXIT_LIBRARY})

if(SCOTCH_INCLUDE_DIRS AND SCOTCH_LIBRARIES)
  set(SCOTCH_FOUND true)
endif()

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
