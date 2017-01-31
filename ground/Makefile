# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

export SCDEP?=$(shell cd .. ; pwd)/build/dep
export SCDL?=$(SCDEP)/downloaded
ifeq ($(shell uname), Darwin)
export NP?=$(shell sysctl -n hw.ncpu)
else ifeq ($(shell uname), Linux)
export NP?=$(shell cat /proc/cpuinfo | grep processor | wc -l)
else
export NP?=1
endif


PYTHON=$(SCDEP)/bin/python
PIP=$(SCDEP)/bin/pip

.PHONY: $(ALIASES)

default: all


ALIASES += openssl
TARGET_openssl := $(SCDEP)/lib/libssl.a
TARGETS += $(TARGET_openssl)

$(TARGET_openssl): build-openssl.sh
	./$<

openssl: $(TARGET_openssl)


ALIASES += python
TARGET_python := $(SCDEP)/bin/python3.6m
TARGETS += $(TARGET_python)

$(TARGET_python): build-python.sh openssl
	./$<

python: $(TARGET_python)


ALIASES += atlas
TARGET_atlas := $(SCDEP)/lib/libatlas.a
TARGETS += $(TARGET_atlas)

$(TARGET_atlas): build-atlas.sh
	./$<

atlas: $(TARGET_atlas)


ALIASES += numpy
TARGET_numpy := $(SCDEP)/lib/python3.6/site-packages/numpy/__init__.py
TARGETS += $(TARGET_numpy)

$(TARGET_numpy): build-numpy.sh atlas
	./$<

numpy: $(TARGET_numpy)


ALIASES += scotch
TARGET_scotch := $(SCDEP)/lib/libscotch.a
TARGETS += $(TARGET_scotch)

$(TARGET_scotch): build-scotch.sh
	./$<

scotch: $(TARGET_scotch)


ALIASES += hdf
TARGET_hdf := $(SCDEP)/bin/h5ls
TARGETS += $(TARGET_hdf)

$(TARGET_hdf): build-hdf.sh
	./$<

hdf: $(TARGET_hdf)


ALIASES += netcdf
TARGET_netcdf := $(SCDEP)/bin/nc-config
TARGETS += $(TARGET_netcdf)

$(TARGET_netcdf): build-netcdf.sh hdf
	./$<

netcdf: $(TARGET_netcdf)


ALIASES += cmake
TARGET_cmake := $(SCDEP)/bin/cmake
TARGETS += $(TARGET_cmake)

$(TARGET_cmake): build-cmake.sh
	./$<

cmake: $(TARGET_cmake)


ALIASES += gmp
TARGET_gmp := $(SCDEP)/lib/libgmp.a
TARGETS += $(TARGET_gmp)

$(TARGET_gmp): build-gmp.sh
	./$<

gmp: $(TARGET_gmp)


ALIASES += gmsh
TARGET_gmsh := $(SCDEP)/bin/gmsh
TARGETS += $(TARGET_gmsh)

$(TARGET_gmsh): build-gmsh.sh cmake gmp numpy
	./$<

gmsh: $(TARGET_gmsh)


ALIASES += vtk
TARGET_vtk := $(SCDEP)/bin/vtkpython
TARGETS += $(TARGET_vtk)

$(TARGET_vtk): build-vtk.sh cmake
	./$<

vtk: $(TARGET_vtk)


ALIASES += pytools

pytools: python
	$(PIP) install nose sphinx


all: $(ALIASES)

# vim: set noet nobomb ff=unix fenc=utf8:
