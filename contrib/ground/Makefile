# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.

SHELL := /bin/bash


export SCGROUND=$(shell pwd)
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


.PHONY: default
default: prerequisite_dev


TARGET_openssl := $(SCDEP)/bin/openssl
TARGETS += $(TARGET_openssl)

$(TARGET_openssl): build-openssl.sh
	time ./$<
	ls -al $@
	cp -f $(SCGROUND)/start $(SCDEP)/start

.PHONY: openssl
openssl: $(TARGET_openssl)


ifeq ($(SCDEBUG), 1)
TARGET_python := $(SCDEP)/bin/python3.6dm
else
TARGET_python := $(SCDEP)/bin/python3.6m
endif
TARGETS += $(TARGET_python)
PREREQUISITE_DEV += $(TARGET_python)

$(TARGET_python): build-python.sh $(TARGET_openssl)
	time ./$<
	ls -al $@
	cp -f $(SCGROUND)/start $(SCDEP)/start

.PHONY: python
python: $(TARGET_python)


TARGET_atlas := $(SCDEP)/lib/libatlas.a
TARGETS += $(TARGET_atlas)

$(TARGET_atlas): build-atlas.sh
	time ./$<
	ls -al $@
	cp -f $(SCGROUND)/start $(SCDEP)/start

.PHONY: atlas
atlas: $(TARGET_atlas)


TARGET_numpy := $(SCDEP)/lib/python3.6/site-packages/numpy-1.12.0-py3.6.egg-info/PKG-INFO
TARGETS += $(TARGET_numpy)
PREREQUISITE_DEV += $(TARGET_numpy)

PROVIDE_numpy := build-numpy.sh $(TARGET_python)
ifeq ($(shell uname), Linux)
PROVIDE_numpy += $(TARGET_atlas)
endif

$(TARGET_numpy): $(PROVIDE_numpy)
	time ./$<
	ls -al $@
	cp -f $(SCGROUND)/start $(SCDEP)/start

.PHONY: numpy
numpy: $(TARGET_numpy)


TARGET_scotch := $(SCDEP)/lib/libscotch.a
TARGETS += $(TARGET_scotch)
PREREQUISITE_DEV += $(TARGET_scotch)

$(TARGET_scotch): build-scotch.sh
	time ./$<
	ls -al $@
	cp -f $(SCGROUND)/start $(SCDEP)/start

.PHONY: scotch
scotch: $(TARGET_scotch)


TARGET_hdf := $(SCDEP)/bin/h5ls
TARGETS += $(TARGET_hdf)

$(TARGET_hdf): build-hdf.sh
	time ./$<
	ls -al $@
	cp -f $(SCGROUND)/start $(SCDEP)/start

.PHONY: hdf
hdf: $(TARGET_hdf)


TARGET_netcdf := $(SCDEP)/bin/nc-config
TARGETS += $(TARGET_netcdf)
PREREQUISITE_DEV += $(TARGET_netcdf)

$(TARGET_netcdf): build-netcdf.sh $(TARGET_hdf)
	time ./$<
	ls -al $@
	cp -f $(SCGROUND)/start $(SCDEP)/start

.PHONY: netcdf
netcdf: $(TARGET_netcdf)


TARGET_cmake := $(SCDEP)/bin/cmake
TARGETS += $(TARGET_cmake)
PREREQUISITE_DEV += $(TARGET_cmake)

$(TARGET_cmake): build-cmake.sh
	time ./$<
	ls -al $@
	cp -f $(SCGROUND)/start $(SCDEP)/start

.PHONY: cmake
cmake: $(TARGET_cmake)


TARGET_gmp := $(SCDEP)/lib/libgmp.a
TARGETS += $(TARGET_gmp)

$(TARGET_gmp): build-gmp.sh
	time ./$<
	ls -al $@
	cp -f $(SCGROUND)/start $(SCDEP)/start

.PHONY: gmp
gmp: $(TARGET_gmp)


TARGET_gmsh := $(SCDEP)/bin/gmsh
TARGETS += $(TARGET_gmsh)
PREREQUISITE_DEV += $(TARGET_gmsh)

$(TARGET_gmsh): build-gmsh.sh $(TARGET_cmake} $(TARGET_gmp) $(TARGET_numpy)
	time ./$<
	ls -al $@
	cp -f $(SCGROUND)/start $(SCDEP)/start

.PHONY: gmsh
gmsh: $(TARGET_gmsh)


TARGET_graphviz := $(SCDEP)/bin/dot
TARGETS += $(TARGET_graphviz)
PREREQUISITE_DEV += $(TARGET_graphviz)

$(TARGET_graphviz): build-graphviz.sh
	time ./$<
	ls -al $@
	cp -f $(SCGROUND)/start $(SCDEP)/start

.PHONY: graphviz
graphviz: $(TARGET_graphviz)


TARGET_vtk := $(SCDEP)/bin/vtkpython
TARGETS += $(TARGET_vtk)
PREREQUISITE_DEV += $(TARGET_vtk)

$(TARGET_vtk): build-vtk.sh $(TARGET_cmake)
	time ./$<
	ls -al $@
	cp -f $(SCGROUND)/start $(SCDEP)/start

.PHONY: vtk
vtk: $(TARGET_vtk)


.PHONY: pytools
pytools: $(TARGET_python)
	$(PIP) install -U \
		nose six paramiko boto cython netcdf4 \
		ipython jupyter pythreejs \
		sphinx sphinxcontrib-issuetracker \
		https://github.com/pybind/pybind11/archive/master.zip
	cp -f $(SCGROUND)/start $(SCDEP)/start


.PHONY: all
all: $(TARGETS)

.PHONY: prerequisite_dev
prerequisite_dev: $(PREREQUISITE_DEV)

.PHONY: touchall
touchall:
	touch $(TARGETS)

.PHONY: extra
extra: pytools

# vim: set noet nobomb ff=unix fenc=utf8:
