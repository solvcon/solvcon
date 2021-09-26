SETUP_FILE ?= ./setup.mk

# The setup file may define RUNENV for runtime settings, e.g.
# RUNENV ?= DYLD_FALLBACK_LIBRARY_PATH=${DEVENVPREFIX}/lib
ifneq (,$(wildcard $(SETUP_FILE)))
	include $(SETUP_FILE)
endif

SHELL = /bin/bash
PYTHON_EXE := $(shell which python3 2>/dev/null)
PYTHON := env ${RUNENV} ${PYTHON_EXE}
PYTHON_VERSION_TO_MINOR := $(shell python3 -c 'from sys import version_info as v; print(str(v[0])+str(v[1]))')
NOSETESTS := $(shell which nosetests3 2>/dev/null)
ifeq (${NOSETESTS},)
	NOSETESTS := env ${RUNENV} PYTHONPATH=$(shell pwd) $(shell which python3 2>/dev/null) -m nose
endif

SC_PURE_PYTHON ?=
export SC_PURE_PYTHON

LIBMARCH_PATH ?= libmarch
BUILD_DIR_NAME ?= opt_from_solvcon_py${PYTHON_VERSION_TO_MINOR}

CMAKE_BUILD_TYPE ?= Release
CMAKE_PASSTHROUGH ?=
VERBOSE ?=

ifneq (${VERBOSE},)
	NOSETESTS := ${NOSETESTS} -v
endif

SCVER := $(shell env SC_PURE_PYTHON=1 ${PYTHON} -c 'import sys; import solvcon; sys.stdout.write("%s"%solvcon.__version__)')
SCVER_NUMONLY := $(shell echo ${SCVER} | tr -dC '[.0-9]')
PKGNAME := SOLVCON-${SCVER}

BUILD_DIR := ${LIBMARCH_PATH}/build/${BUILD_DIR_NAME}

PREFIX ?= $(realpath $(dir ${PYTHON})/../)
INSTALL_TO_DEBIAN ?=
ifeq (${INSTALL_TO_DEBIAN},)
	PYTHON_LIBRARY_DIR := ${PREFIX}/lib/$(shell ${PYTHON} -c "import sys; print('python%d.%d'%sys.version_info[:2])")/site-packages
else
	PYTHON_LIBRARY_DIR := ${PREFIX}/lib/$(shell ${PYTHON} -c "import sys; print('python%d'%sys.version_info[0])")/dist-packages
endif

.PHONY: default
default: build

.PHONY: build
build: libmarch legacy

${BUILD_DIR}/Makefile:
	mkdir -p ${BUILD_DIR}
	cd ${BUILD_DIR} ; \
	env $(RUNENV) \
		cmake ../.. \
		-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
		-DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXE} \
		-DMARCH_TEST=OFF \
		-DMARCH_DESTINATION=$(realpath .) \
		${CMAKE_PASSTHROUGH}

.PHONY: cmake
cmake: ${BUILD_DIR}/Makefile

.PHONY: clean_cmake
clean_cmake:
	rm -rf ${BUILD_DIR}

.PHONY: libmarch
libmarch: ${BUILD_DIR}/Makefile
	env $(RUNENV) \
		make -C ${BUILD_DIR} install VERBOSE=${VERBOSE}

.PHONY: clean_build
clean_build:
	rm -rf libmarch.*.so

.PHONY: clean_current
clean_current: clean_build clean_cmake

.PHONY: legacy
legacy:
	${PYTHON} setup.py build_ext --inplace

.PHONY: clean_legacy
clean_legacy:
	env SC_PURE_PYTHON=1 ${PYTHON} setup.py clean

dist/${PKGNAME}.tar.gz: Makefile
	rm -rf dist/SOLVCON-${SCVER}* ; \
	${PYTHON} setup.py clean ; \
	${PYTHON} setup.py sdist

.PHONY: package
package: dist/${PKGNAME}.tar.gz

dist/${PKGNAME}/make.log: dist/${PKGNAME}.tar.gz
	if [[ ! -d "dist" ]] ; then \
		echo "fatal error: dist doesn't exist" ; \
		exit 1 ; \
	fi ; \
	cd dist ; \
	rm -rf ${PKGNAME}/ ; \
	tar xfz ${PKGNAME}.tar.gz ; \
	cd ${PKGNAME} ; \
	make 2>&1 | tee make.log

.PHONY: build_from_package
build_from_package: dist/${PKGNAME}/make.log

.PHONY: test_from_package
test_from_package: dist/${PKGNAME}/make.log
	cd dist/${PKGNAME} ; \
	PYTHONPATH=`pwd` ; \
	${NOSETESTS} --with-doctest

.PHONY: test
test:
	${NOSETESTS} --with-doctest

.PHONY: clean_package
clean_package:
	rm -rf dist/${PKGNAME}*

.PHONY: install
install: build
	echo PREFIX=${PREFIX}
	echo PYTHON_LIBRARY_DIR=${PYTHON_LIBRARY_DIR}
	mkdir -p ${PYTHON_LIBRARY_DIR} ; cp libmarch.*.so ${PYTHON_LIBRARY_DIR}
	${PYTHON} setup.py install \
		--prefix=${PREFIX} \
		--install-lib=${PYTHON_LIBRARY_DIR} \
		--install-data=${PREFIX}

.PHONY: deb
deb: dist/${PKGNAME}.tar.gz
	mkdir -p dist/debbuild ; cd dist/debbuild ; \
	ln -s ../${PKGNAME}.tar.gz solvcon_${SCVER_NUMONLY}.orig.tar.gz ; \
	tar xfz ../${PKGNAME}.tar.gz ; \
	cd ${PKGNAME} ; \
	dpkg-buildpackage -rfakeroot -uc -us 2>&1 | tee ../buildpackage.log

.PHONY: clean
clean: clean_legacy clean_current clean_package
