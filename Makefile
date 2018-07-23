SHELL = /bin/bash
PYTHON := $(shell which python3)
NOSETESTS := $(shell which nosetests3)
ifeq (${NOSETESTS},)
	NOSETESTS := $(shell which nosetests)
endif

SC_PURE_PYTHON ?=
export SC_PURE_PYTHON

LIBMARCH_PATH ?= libmarch
BUILD_DIR_NAME ?= opt_from_solvcon

CMAKE_BUILD_TYPE ?= Release
CMAKE_PASSTHROUGH ?=
VERBOSE ?=

ifneq (${VERBOSE},)
	NOSETESTS := ${NOSETESTS} -v
endif

SCVER := $(shell env SC_PURE_PYTHON=1 ${PYTHON} -c 'import sys; import solvcon; sys.stdout.write("%s"%solvcon.__version__)')
PKGNAME := SOLVCON-${SCVER}

BUILD_DIR := ${LIBMARCH_PATH}/build/${BUILD_DIR_NAME}

.PHONY: default
default: build legacy

.PHONY: everything
everything: build legacy package

${BUILD_DIR}/Makefile:
	mkdir -p ${BUILD_DIR}
	cd ${BUILD_DIR} ; \
	cmake ../.. \
		-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
		-DPYTHON_EXECUTABLE:FILEPATH=${PYTHON} \
		-DMARCH_TEST=OFF \
		-DMARCH_DESTINATION=$(realpath .) \
		${CMAKE_PASSTHROUGH}

.PHONY: cmake
cmake: ${BUILD_DIR}/Makefile

.PHONY: clean_cmake
clean_cmake:
	rm -rf ${BUILD_DIR}

.PHONY: build
build: ${BUILD_DIR}/Makefile
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

.PHONY: clean_package
clean_package:
	rm -rf dist/${PKGNAME}*

.PHONY: clean
clean: clean_legacy clean_current clean_package
