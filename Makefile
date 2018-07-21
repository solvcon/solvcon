SHELL = /bin/bash
PYTHON = $(shell which python3)
LIBMARCH_PATH ?= libmarch
BUILD_DIR_NAME ?= opt_from_solvcon

CMAKE_BUILD_TYPE ?= Release
CMAKE_PASSTHROUGH ?=
VERBOSE ?=

build_dir = ${LIBMARCH_PATH}/build/${BUILD_DIR_NAME}

.PHONY: default
default: everything

.PHONY: everything
everything: build legacy

${build_dir}/Makefile:
	mkdir -p ${build_dir}
	cd ${build_dir} ; \
		cmake ../.. \
			-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
			-DPYTHON_EXECUTABLE:FILEPATH=${PYTHON} \
			-DMARCH_TEST=OFF \
			-DMARCH_DESTINATION=$(realpath .) \
			${CMAKE_PASSTHROUGH}

.PHONY: cmake
cmake: ${build_dir}/Makefile

.PHONY: clean_cmake
clean_cmake:
	rm -rf ${build_dir}

.PHONY: build
build: ${build_dir}/Makefile
	make -C ${build_dir} install VERBOSE=${VERBOSE}

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
	${PYTHON} setup.py clean

.PHONY: clean
clean: clean_legacy clean_current
