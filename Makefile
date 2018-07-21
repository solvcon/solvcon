SHELL = /bin/bash
LIBMARCH_PATH ?= libmarch
BUILD_DIR_NAME ?= opt_from_solvcon

CMAKE_BUILD_TYPE ?= Release
CMAKE_PASSTHROUGH ?=
VERBOSE ?=

build_dir = ${LIBMARCH_PATH}/build/${BUILD_DIR_NAME}

.PHONY: default
default: build

${build_dir}/Makefile:
	mkdir -p ${build_dir}
	cd ${build_dir} ; \
		cmake ../.. \
			-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
			-DPYTHON_EXECUTABLE:FILEPATH=`which python3` \
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

.PHONY: clean
clean: clean_build clean_cmake
