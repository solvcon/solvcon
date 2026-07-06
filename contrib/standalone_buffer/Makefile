# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

MMROOT := $(realpath ../..)
SETUPROOT ?= $(MMROOT)/$(BUILD_PATH)/standalone_buffer

ifeq ($(SETUPROOT),)
	SETUPROOT := local_standalone_buffer
endif

CFLAGS := \
	-std=c++23 -fPIC \
	-I$(SETUPROOT) \
	-I$(shell python3 -c 'import numpy as np; print(np.get_include())') \
	-I$(shell python3 -c 'import pybind11; print(pybind11.get_include())') \
	-I$(shell $(DIRNAME_PYTHON)/python3-config --prefix)/include \
	$(shell $(DIRNAME_PYTHON)/python3-config --cflags)

LDFLAGS := \
	-shared \
	-std=c++23 \
	-L$(shell $(DIRNAME_PYTHON)/python3-config --prefix)/lib \
	-lpython$(shell python3 -c 'import sys; print("%d.%d" % sys.version_info[0:2])') \
	$(shell $(DIRNAME_PYTHON)/python3-config --ldflags) \

MODPATH := $(SETUPROOT)/modbuf$(shell $(DIRNAME_PYTHON)/python3-config --extension-suffix)

.PHONY: build
build: $(MODPATH)

.PHONY: run
run: build
	env PYTHONPATH=$(SETUPROOT) python3 -c \
		"import modbuf; print(modbuf.SimpleArrayFloat64(3).ndarray)"

$(SETUPROOT)/wrap_ConcreteBuffer.o: $(SETUPROOT)/solvcon/buffer/pymod/wrap_ConcreteBuffer.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/wrap_SimpleArray.o: $(SETUPROOT)/solvcon/buffer/pymod/wrap_SimpleArray.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/wrap_SimpleArray_bool.o: $(SETUPROOT)/solvcon/buffer/pymod/wrap_SimpleArray_bool.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/wrap_SimpleArray_int.o: $(SETUPROOT)/solvcon/buffer/pymod/wrap_SimpleArray_int.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/wrap_SimpleArray_uint.o: $(SETUPROOT)/solvcon/buffer/pymod/wrap_SimpleArray_uint.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/wrap_SimpleArray_float.o: $(SETUPROOT)/solvcon/buffer/pymod/wrap_SimpleArray_float.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/wrap_SimpleArray_complex.o: $(SETUPROOT)/solvcon/buffer/pymod/wrap_SimpleArray_complex.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/wrap_SimpleArrayPlex.o: $(SETUPROOT)/solvcon/buffer/pymod/wrap_SimpleArrayPlex.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/buffer_pymod.o: $(SETUPROOT)/solvcon/buffer/pymod/buffer_pymod.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/BufferExpander.o: $(SETUPROOT)/solvcon/buffer/BufferExpander.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/SimpleArray.o: $(SETUPROOT)/solvcon/buffer/SimpleArray.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/blas_compat.o: $(SETUPROOT)/solvcon/math/blas_compat.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/simd_support.o: $(SETUPROOT)/solvcon/simd/simd_support.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/modbuf.o: modbuf.cpp
	c++ $(CFLAGS) -c $< -o $@

$(SETUPROOT)/radix_tree.o: $(SETUPROOT)/solvcon/profiling/RadixTree.cpp Makefile
	c++ $(CFLAGS) -c $< -o $@

OBJS = \
	$(SETUPROOT)/BufferExpander.o \
	$(SETUPROOT)/SimpleArray.o \
	$(SETUPROOT)/blas_compat.o \
	$(SETUPROOT)/wrap_ConcreteBuffer.o \
	$(SETUPROOT)/wrap_SimpleArray.o \
	$(SETUPROOT)/wrap_SimpleArray_bool.o \
	$(SETUPROOT)/wrap_SimpleArray_int.o \
	$(SETUPROOT)/wrap_SimpleArray_uint.o \
	$(SETUPROOT)/wrap_SimpleArray_float.o \
	$(SETUPROOT)/wrap_SimpleArray_complex.o \
	$(SETUPROOT)/wrap_SimpleArrayPlex.o \
	$(SETUPROOT)/buffer_pymod.o \
	$(SETUPROOT)/simd_support.o \
	$(SETUPROOT)/radix_tree.o \
	$(SETUPROOT)/modbuf.o

$(MODPATH): Makefile $(OBJS)
	c++ $(LDFLAGS) $(OBJS) -o $@

.PHONY: copy
copy:
	@echo "BUILD_PATH": $(BUILD_PATH)
	@echo "standalone buffer root": $(SETUPROOT)
	rm -rf $(SETUPROOT)/solvcon
	mkdir -p $(SETUPROOT)/solvcon
	cp $(MMROOT)/cpp/solvcon/base.hpp $(SETUPROOT)/solvcon/base.hpp
	cp -a $(MMROOT)/cpp/solvcon/buffer $(SETUPROOT)/solvcon/buffer
	mkdir -p $(SETUPROOT)/solvcon/python
	cp $(MMROOT)/cpp/solvcon/python/common.hpp $(SETUPROOT)/solvcon/python/common.hpp
	cp -a $(MMROOT)/cpp/solvcon/toggle $(SETUPROOT)/solvcon/toggle
	rm -rf $(SETUPROOT)/solvcon/toggle/pymod
	cp -a $(MMROOT)/cpp/solvcon/profiling $(SETUPROOT)/solvcon/profiling
	rm -rf $(SETUPROOT)/solvcon/profiling/pymod
	cp -a $(MMROOT)/cpp/solvcon/math $(SETUPROOT)/solvcon/math
	rm -rf $(SETUPROOT)/solvcon/math/pymod
	cp -a $(MMROOT)/cpp/solvcon/simd $(SETUPROOT)/solvcon/simd
	cp $(MMROOT)/cpp/solvcon/base.hpp $(SETUPROOT)/solvcon/
	find $(SETUPROOT) -name CMakeLists.txt -delete