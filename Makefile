# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

# Build solvcon Python extension (even when the timestamp is clean):
#   make
# Build verbosely:
#   make VERBOSE=1
# Build with clang-tidy
#   make USE_CLANG_TIDY=ON
# Build without debug symbols, which compiles faster:
#   make CMAKE_BUILD_TYPE=Release

SETUP_FILE ?= ./setup.mk

ifneq (,$(wildcard $(SETUP_FILE)))
	include $(SETUP_FILE)
endif

# Optional extension appended to the auto-computed BUILD_PATH
# (e.g. BUILD_PATH_EXT=_noqt).
BUILD_PATH_EXT ?=

# To workaround macos SIP: https://github.com/solvcon/solvcon/pull/16.
# Additional configuration can be loaded from SETUP_FILE.
RUNENV += PYTHONPATH=$(SOLVCON_ROOT)

# The configure knobs are written down once, in CMakePresets.json; see
# doc/source/devguide/cmake.md. A knob set on the command line, in the
# environment, or in setup.mk is layered on the selected preset as a -D flag,
# and nothing else is passed, so a make build and a preset build stay the same
# build. Setting a knob to its preset value therefore costs nothing.
CMAKE_KNOBS = SKIP_PYTHON_EXECUTABLE HIDE_SYMBOL SOLVCON_PROFILE \
	BUILD_METAL BUILD_QT USE_CLANG_TIDY LINT_AS_ERRORS USE_GOOGLETEST \
	USE_SANITIZER USE_CCACHE CMAKE_INSTALL_PREFIX \
	CMAKE_LIBRARY_OUTPUT_DIRECTORY CMAKE_PREFIX_PATH
CMAKE_OVERRIDES = $(strip $(foreach knob,$(CMAKE_KNOBS), \
	$(if $(filter-out undefined default,$(origin $(knob))),\
	-D$(knob)=$($(knob)))))

# Debugging is the default story. Release drops the symbols for a session
# that is not going to open a debugger.
CMAKE_BUILD_TYPE ?= RelWithDebInfo
# The build type picks the preset and the build tree, and it needs no -D of
# its own because the preset states it. One table decides both, so the two
# cannot drift apart. An unsupported build type leaves them empty, which the
# configure rule below reports; the check waits until then so that a target
# needing no build tree, `make lint` or `make cmakeclean`, still runs.
CMAKE_PRESET_Debug = dev-dbg
CMAKE_PRESET_Release = dev-rel
CMAKE_PRESET_RelWithDebInfo = dev-reldbg
BUILD_TAG_Debug = dbg
BUILD_TAG_Release = rel
BUILD_TAG_RelWithDebInfo = reldbg
# Override CMAKE_PRESET to build another preset, for instance one of your own
# from CMakeUserPresets.json. Set BUILD_PATH with it: the tree is named after
# CMAKE_BUILD_TYPE, which a preset of another build type would contradict.
CMAKE_PRESET ?= $(CMAKE_PRESET_$(CMAKE_BUILD_TYPE))
# Number of online processors. Drives both build parallelism (MAKE_PARALLEL
# below) and the lint targets. getconf works on both Linux and macOS; fall
# back to 1 if unavailable. Override to cap parallelism, e.g. NPROC=2.
NPROC ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
MAKE_PARALLEL ?= -j $(NPROC)
SOLVCON_ROOT ?= $(shell pwd)
# Set CMAKE_PREFIX_PATH to make it easier to build with Qt, e.g.,
# CMAKE_PREFIX_PATH=/path/to/qt/6.2.3/macos. It is one of the knobs above, so
# it reaches CMake only when it is set.
CMAKE_ARGS ?=
VERBOSE ?=
FORCE_CLANG_FORMAT ?=
RELEASE_OUTPUT ?= $(SOLVCON_ROOT)/build
RELEASE_ARTIFACT ?= $(RELEASE_OUTPUT)/pilot.dmg
RELEASE_ARGS ?=

# Let CMake find vcpkg-provided OpenBLAS/LAPACK headers, import libraries, and
# package metadata during configure and link on Windows.
ifeq ($(OS),Windows_NT)
VCPKG_INSTALLATION_ROOT ?= C:/vcpkg
CMAKE_TOOLCHAIN_FILE ?= $(VCPKG_INSTALLATION_ROOT)/scripts/buildsystems/vcpkg.cmake
CMAKE_ARGS += -DCMAKE_TOOLCHAIN_FILE=$(CMAKE_TOOLCHAIN_FILE)
CMAKE_ARGS += -DVCPKG_TARGET_TRIPLET=x64-windows
endif

# !!! NOTE: USING ANY VENV IS STRONGLY DISCOURAGED IN DEVELOPING SOLVCON !!!
# This treatment is a "smarter" way to find python3-config executable.
# In case Python is not system Python. For example. Python virtual environment
# is used.
# However, please note a Python virtual environment is strongly discouraged in
# developing solvcon. We do not actively resolve bugs related to any virtual
# env including venv or conda.
# See https://github.com/solvcon/solvcon/pull/177 for more details.
WHICH_PYTHON := $(shell which python3)
REALPATH_PYTHON := $(realpath $(WHICH_PYTHON))
export DIRNAME_PYTHON := $(dir $(REALPATH_PYTHON))

pyvminor := $(shell python3 -c 'import sys; print("%d%d" % sys.version_info[0:2])')

BUILD_PATH ?= build/$(BUILD_TAG_$(CMAKE_BUILD_TYPE))$(pyvminor)$(BUILD_PATH_EXT)
export BUILD_PATH

# Test with the build interpreter; an ABI-tagged _solvcon cannot load under a
# py.test-3 launcher bound to a different Python.
PYTEST ?= $(WHICH_PYTHON) -m pytest
ifneq ($(VERBOSE),)
	PYTEST_OPTS ?= -v -s
else
	PYTEST_OPTS ?=
endif

.PHONY: default
default: buildext

.PHONY: cmake
cmake: $(BUILD_PATH)/Makefile

.PHONY: xcode
xcode: $(BUILD_PATH)_xcode/Makefile

# The preset carries the configure; -B and -G layer on top of it. The build
# tree keeps its ABI tag, which a preset cannot compute, and the generator
# stays the one the clean target and the workflows expect from a make build.
CMAKE_CMD = cmake --preset $(CMAKE_PRESET) $(CMAKE_OVERRIDES) $(CMAKE_ARGS)

# The table above leaves CMAKE_PRESET empty for a build type it does not name.
CHECK_BUILD_TYPE = test -n "$(CMAKE_PRESET)" || { \
	echo "Error: CMAKE_BUILD_TYPE is '$(CMAKE_BUILD_TYPE)'."; \
	echo "  Use RelWithDebInfo, Release, or Debug."; \
	exit 1; \
	}

$(BUILD_PATH)/Makefile: CMakeLists.txt CMakePresets.json Makefile
	@$(CHECK_BUILD_TYPE)
	env $(RUNENV) $(CMAKE_CMD) -B $(BUILD_PATH) -G "Unix Makefiles"

$(BUILD_PATH)_xcode/Makefile: CMakeLists.txt CMakePresets.json Makefile
	@$(CHECK_BUILD_TYPE)
	env $(RUNENV) $(CMAKE_CMD) -B $(BUILD_PATH)_xcode -G Xcode

.PHONY: buildext
buildext: cmake
	cmake --build $(BUILD_PATH) --target _solvcon_py VERBOSE=$(VERBOSE) $(MAKE_PARALLEL)

.PHONY: install
install: cmake
	cmake --build $(BUILD_PATH) --target $@ VERBOSE=$(VERBOSE) $(MAKE_PARALLEL)

# Pass PYTEST_OPTS to forward arguments to the pytest harness. Examples:
# Example for one file:
#   make pytest PYTEST_OPTS='-k test_buffer.py'
# Example for one class:
#   make pytest PYTEST_OPTS='-v -k SimpleArrayBasicTC'
# The GUI tests keep their windows off the screen (tests/conftest.py); export
# SOLVCON_TEST_SHOW_WINDOWS=ON to watch them.
.PHONY: pytest
pytest: buildext
	env $(RUNENV) \
		$(PYTEST) $(PYTEST_OPTS) tests/

.PHONY: pytest-fast
pytest-fast: buildext
	env $(RUNENV) \
		$(PYTEST) $(PYTEST_OPTS) tests/ --ignore=tests/gui

.PHONY: pytest-gui
pytest-gui: buildext
	env $(RUNENV) \
		$(PYTEST) $(PYTEST_OPTS) tests/gui/

PROFFILES = $(shell find profiling -type f -name 'profile_*.py' | sort)
PROFRESDIR = profiling/results

.PHONY: pyprof
pyprof: buildext $(PROFFILES)
	@mkdir -p profiling/results
	@mkdir -p profiling/results/png
	@for fn in $(PROFFILES); \
	do \
		outfn=$${fn%%.py}; \
		outfn=profiling/results/$${outfn##profiling/}.output; \
		echo "$(WHICH_PYTHON) $${fn} > $${outfn}"; \
		env $(RUNENV) \
			$(WHICH_PYTHON) $${fn} > $${outfn} || exit 1; \
	done

.PHONY: pilot
pilot: cmake
	cmake --build $(BUILD_PATH) --target $@ VERBOSE=$(VERBOSE) $(MAKE_PARALLEL)

.PHONY: pilot_clang_tidy_diff
pilot_clang_tidy_diff: cmake
	@test -n "$(SOLVCON_DIFF_BASE)" || { \
		echo "Error: SOLVCON_DIFF_BASE is required."; \
		exit 1; \
	}
	env SOLVCON_DIFF_BASE="$(SOLVCON_DIFF_BASE)" \
		cmake --build $(BUILD_PATH) --target $@ VERBOSE=$(VERBOSE)

.PHONY: gtest
gtest: cmake
	cmake --build $(BUILD_PATH) --target run_gtest VERBOSE=$(VERBOSE) $(MAKE_PARALLEL)

# Build and launch the pilot GUI. PYTHONPATH is set via RUNENV so the
# in-tree package is found; CMake resolves the platform binary path.
.PHONY: run_pilot
run_pilot: pilot
	env $(RUNENV) \
		cmake --build $(BUILD_PATH) --target $@ VERBOSE=$(VERBOSE)

# Pass PYTEST_OPTS to forward arguments to the pytest harness running
# inside the pilot binary.
# Example for one file:
#   make run_pilot_pytest PYTEST_OPTS='-k test_buffer.py'
# Example for one class:
#   make run_pilot_pytest PYTEST_OPTS='-v -k SimpleArrayBasicTC'
# The GUI tests keep their windows off the screen (tests/conftest.py); export
# SOLVCON_TEST_SHOW_WINDOWS=ON to watch them.
.PHONY: run_pilot_pytest
run_pilot_pytest: pilot
	env $(RUNENV) PYTEST_OPTS="$(PYTEST_OPTS)" \
		cmake --build $(BUILD_PATH) --target $@ VERBOSE=$(VERBOSE)

.PHONY: bundle-precheck
bundle-precheck:
	$(SOLVCON_ROOT)/contrib/bundle/bundle-with-homebrew.sh check

.PHONY: bundle
bundle:
	$(SOLVCON_ROOT)/contrib/bundle/bundle-with-homebrew.sh all \
		--output "$(RELEASE_OUTPUT)" $(RELEASE_ARGS)

.PHONY: bundle-test
bundle-test:
	$(SOLVCON_ROOT)/contrib/bundle/bundle-with-homebrew.sh verify \
		"$(RELEASE_ARTIFACT)"

.PHONY: standalone_buffer_setup
standalone_buffer_setup:
	$(MAKE) -C contrib/standalone_buffer copy

# A recursive make inherits no job slots from a parent invoked without -j, so
# the sub-make has to state its own. Under a parent that does carry -j, this
# opts out of the jobserver and the two counts add up.
.PHONY: standalone_buffer
standalone_buffer:
	$(MAKE) $(MAKE_PARALLEL) -C contrib/standalone_buffer build
	$(MAKE) -C contrib/standalone_buffer run

CLANG_FORMAT ?= clang-format
FLAKE8 ?= flake8
AUTOPEP8 ?= autopep8
# Pinned to the clang-format major version used by CI; see
# .github/workflows/lint.yml. A different major version may produce a different
# formatting output and cause CI disagreement with local runs.
CLANG_FORMAT_CI_VERSION ?= 20
# Keep autopep8 a no-op against the current code base: only fix codes that
# flake8 also reports here, leave the rest alone. Specifically ignore:
#   E121,E123,E126        continuation indent variants (flake8 default-ignored)
#   E201,E202,E203,E241   whitespace inside brackets / around commas; preserve
#                         deliberate `# noqa` numeric alignment such as in
#                         `tests/test_mesh.py`
#   E301,E303             blank-line rules that pycodestyle does not flag in
#                         their current uses (docstring-followed methods and
#                         nested defs inside `if HAS_SPHINX:`); autopep8 would
#                         add or remove blank lines that flake8 never reports
#   E501                  line too long; autopep8's wraps are often ugly, so
#                         leave long-line decisions to humans
#   W503,W504             line-break style around binary operators
#                         (flake8 default-ignored)
AUTOPEP8_OPTS ?= --recursive --max-line-length=79 \
                 --ignore=E121,E123,E126,E201,E202,E203,E241,E301,E303,E501,W503,W504 \
                 --exclude=thirdparty,tmp,_deps

CFFILES = $(shell find cpp gtests -type f -name '*.[ch]pp' | sort)
ifeq ($(FORCE_CLANG_FORMAT),inplace)
	CFCMD ?= $(CLANG_FORMAT) -i
else
	CFCMD ?= $(CLANG_FORMAT) --dry-run -Werror
endif

.PHONY: cformat
cformat: $(CFFILES)
	@command -v $(CLANG_FORMAT) >/dev/null 2>&1 || { \
		echo "Error: '$(CLANG_FORMAT)' not found in PATH."; \
		echo "  Install: pip install 'clang-format==$(CLANG_FORMAT_CI_VERSION).*'"; \
		echo "  (CI pins clang-format $(CLANG_FORMAT_CI_VERSION))"; \
		exit 1; \
	}
	@ver=$$($(CLANG_FORMAT) --version 2>/dev/null | sed -nE 's/.*version ([0-9]+).*/\1/p' | head -n1); \
	if [ -n "$$ver" ] && [ "$$ver" != "$(CLANG_FORMAT_CI_VERSION)" ]; then \
		echo "Warning: $(CLANG_FORMAT) major version $$ver differs from CI ($(CLANG_FORMAT_CI_VERSION)); formatting output may differ."; \
	fi
	@echo "Checking $(words $(CFFILES)) C++ files with clang-format..."
	@printf '%s\n' $(CFFILES) | xargs -P $(NPROC) -n1 $(CFCMD)

.PHONY: cinclude
cinclude: $(CFFILES)
	@if grep -rnE '^[[:space:]]*#[[:space:]]*include[[:space:]]*"' cpp/ gtests/ 2>/dev/null; then \
		echo "Error: use angle brackets for #include, not quotes (see lines above)."; \
		exit 1; \
	fi

.PHONY: flake8
flake8:
	@command -v $(FLAKE8) >/dev/null 2>&1 || { \
		echo "Error: '$(FLAKE8)' not found in PATH."; \
		echo "  Install: pip install flake8"; \
		exit 1; \
	}
	$(FLAKE8) . --jobs $(NPROC)

.PHONY: checkascii
checkascii:
	$(WHICH_PYTHON) contrib/lint/check_ascii.py

.PHONY: checktws
checktws:
	$(WHICH_PYTHON) contrib/lint/check_ascii.py --check-tws

# Keep the window tests under tests/gui and everything else out of it, so the
# two CI lanes keep selecting what they mean to. tests/gui/README.md states
# the rule.
.PHONY: checktests
checktests:
	$(WHICH_PYTHON) contrib/lint/check_test_layout.py

# Run the lint targets concurrently, scaled to the processor count, and keep
# going on failure so every check reports before make exits non-zero.
.PHONY: lint
lint:
	@$(MAKE) --no-print-directory -j $(NPROC) -k lint_targets

.PHONY: lint_targets
lint_targets: cformat cinclude flake8 checkascii checktws checktests

.PHONY: pyformat
pyformat:
	@command -v $(AUTOPEP8) >/dev/null 2>&1 || { \
		echo "Error: '$(AUTOPEP8)' not found in PATH."; \
		echo "  Install: pip install autopep8"; \
		exit 1; \
	}
	$(AUTOPEP8) $(AUTOPEP8_OPTS) --in-place .

.PHONY: format
format: pyformat
	@$(MAKE) FORCE_CLANG_FORMAT=inplace cformat

.PHONY: clean
clean:
	cmake --build $(BUILD_PATH) --target remove_solvcon_py
	make -C $(BUILD_PATH) clean

.PHONY: cmakeclean
cmakeclean:
	@if [ -f $(BUILD_PATH)/CMakeCache.txt ]; then \
		cmake --build $(BUILD_PATH) --target remove_solvcon_py; \
	fi
	rm -rf $(BUILD_PATH)
