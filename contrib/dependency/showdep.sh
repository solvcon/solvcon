#!/bin/bash
#
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

echo "gcc path: $(which gcc)"
echo "gcc version: $(gcc --version)"
echo "g++ path: $(which g++ 2>/dev/null || echo not-installed)"
echo "g++ version: $(g++ --version 2>/dev/null | head -1 || echo not-installed)"
# modmesh's SimpleArray.hpp needs C++23 <mdspan> (GCC >= 14); flag a compiler
# too old to build it before the build itself does.
echo "C++23 <mdspan>: $(echo '#include <mdspan>' \
  | ${CXX:-g++} -std=c++23 -x c++ -fsyntax-only - 2>/dev/null \
  && echo ok || echo MISSING)"
echo "cmake path: $(which cmake)"
echo "cmake version: $(cmake --version)"
echo "python3 path: $(which python3)"
echo "python3 version: $(python3 --version)"
echo "python3-config --prefix: $(python3-config --prefix)"
echo "python3-config --exec-prefix: $(python3-config --exec-prefix)"
echo "python3-config --includes: $(python3-config --includes)"
echo "python3-config --libs: $(python3-config --libs)"
echo "python3-config --cflags: $(python3-config --cflags)"
echo "python3-config --ldflags: $(python3-config --ldflags)"
echo "pip3 path: $(which pip3)"
python3 -c 'import numpy as np; print("np.__version__:", np.__version__, np.get_include())'
echo "pytest path: $(which pytest)"
echo "pytest version: $(pytest --version)"
echo "clang-tidy path: $(which clang-tidy)"
echo "clang-tidy version: $(clang-tidy -version)"
echo "clang-format path: $(which clang-format 2>/dev/null || echo not-installed)"
echo "clang-format version: $(clang-format --version 2>/dev/null || echo not-installed)"
echo "flake8 path: $(which flake8)"
echo "flake8 version: $(flake8 --version)"
echo "black path: $(which black 2>/dev/null || echo not-installed)"
echo "black version: $(black --version 2>/dev/null || echo not-installed)"
# Qt toolchain for the pilot GUI. qsb (the shader baker from ShaderTools) and
# the Qt platform plugins are what most often go missing on a fresh machine.
echo "qmake path: $(which qmake6 qmake 2>/dev/null | head -1 || echo not-installed)"
echo "qmake QT_VERSION: $(qmake6 -query QT_VERSION 2>/dev/null \
  || qmake -query QT_VERSION 2>/dev/null || echo not-installed)"
echo "qsb path: $(which qsb 2>/dev/null || echo not-installed)"
echo "PySide6 version: $(python3 -c 'import PySide6; print(PySide6.__version__)' \
  2>/dev/null || echo not-installed)"
echo "shiboken6 version: $(python3 -c 'import shiboken6; print(shiboken6.__version__)' \
  2>/dev/null || echo not-installed)"
