# solvcon documentation

solvcon is a *hybrid C++/Python* numerical library.  This directory contains
the Sphinx-based documentation.

## Build

```sh
pip install -r requirements.txt   # Python deps
make doxygen                      # optional: C++ API XML (needs doxygen)
make html                         # -> build/html/index.html
```

`make html` works without `make doxygen`; the C++ API page simply renders
empty (a warning, not an error) until the XML exists.

A configured CMake build tree offers the same build as targets.  `doc` builds
the HTML site and runs Doxygen first when it is installed, and `doc_doxygen`
runs Doxygen alone.  Both write to `build/` in this directory, and Sphinx runs
under the Python interpreter the tree was configured with:

```sh
cmake --build <build-tree> --target doc
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
