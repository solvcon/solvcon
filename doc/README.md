# solvcon documentation

solvcon is a *hybrid C++/Python* numerical library.  This directory contains
the Sphinx-based documentation.

## Build

```sh
pip install -r requirements.txt   # Python deps
make html                         # needs Doxygen; -> build/html/index.html
```

`make html` runs `make doxygen` first so that the C++ API is current. Install
Doxygen before building. Run `make doxygen` by itself only to regenerate the
XML under `build/doxygen` without rebuilding the HTML site.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
