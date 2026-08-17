# solvcon: modules to solve conservation laws with unstructured meshes

solvcon is a modularized code implementing [the space-time conservation element
and solution element (CESE)
method](https://yyc.solvcon.net/en/latest/cese/index.html) based on
unstructured meshes of mixed element to solve conservation laws. It is
developed by using C++ and Python to provide:

1. Contiguous buffers and multi-dimensional arrays.
2. Linear algebra built on BLAS and LAPACK, including a general eigensolver,
   LU factorization, and a Kalman filter.
3. Integral transform (the Fourier transform).
4. One-dimensional solvers for the Euler and linear scalar equations to
   demonstrate the CESE method.
5. Two- and three-dimensional solvers for the Euler equations using the
   CESE method. (Under development.)
6. Mesh and field file input and output for the Gmsh and Plot3D formats.
7. A geometry processor with polygons, Bezier curves, and R-tree spatial
   indexing.
8. Two- and three-dimensional body mesh generation. (Under development.)
9. An integrated runtime profiler.
10. A graphical user interface (GUI) application based on Qt for the spatial
    data and analysis.

The [nightly-build_windows GitHub
Action](https://github.com/solvcon/solvcon/actions/workflows/nightly-build_windows.yml?query=event%3Aschedule+is%3Asuccess+branch%3Amaster)
builds an experimental portable Windows binary. Open the newest successful run
and scroll down to the "artifacts" section to download the zip file. You need a
[GitHub](https://github.com/) login.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
