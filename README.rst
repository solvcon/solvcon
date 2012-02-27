==================
README for SOLVCON
==================

:author: Yung-Yu Chen <yyc@solvcon.net>
:copyright: c 2008-2012.

SOLVCON: A software framework to develop high-fidelity solvers of partial
differential equtions (PDEs).

SOLVCON uses the space-time `Conservation Element and Solution Element CESE
<http://www.grc.nasa.gov/WWW/microbus/>`__) method to solve generic
conservation laws.  Python is used to host code written in C, `CUDA
<http://www.nvidia.com/object/cuda_home_new.html>`__, or other programming
languages for high-performance computing (HPC).  Hybrid parallelism is achieved
by segregating share- and distributed-memory parallel computing in the
different layers of the software framework established by Python.

SOLVCON is developed by `Yung-Yu Chen <mailto:yyc@solvcon.net>`__ and
`Sheng-Tao John Yu <mailto:yu.274@osu.edu>`__, and released under `GNU GPLv2
<http://www.gnu.org/licenses/gpl-2.0.html>`__.  Please consult the web site
http://solvcon.net/ for more information.

Key Features:

- Pluggable multi-physics
- Unstructured meshes for modeling complex geometry
- Hybrid parallel computing
- Ready-to-use I/O formats
- Parallel I/O and in situ visualization
- Automated work flow

.. vim: set ft=rst ff=unix fenc=utf8: