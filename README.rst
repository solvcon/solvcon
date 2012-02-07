==================
README for SOLVCON
==================

:author: Yung-Yu Chen <yyc@solvcon.net>
:copyright: c 2008-2012.

SOLVCON: a multi-physics software framework for high-fidelity solutions of
partial differential equations (PDEs) by hybrid parallelism.

SOLVCON uses the space-time Conservation Element and Solution Element (`CESE
<http://www.grc.nasa.gov/WWW/microbus/>`_) method to solve generic conservation
laws.  SOLVCON focuses on rapid development of high-performance computing (HPC)
code for large-scale simulations.  SOLVCON is developed by using Python for the
main structure, to incorporate C, `CUDA
<http://www.nvidia.com/object/cuda_home_new.html>`_, or other programming
languages for HPC.

SOLVCON is released under `GNU GPLv2
<http://www.gnu.org/licenses/gpl-2.0.html>`_, and developed by `Yung-Yu Chen
<mailto:yyc@solvcon.net>`_ and `Sheng-Tao John Yu <mailto:yu.274@osu.edu>`_.
The official web site is at http://solvcon.net/ .

Key Features:

- Pluggable multi-physics
- Unstructured meshes for modeling complex geometry
- Hybrid parallel computing
- Ready-to-use I/O formats
- Parallel I/O and in situ visualization
- Automated work flow

.. vim: set ft=rst ff=unix fenc=utf8: