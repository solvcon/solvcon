================
Development Plan
================

(Deprecated).

Version 0.1.3
=============

- Incorporate the two-dimensional aero-/hydro-acoustic solver.

Version 0.1.4
=============

- Replicate all solvers that is derived from solvcon.kerpak.cuse to use the Cython-based solver system.
- Deprecate solvcon.kerpak.cese series solvers.

Version 0.1.5
=============

- Move all solvers out of solvcon.kerpak to another structure.
- Discard the solvcon.kerpak namespace.

Version 0.1.6
=============

- Use mpi4py instead of interfacing an arbitrary MPI library with ctypes.
- Replace ctypes with Cython for a more robust interface to C/C++ code.

Version 0.2
===========

- Organize verification, documentation, and examples for the included solvers.

Further Items
=============

- Consolidate the class hierarchy within solvcon.io sub-package.
- Improve the BC hierarchy to allow storing per-object data.

.. vim: set ft=rst ff=unix fenc=utf8:
