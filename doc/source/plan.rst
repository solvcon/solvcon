================
Development Plan
================

Version 0.1.2
=============

This release starts to document by using Sphinx and to improve the class
hierarchy of IO and boundary-condition treatments.

Changes:

- Remove the SCons METIS builder and the corresponding options of
  ``--download``, ``--extract``, and ``--apply-patches``.  Now the SCOTCH
  library is used for graph partitioning with its METIS interface.
- Move the counter of lines of code in SOLVCON from SCons into a standalone
  script ``contrib/countloc``, and thus remove the SCons option ``--count``.
- Remove the SCons option ``--cmpvsn``.  For changing the command of C
  compiler, you can now set the environment variable ``CC`` to whatever the
  command you want.
- Renovate the documentation by using Sphinx.  (#61)
- Add a directory ``contrib/verify_scripts`` to collect scripts for running
  verification examples.
- Design a new hierarchy for solvers by using Cython.  (#59, #60, #62, #63)

  - A new series of "sach" (:py:class:`MeshSolver <solvcon.solver.MeshSolver>`,
    :py:class:`Anchor <solvcon.anchor.MeshAnchor>`, :py:class:`MeshCase
    <solvcon.case.MeshCase>`, and :py:class:`MeshHook <solvcon.hook.MeshHook>`)
    is built.
  - The new sach is built upon pure Python :py:class:`Block
    <solvcon.block.Block>` and Cython :py:class:`Mesh <solvcon.mesh.Mesh>`.

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
