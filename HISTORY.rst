0.0.3
=====

Date: 2011/2/20

The biggest improvement of this release is the addition of CUDA-enabled, CESE
base solver kernel ``solvcon.kerpak.cuse``.  ``cuse`` is designed to use either
pthread on CPU or CUDA on GPU.  The release also contains many important
features for future development, including interface with CUBIT, incorporation
of SCOTCH-5.1 for partitioning large graph.

New features:

- Add ctypes-based netCDF reading support in ``solvcon.io.netcdf``.
- Add Cubit/Genesis/ExodosII reader in ``solvcon.io.genesis``.
- Add Cubit invokation helper for on-the-fly mesh generation.
- Add special CESE solver for linear equations in ``solvcon.kerpak.lincese``.
- Add 2/3D anisotropic, linear elastic solver based on linear CESE solver in
  ``solvcon.kerpak.elaslin``.
- Add an example for custom solver in ``examples/misc/elas3d``.
- Add a ctypes-based CUDA wrapper in ``solvcon.scuda``.
- Add CUDA-enabled 2nd-order CESE solver.
- Add non-slip wall to ``solvcon.kerpak.gasdyn``.

Changes:

- Refactor coupling of periodic boundary condition.
- Remove ``\*ptr`` in ``solvcon.dependency``.
- Correct sol() to soln() and dsol() to dsoln() in BC.
- Move sol()/soln() and dsol()/dsoln() from ``solvcon.boundcond`` to kerpak.
- Remove FORTRAN-related code.
- Create ``include/`` directory and put header files in it.
- By default, use SCOTCH-5.1 instead of METIS-4.  METIS-4 fails on allocating
  memory for meshes with more than 35 million cells.  If SCOTCH cannot be found
  in system, fall back to METIS-4.
- Refactor ``solvcon.domain.Collective.split()``.

0.0.2
=====

- Bring in anisotropic elastic solver.
- Implement proof-of-concept in situ visualization.
- Refactor str_path property in solvcon.batch.Batch.

0.0.1
=====

- The first alpha release: a technology preview.

.. vim: set ft=rst ff=unix fenc=utf8:
