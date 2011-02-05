Developing
==========

New features:

- Add ctypes-based netCDF reading support in ``solvcon.io.netcdf``.
- Add Cubit/Genesis/ExodosII reader in ``solvcon.io.genesis``.
- Add Cubit invokation helper for on-the-fly mesh generation.
- Add special CESE solver for linear equations in ``solvcon.kerpak.lincese``.
- Add 2/3D anisotropic, linear elastic solver based on linear CESE solver in
  ``solvcon.kerpak.elaslin``.
- Add an example for custom solver in ``example/misc/elas3d``.
- Add a ctypes-based CUDA wrapper in ``solvcon.scuda``.

Changes:

- Refactor coupling of periodic boundary condition.
- Remove ``\*ptr`` in ``solvcon.dependency``.
- Correct sol() to soln() and dsol() to dsoln() in BC.
- Move sol()/soln() and dsol()/dsoln() from ``solvcon.boundcond`` to kerpak.

0.0.2
=====

- Bring in anisotropic elastic solver.
- Implement proof-of-concept in situ visualization.
- Refactor str_path property in solvcon.batch.Batch.

0.0.1
=====

- The first alpha release: a technology preview.

.. vim: set ft=rst ff=unix fenc=utf8:
