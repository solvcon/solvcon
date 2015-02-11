Version 0.1.4+
++++++++++++++

Release date:

Changes:

- #105: Split examples/run_linear_cvg.sh into two files 

Bug-fixes:

Version 0.1.4
+++++++++++++

Release date: 2015/2/9 (GMT+0800)

This is released simply because PyPI doesn't allow re-uploading a file of the
same name, and I didn't know it and deleted the file for version 0.1.3.  See
https://mail.python.org/pipermail/distutils-sig/2015-January/025683.html.

Version 0.1.3
+++++++++++++

Release date: 2015/2/9 (GMT+0800)

Changes:

- #75 Change the default dependency directory (``$SCROOT``) from
  ``$HOME/opt/scruntime`` to ``$root/opt``.
- #77 SCons script should respect LIBPATH environment variable.
- #95: Support anaconda/miniconda/conda for managing dependencies
- #98: Migrate the old gas-dynamics solvers into the new MeshSolver system
- #103, #106, #110: Improve documents

Bug-fixes:

- #73 scotch build script doesn't work in OS X.
- #74 SCons doesn't build Python extension modules.
- #78 solvcon.rpc.Dealer.bridge doesn't work for localhost in OS X
- #79 Legacy case setting steps_dump results into error
- #82: SCons stops when there's file removed/moved
- #87: Fix run_vewave_cvg.sh solution test
- #89: examples/vewave/cvg/go failed due to missing mesh
- #90: parcel/* solvers miss ghostgeom
- #92: Clean up solvcon.parcel.bulk.solver.BulkSolver initial condition code
- #93: http://ci.solvcon.net/job/examples.vewave_cvg/ fails around revision
  51e85bf107a3a85b9930b4cabc5f3aa13b9c59bf
- #99: Move solvcon.solver.ALMOST_ZERO into
  solvcon.solver.MeshSolver.ALMOST_ZERO
- #100: The exception wrapper in solvcon.cmdutil.go is not useful
- #101: MeshSolver should use MeshAnchorList instead of AnchorLIst (legacy)
- #109: Add dependency package gmp (required by Gmsh) packaging
- #114: Add missing package to dependency installation scripts packaging 

Version 0.1.2
+++++++++++++

Release date: 2013/8/8 (GMT+0800)

This release starts to document by using Sphinx and renew the architecture of
problem solvers.

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
- Design a new hierarchy for solvers by using Cython.  (#59, #60, #62, #63,
  #65)

  - A new series of "sach" (:py:class:`MeshSolver <solvcon.solver.MeshSolver>`,
    :py:class:`MeshAnchor <solvcon.anchor.MeshAnchor>`, :py:class:`MeshCase
    <solvcon.case.MeshCase>`, and :py:class:`MeshHook <solvcon.hook.MeshHook>`)
    is built.
  - The new sach is built upon pure Python :py:class:`Block
    <solvcon.block.Block>` and Cython :py:class:`Mesh <solvcon.mesh.Mesh>`.

Version 0.1.1
+++++++++++++

Release date: 2012/1/21 (GMT+0800)

This release adds a loader of Gmsh mesh format and fixes several bugs.

New features:

- Add a loader for Gmsh ASCII mesh format.  The loader locates in
  solvcon.io.gmsh and is implemented as pure Python code.  ``scg mesh`` command
  line tool can recognize the format.  Issue #52.
- Revamp the dependency building system to support older OSes and proxies that
  need authentication.  Issue #53.
- Extract the SCons commands for building the Epydoc and Sphinx document from
  SConstruct into standalone SCons tools.  Two new tools are added in the
  directory ``site_scons/site_tools/``: ``sphinx.py`` and ``scons_epydoc.py``.
  Note that the SCons tool for Epydoc cannot be named as ``epydoc.py`` or the
  name collides with the real ``epydoc`` package.
- Add Gmsh and Sphinx into ground/.

Bug-fix:

- Issue #49: "No Vtk for final time step".  Output timing of CollectHook and
  MarchSave.
- Issue #54: "Shared objects are not found under Mac OS X".
- Issue #38: "soln/dsoln shouldn't be hard-coded".

Version 0.1
+++++++++++

Release date: 2011/8/11 (GMT-0500)

This release marks a milestone of SOLVCON.  Future development of SOLVCON will
focus on production use.  The planned directions include (i) the high-order
CESE method, (ii) improving the scalability by consolidating the
distributed-memory parallel code, (iii) expanding the capabilities of the
existing solver kernels, and (iv) incorporating more physical processes.

New features:

- Glue BCs are added.  A pair of collocated BCs can now be glued together to
  work as an internal interface.  The glued BCs helps to dynamically turn on or
  off the BC pair.
- ``solvcon.kerpak.cuse`` series solver kernels are changed to use OpenMP for
  multi-threaded computing.  They were using a thread pool built-in SOLVCON for
  multi-threading.  OpenMP makes multi-threaded functions more flexible in
  argument specification.
- Add the ``soil/`` directory for providing building helpers for GCC 4.6.1.
  Note, the name ``gcc/`` is deliberately avoided for the directory, because of
  a bug in gcc itself (bug id 48306
  http://gcc.gnu.org/bugzilla/show_bug.cgi?id=48306 ).
- Add ``-j`` command line option for building dependencies in the ``ground/``
  directory and the ``soil/`` directory.  Note that ATLAS doesn't work with
  ``make -j N``.

Bug-fix:

- METIS changes its download URL.  Modify SConstruct accordingly.

Version 0.0.7
+++++++++++++

Release date: 2011/6/8 (GMT-0500)

In this release, SOLVCON starts to support using incenters or centroids for
constructing basic Conservation Elements (BCEs) of the CESE method.  Incenters
are only enabled for simplex cells.  Three more examples for supersonic flows
are also added, in addition to the new capability.

New features:

- A set of building scripts for dependencies of SOLVCON is written in
  ``ground/`` directory.  A Python script ``ground/get`` download all depended
  source tarballs according to ``ground/get.ini``.  A make file
  ``ground/Makefile`` directs the building with targets ``binary``, ``python``,
  ``vtk``.  The targets must be built in order.  An environment variable
  ``$SCPREFIX`` can be set when making to specify the destination of
  installation.  The make file will create a shell script
  ``$SCROOT/bin/scvars.sh`` exporting necessary environment variables for using
  the customized runtime.  ``$SCROOT`` is the installing destination (i.e.,
  ``$SCPREFIX``), and is set in the shell script as well.
- The center of a cell can now be calculated as an incenter.  Use of incenter
  or centroid is controlled by a keyword parameter ``use_incenter`` of
  ``solvcon.block.Block`` constructor.  This enables incenter-based CESE
  implementation that will benefit calculating Navier-Stokes equations in the
  future.
- More examples for compressible inviscid flows are provided.

Bug-fix:

- A bug in coordiate transformation for wall boundary conditions of gas
  dynamics module (``solvcon.kerpak.gasdyn``).

Version 0.0.6
+++++++++++++

Release date: 2011/5/18 (GMT-0500)

This release also contains enhancements planned for 0.0.5, which would not be
released.  SOLVCON now partially supports GPU clusters.  Solvers for linear
equations and the velocity-stress equations are updated.  The CESE base solver
is enhanced.

New features:

- Support GPU clusters.  SOLVCON can spread decomposed sub-domains to multiple
  GPU devices distributed over network.  Currently only one GPU device per
  compute node is supported.
- A generic solver for linear equations: ``solvcon.kerpak.lincuse``.  The new
  version of generic linear solver support both CPU and CPU.
- A velocity-stress equaltions solver is ported to be based on
  ``solvcon.kerpak.lincuse``.  The new solver is packaged in
  ``solvcon.kerpak.vslin``.
- Add W-3 weighting scheme to ``solvcon.kerpak.cuse``.  W-3 scheme is more
  stable than W-1 and W-2.

Bug-fixes:

- Consolidate reading quadrilateral mesh from CUBIT/Genesis/ExodusII; CUBIT
  uses 'SHELL4' for 2D quad.
- Update SCons scripts for the upgrade of METIS to 4.0.3.

Version 0.0.4
+++++++++++++

Release date: 2011/3/2 (GMT-0500)

This release enhances pre-procesing and start-up for large-scale simulations.
Unstructured meshes using up to 66 million elements have been tested.  Two new
options to ``solvcon.case.BlockCase`` are added: (i) ``io.domain.with_arrs``
and (ii) ``io.domain.with_whole``.  They can be used to turn off arrays in the
``Collective`` object.  By omitting those arrays on head node, memory usage is
significantly reduced.  Available memory on head node will not constrain the
size of simulations.

Bug-fix:

- Issue #12: Order of variables for in situ visualization can be specified to
  make the order of data arrays of VTK poly data consistent among head and
  slave nodes.

Version 0.0.3
+++++++++++++

Release date: 2011/2/20 (GMT-0500)

The biggest improvement of this release is the addition of CUDA-enabled, CESE
base solver kernel ``solvcon.kerpak.cuse``.  ``cuse`` module is designed to use
either pthread on CPU or CUDA on GPU.  The release also contains many important
features for future development, including interface with CUBIT, incorporation
of SCOTCH-5.1 for partitioning large graph.

New features:

- Add ctypes-based netCDF reading support in ``solvcon.io.netcdf``.
- Add Cubit/Genesis/ExodosII reader in ``solvcon.io.genesis``.
- Add Cubit invocation helper for on-the-fly mesh generation.
- Add special CESE solver for linear equations in ``solvcon.kerpak.lincese``.
- Add 2/3D anisotropic, linear elastic solver based on linear CESE solver in
  ``solvcon.kerpak.elaslin``.
- Add an example for custom solver in ``examples/misc/elas3d``.
- Add a ctypes-based CUDA wrapper in ``solvcon.scuda``.
- Add CUDA-enabled 2nd-order CESE solver.
- Add non-slip wall to ``solvcon.kerpak.gasdyn``.

Changes:

- Refactor coupling of periodic boundary condition.
- Remove ``*ptr`` in ``solvcon.dependency``.
- Correct sol() to soln() and dsol() to dsoln() in BC.
- Move sol()/soln() and dsol()/dsoln() from ``solvcon.boundcond`` to kerpak.
- Remove FORTRAN-related code.
- Create ``include/`` directory and put header files in it.
- By default, use SCOTCH-5.1 instead of METIS-4.  METIS-4 fails on allocating
  memory for meshes with more than 35 million cells.  If SCOTCH cannot be found
  in system, fall back to METIS-4.
- Refactor ``solvcon.domain.Collective.split()``.

Version 0.0.2
+++++++++++++

- Bring in anisotropic elastic solver.
- Implement proof-of-concept in situ visualization.
- Refactor str_path property in solvcon.batch.Batch.

Version 0.0.1
+++++++++++++

- The first alpha release: a technology preview.

.. vim: set ft=rst ff=unix fenc=utf8:
