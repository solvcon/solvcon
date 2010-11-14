==================
README for SOLVCON
==================

:author: Yung-Yu Chen <yyc@solvcon.net>
:copyright: c 2008-2010.

SOLVCON: a multi-physics software framework for high-fidelity solutions of
partial differential equations by hybrid parallelism.

Python is the primary programming language for constructing SOLVCON.
Number-crunching is performed by high-speed subroutines written in C.  By
taking the mixed-language approach, PDE solvers can be rapidly developed and
simultaneously utilize hundreds of nodes in a supercomputer by parallel
computing.  SOLVCON is multi-physics, and stocking numerical algorithms and
physical models are ready for use in the namespace ``solvcon.kerpak``.  See
http://solvcon.net/ or contact the author `Yung-Yu Chen <yyc@solvcon.net>`_ for
detail.

The default numerical algorithm in SOLVCON is the space-time Conservation
Element and Solution Element (CESE_) method.  The CESE_ method delivers
time-accurate solutions for hyperbolic PDEs.

SOLVCON is free software (for freedom, not price) and released under GPLv2.
See http://www.gnu.org/licenses/gpl-2.0.html or ``COPYING`` for the complete
license.  SOLVCON is still in alpha and subjects to changes.  No effort is made
for backward compatibility at the current stage.

Key Features
============

- Unstructured mesh consisting of mixed elements in two- and three-dimensional
  space.
- Use of advanced Message-Passing Interface (MPI) libraries.
- Automatic distributed-memory parallelization by domain decomposition.
- Highly modularized solving kernels of PDEs to decouple pthread and CUDA from
  domain decomposition for hybrid parallelism.
- Integration to supercomputer (cluster) batch systems: automatic construction
  of submit scripts.
- Built-in writers to VTK legacy and XML formats.
- Built-in communication layer by using socket: working without MPI installed.

Install
=======

The C codes in SOLVCON are intentionally made to be generic shared libraries
rather than Python extension modules.  SOLVCON uses ``ctypes`` to load and call
these binary codes.  In this way, the binary codes can be flexibly built and
optimized for performance.  Hence, installing SOLVCON requires building these
libraries.  SOLVCON uses SCons_ as the binary builder.

For SOLVCON to be built and run, it requires the following packages: (i)
Python_ 2.6, (ii) SCons_, (iii) a C compiler, gcc_ or icc is OK, (iv) Numpy_,
and (v) METIS_ for graph partitioning (SOLVCON will download it for you on
building).  If you want to run the unit tests after building SOLVCON, you
should also install Nose_.  It is recommended to run SOLVCON on 64-bits Linux
for high-resolution simulations.

Procedures to install are:

1. First, obtain the latest release from
   https://bitbucket.org/yungyuc/solvcon/downloads .  Unpack the source
   tarball.  Assume ``$SCSRC`` indicates the root directory of unpacked source
   tree.

2. Get into the source tree and run SCons_ to build the binary codes::

     $ cd $SCSRC
     $ scons --download --extract --apply-patches=metislog2

3. Install everything::

     $ python setup.py install

The option ``--download`` used above asks the building script to download
necessary external packages, e.g., METIS, from Internet.  Option ``--extract``
extracts the downloaded packages.  Since METIS is incompatible to the current
release of gcc, a patch is supplied with SOLVCON and can be automatically
applied to the downloaded METIS source with the ``--apply-patches`` option.

If you want to rebuild the binary after the installation, you can run::

  $ cd $SCSRC
  $ scons
  $ python setup.py install

without using the options ``--download``, ``--extract``, and
``--apply-patches``.  If you want a clean rebuild, run ``scons -c`` before
``scons``.

Optionally, if you have Nose_ installed, you can run::

  $ nosetests

for unit tests.  Every test should pass, except something specific to cluster
batch systems could be skipped (indicated by S).

How to Use
==========

Examples for using SOLVCON are put in ``$SCSRC/examples``.  To run these
examples, you need corresponding mesh data, which is kept in a standalone
repository at https://bitbucket.org/yungyuc/scdata .  The ``scdata`` directory
should be put in a directory upper than ``$SCSRC/examples``.  The examples will
find the ``scdata`` directory automatically.

These examples are useful for you to learn how to use SOLVCON to construct your
own solvers or applications.

.. _CESE: http://www.grc.nasa.gov/WWW/microbus/
.. _SCons: http://www.scons.org/
.. _Python: http://www.python.org/
.. _gcc: http://gcc.gnu.org/
.. _Numpy: http://www.numpy.org/
.. _METIS: http://glaros.dtc.umn.edu/gkhome/views/metis/
.. _Nose: http://somethingaboutorange.com/mrl/projects/nose/

.. vim: set ft=rst ff=unix fenc=utf8: