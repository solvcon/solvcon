==================
README for SOLVCON
==================

:author: Yung-Yu Chen <yyc@solvcon.net>
:copyright: c 2008-2011.

SOLVCON: a multi-physics, supercomputing software framework for high-fidelity
solutions of partial differential equations (PDEs) by hybrid parallelism.

SOLVCON facilitates rapid devlopment of PDE solvers for massively parallel
computing.  C or CUDA_ is used for fast number-crunching.  SOLVCON is designed
for extension to various physical processes.  Numerical algorithms and physical
models are pluggable.  Sub-package ``solvcon.kerpak`` contains default
implementations.  The default numerical algorithm in SOLVCON is the space-time
Conservation Element and Solution Element (CESE_) method that solves generic
conservation laws.

SOLVCON is released under `GNU GPLv2
<http://www.gnu.org/licenses/gpl-2.0.html>`_, and developed by `Yung-Yu Chen
<mailto:yyc@solvcon.net>`_ and `Sheng-Tao John Yu <mailto:yu.274@osu.edu>`_.

Key Features
============

- **Multi-physics**: Pluggable physical models by the built-in CESE_ solvers
- **Complex geometry**: 2/3D unstructured mesh consisting of mixed shapes
- **Massively parallel**: Automatic domain decomposition with MPI or socket
- **GPGPU computing**: Hybrid parallelism with CUDA_
- **Large data set**: In situ visualization by using VTK_
- **I/O formats**: VTK, GAMBIT Neutral, CUBIT Genesis/ExodosII, etc.
- **Productive work flow**: Python scripts as programmable input files

Install
=======

The C code in SOLVCON is intentionally made to be standard shared libraries
rather than Python extension modules.  SOLVCON uses ctypes_ to load and call
these binary codes.  In this way, the binary codes can be flexibly built and
optimized for performance.  Hence, installing SOLVCON requires building these
libraries.  SOLVCON uses SCons_ as the builder.  It is recommended to run
SOLVCON on 64-bit Linux.

SOLVCON depends on the following packages: (i) Python_ 2.6 or 2.7 (preferred),
(ii) SCons_, (iii) gcc_ (version 4.3 or higher) or icc, (iv) Numpy_ (version
1.6 or higher), (v) LAPACK_, (vi) NetCDF_ (version 4 or higher), and (vii)
METIS_ (version 4.0.3; SOLVCON will download it for you on building).  Optional
dependencies include: (i) SCOTCH_ (version 5.1 or higher) as an alternative of
METIS, (ii) Nose_ for running unit tests, (iii) Epydoc_ for generating API
documentation, (iv) VTK_ for in situ visualization, and (v) docutils and
pygraphviz for Epydoc formatting.  Debian_ or Ubuntu_ users can use the
following command to install the dependencies::

  $ sudo apt-get install scons build-essential gcc liblapack-pic
    libnetcdf-dev libnetcdf6 netcdf-bin
    python2.7 python2.7-dev python-profiler python-numpy
    libscotch-5.1 python-nose python-epydoc python-vtk
    python-docutils python-pygraphviz 

CUDA_ needs to be separately installed and configured.  For using meshes with
more then 35 million cells, SCOTCH-5.1 is recommended.  METIS-4 has issues on
memory allocation for large graphs.

The end of this section describes how to manually compile these dependencies
with helper scripts shipped with SOLVCON.

The three steps to install:

1. Obtain the latest release from
   https://bitbucket.org/yungyuc/solvcon/downloads .  Unpack the source
   tarball.

2. Get into the source tree and run SCons_ to build the binary codes::

     $ cd $SCSRC
     $ scons --download --extract

   ``$SCSRC`` indicates the root directory of unpacked source tree.

3. Install everything::

     $ python setup.py install

The option ``--download`` used above asks the building script to download
necessary external source packages, e.g., METIS_, from Internet.  Option
``--extract`` extracts the downloaded packages.

Although not recommended, you can optionally install SOLVCON to your
``$HOME/.local`` directory.  It is one of the workarounds when you don't have
the root permission on the system.  To do this, add the ``--user`` when
invoking the ``setup.py`` script::

 $ python setup.py install --user

Install from Repository
=======================

To use the latest source from the code repository, you need to use Mercurial_
to clone the repository to your local disk::

  $ sudo apt-get install mercurial
  $ hg clone https://bitbucket.org/yungyuc/solvcon

and then follow steps 2 and 3.

Rebuild/Reinstall
=================

If you want to rebuild and reinstall, you can run::

  $ cd $SCSRC
  $ scons
  $ python setup.py install

without using the options ``--download`` and ``--extract``.  If you want a
clean rebuild, run ``scons -c`` before ``scons``.

Unit Test
=========

If you have Nose_ installed, you can run::

  $ nosetests

inside the source tree for unit tests.  To test installed version, use the
following command instead::

  $ python -c 'import solvcon; solvcon.test()'

When testing installed version, make sure your current directory does not have
a sub-directory named as ``solvcon``.

Because SOLVCON uses ssh_ as its default approach for remote procedure call
(RPC), you need to set up the public key authentication for ssh, or some of the
unit tests for RPC could fail.  Some tests using optional libraries could be
skipped (indicated by S), if you do not have the libraries installed.
Everything else should pass.

Build and Install Dependencies (Optional)
=========================================

SOLVCON depends on a number of external software packages.  Although these
dependencies should be met by using the package management of the OSes, getting
the support staffs to install missing packages on a supercomputer/cluster takes
time.  As such, SOLVCON provides a simple building system to facilitate the
installation into a user's home directory or a customized path.

Some Python installation does not include the VTK wrapper.  In this case, one
might also need to use self-compiled Python runtime to use VTK in SOLVCON for
in situ visualization.

The ``$SCSRC/ground`` directory contains scripts to build most of the packages
that SOLVCON depends on.  The ``$SCSRC/ground/Makefile`` file has three default
targets: ``binary``, ``python``, and ``vtk``.  And the additional ``all``
target will run all of them in order.  The built files will be automatically
installed into the path specified by the ``$SCROOT`` environment variable,
which is set to ``$HOME/opt/scruntime`` by default.  The
``$SCROOT/bin/scvars.sh`` script will be created to export necessary
environment variables for the installed dependencies, including the ``$SCROOT``
environment variable.

The ``$SCSRC/gcc`` directory contains scripts to build gcc_.  The
``$SCROOT/bin/scgccvars.sh`` script will be created to export necessary
environment variables for the installed gcc.  The enabled languages include
only C, C++, and Fortran.  The default value of ``$SCROOT`` remains to be
``$HOME/opt/scruntime``, while the built compiler will be installed into
``$SCROOT/gcc``.  Note: do not use different ``$SCROOT`` when compiling
``$SCSRC/gcc`` and ``$SCSRC/ground``.

``$SCROOT/bin/scvars.sh`` and ``$SCROOT/bin/scgccvars.sh`` can be separately
imported.  The two sets of packages reside in different directories and do not
mix with each other nor system software.  Users can diable these environments
by not importing the two scripts.

Some packages have not been incorporated into the dependency building system
described above.  Debian_ or Ubuntu_ users should install the additional
dependencies by using::

  $ sudo apt-get install build-essential gcc gfortran gcc-multilib m4
  libreadline6 libreadline6-dev libncursesw5 libncurses5-dev libbz2-1.0
  libbz2-dev libdb4.8 libdb-dev libgdbm3 libgdbm-dev libsqlite3-0
  libsqlite3-dev libcurl4-gnutls-dev libhdf5-serial-dev libgl1-mesa-dev
  libxt-dev

These building scripts have only been tested with 64-bit Linux.

Resources
=========

- Portal (with API document): http://solvcon.net/
- Mailing list: http://groups.google.com/group/solvcon
- Downloads: http://bitbucket.org/yungyuc/solvcon/downloads

.. _CESE: http://www.grc.nasa.gov/WWW/microbus/
.. _SCons: http://www.scons.org/
.. _Python: http://www.python.org/
.. _gcc: http://gcc.gnu.org/
.. _Numpy: http://www.numpy.org/
.. _LAPACK: http://www.netlib.org/lapack/
.. _NetCDF: http://www.unidata.ucar.edu/software/netcdf/index.html
.. _METIS: http://glaros.dtc.umn.edu/gkhome/views/metis/
.. _SCOTCH: http://www.labri.fr/perso/pelegrin/scotch/
.. _Epydoc: http://epydoc.sf.net/
.. _CUDA: http://www.nvidia.com/object/cuda_home_new.html
.. _Mercurial: http://mercurial.selenic.com/
.. _ssh: http://www.openssh.com/
.. _Nose: http://somethingaboutorange.com/mrl/projects/nose/
.. _VTK: http://vtk.org/
.. _ctypes: http://docs.python.org/library/ctypes.html
.. _Debian: http://debian.org/
.. _Ubuntu: http://ubuntu.com/

.. vim: set ft=rst ff=unix fenc=utf8: