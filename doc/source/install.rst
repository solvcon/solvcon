============
Installation
============

Prerequisites
=============

SOLVCON itself depends on the following packages:

- `gcc <http://gcc.gnu.org/>`_ 4.3+
- `SCons <http://www.scons.org/>`_ 2+
- `Python <http://www.python.org/>`_ 2.7
- `Cython <http://www.cython.org/>`_ 0.16+
- `Numpy <http://www.numpy.org/>`_ 1.5+
- `LAPACK <http://www.netlib.org/lapack/>`_
- `NetCDF <http://www.unidata.ucar.edu/software/netcdf/index.html>`_ 4+
- `SCOTCH <http://www.labri.fr/perso/pelegrin/scotch/>`_ 5.1+
- `Nose <https://nose.readthedocs.org/en/latest/>`_ 1.0+
- `gmsh <http://geuz.org/gmsh/>`_ 2.5+
- `VTK <http://vtk.org/>`_ 5.6+

Building document of SOLVCON requires the following packages:

- `pygraphviz <http://networkx.lanl.gov/pygraphviz/>`_ 1.1+
- `Sphinx <http://sphinx.pocoo.org/>`_ 1.1.2+
- `Sphinxcontrib issue tracker
  <http://sphinxcontrib-issuetracker.readthedocs.org/>`__ 0.11

You will also need `Mercurial <http://mercurial.selenic.com/>`_ (hg) to clone
the development codebase.

The following command will install the dependencies on Debian 7.x (wheezy):

.. literalinclude:: ../../contrib/aptget.debian.7wheezy.sh

On Ubuntu 12.04LTS please use:

.. literalinclude:: ../../contrib/aptget.ubuntu.12.04LTS.sh

Note: For Debian 6.x (squeeze), you need also ``apt-get install
python-profiler`` for the Python built-in profiler.

Download
========

It is suggested to use the in-development codebase.  Please clone the hg
repository::

  hg clone https://bitbucket.org/solvcon/solvcon

You can also obtain a tarball from
https://bitbucket.org/solvcon/solvcon/downloads.

Build
=====

The binary part of SOLVCON is built by using SCons_::

  cd $SCSRC
  scons

where ``$SCSRC`` indicates the root directory of unpacked source tree.

SOLVCON is designed to work without explicit installation.  You can simply set
the environment variable ``$PYTHONPATH`` to point to the source code, i.e.,
``$SCSRC``.  Note that the binary code is needed to be compiled.

.. note::

  The Python runtime will search the paths in the environment variable
  ``$PYTHONPATH`` for Python modules.  See
  http://docs.python.org/tutorial/modules.html#the-module-search-path for
  detail.

Run Tests
=========

If you have Nose_ installed, you can run::

  nosetests

in ``$SCSRC`` to run unit tests.  Another set of tests are collected in
``$SCSRC/ftests/`` directory, and can be run with::

  nosetests ftests/*

Some tests in ``$SCSRC/ftests/`` involve remote procedure call (RPC) by using
`ssh <http://www.openssh.com/>`_, so you need to set up the public key
authentication of ssh.

.. _manual-prerequisites:

Manually Build Prerequisites (Optional)
=======================================

On a cluster or a supercomputer, it is impossible for a user to use package
managers (e.g., apt or yum) to install the prerequisites.  It is also
time-consuming to ask support people to install those packages.  Building the
required software manually is the most feasible approach to get the
prerequisites.  SOLVCON provides a suite of scripts and makefiles to facilitate
the tedious process.

The ``$SCSRC/ground`` directory contains scripts to build most of the software
that SOLVCON depends on.  The ``$SCSRC/ground/get`` script downloads the source
packages to be built.  By default, the ``$SCSRC/ground/Makefile`` file does not
make large packages related to visualization, e.g., VTK.  Visualization
packages must be manually built by specifying the target ``vislib``.  The built
files will be automatically installed into the path specified by the
``$SCROOT`` environment variable, which is set to ``$HOME/opt/scruntime`` by
default.  The ``$SCROOT/bin/scvars.sh`` script will be created to export
necessary environment variables for the installed software, and the ``$SCROOT``
environment variable itself.

The ``$SCSRC/soil`` directory contains scripts to build gcc_.  The
``$SCROOT/bin/scgccvars.sh`` script will be created to export necessary
environment variables for the self-compiled gcc.  The enabled languages include
only C, C++, and Fortran.  The default value of ``$SCROOT`` remains to be
``$HOME/opt/scruntime``, while the software will be installed into
``$SCROOT/soil``.  Note: (i) Do not use different ``$SCROOT`` when building
``$SCSRC/soil`` and ``$SCSRC/ground``.  (ii) On hyper-threading CPUs the ``NP``
environment variable should be set to the actual number of cores, or
compilation of gcc could exhaust system memory.

``$SCROOT/bin/scvars.sh`` and ``$SCROOT/bin/scgccvars.sh`` can be separately
sourced.  The two sets of packages reside in different directories and do not
mix with each other nor system software.  Users can disable these environments
by not sourcing the two scripts.

Some packages have not been incorporated into the dependency building system
described above.  Debian or Ubuntu users should install the additional
dependencies by using::

  sudo apt-get install build-essential gcc gfortran gcc-multilib m4
  libreadline6 libreadline6-dev libncursesw5 libncurses5-dev libbz2-1.0
  libbz2-dev libdb4.8 libdb-dev libgdbm3 libgdbm-dev libsqlite3-0
  libsqlite3-dev libcurl4-gnutls-dev libhdf5-serial-dev libgl1-mesa-dev
  libxt-dev

These building scripts have only been tested with 64-bit Linux.

.. vim: set ft=rst ff=unix fenc=utf8 ai:
