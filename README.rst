|build_status|

.. |build_status| image:: https://drone.io/bitbucket.org/solvcon/solvcon/status.png

SOLVCON is a collection of `Python <http://www.python.org>`__-based
conservation-law solvers that use the space-time `Conservation Element and
Solution Element (CESE) method <http://www.grc.nasa.gov/WWW/microbus/>`__.
SOLVCON targets at solving problems that can be formulated as a system of
first-order, linear or non-linear partial differential equations (PDEs).

Install
=======

Please use the development version in the `Mercurial
<http://mercurial.selenic.com/>`_ repository::

  hg clone https://bitbucket.org/solvcon/solvcon

Prerequisites
+++++++++++++

Like many scientific packages, SOLVCON has a lot of dependencies: `gcc
<http://gcc.gnu.org/>`_ 4.3+, `SCons <http://www.scons.org/>`_ 2+, `Python
<http://www.python.org/>`_ 2.7, `Cython <http://www.cython.org/>`_ 0.16+,
`Numpy <http://www.numpy.org/>`_ 1.5+, `LAPACK
<http://www.netlib.org/lapack/>`_, `NetCDF
<http://www.unidata.ucar.edu/software/netcdf/index.html>`_ 4+, `SCOTCH
<http://www.labri.fr/perso/pelegrin/scotch/>`_ 6.0+, `Nose
<https://nose.readthedocs.org/en/latest/>`_ 1.0+, `gmsh
<http://geuz.org/gmsh/>`_ 2.5+, `VTK <http://vtk.org/>`_ 5.6+.  Building
document requires: `pygraphviz <http://networkx.lanl.gov/pygraphviz/>`_ 1.1+,
`Sphinx <http://sphinx.pocoo.org/>`_ 1.1.2+, `Sphinxcontrib issue tracker
<http://sphinxcontrib-issuetracker.readthedocs.org/>`__ 0.11.

The hard way to install the dependencies is to build them manually, but SOLVCON
has a ``$SCSRC/ground/`` directory that provides scripts for some help::

  cd $SCSRC/ground
  ../contrib/get
  make all
  source $SCSRC/opt/etc/scvars.sh

A directory ``$SCSRC/opt`` will be created to hold the built binaries.  The
last line will enable the runtime environment, and also ``export
SCROOT=$SCSRC/opt``.  If we don't even have a compatible gcc_, the
``$SCSRC/soil/`` directory can help::

  cd $SCSRC/soil
  ../contrib/get
  make
  source $SCSRC/opt/etc/scgccvars.sh

``$SCROOT/etc/scvars.sh`` and ``$SCROOT/etc/scgccvars.sh`` can be separately
sourced.  The two sets of packages reside in different directories and do not
mix with each other nor system software.

With a package management system, life can be easier.  On Debian jessie, the
following command will install the dependencies:

.. literalinclude:: ../../contrib/aptget.debian.jessie.sh

and on Ubuntu 12.04LTS please use:

.. literalinclude:: ../../contrib/aptget.ubuntu.12.04LTS.sh

on Ubuntu 14.04LTS please use:

.. literalinclude:: ../../contrib/aptget.ubuntu.14.04LTS.sh

.. note::

  On Debian 6.x (squeeze), you need also ``apt-get install python-profiler``
  for the Python built-in profiler.

Build
+++++

The binary part of SOLVCON should be built with SCons_::

  cd $SCSRC
  scons

The source tarball supports distutils and can be built alternatively::

  python setup.py build_ext --inplace

SOLVCON is designed to work without explicit installation.  Setting the
environment variables ``$PATH`` and ``$PYTHONPATH`` is sufficient.

Run Tests
+++++++++

Tests should be run with Nose_::

  nosetests

in the project root directory ``$SCSRC``.  Another set of tests are collected
in ``$SCSRC/ftests/`` directory, and can be run with::

  nosetests ftests/*

Some tests in ``$SCSRC/ftests/`` involve remote procedure call (RPC) that uses
`ssh <http://www.openssh.com/>`_, so you need to set up the public key
authentication of ssh.

.. vim: set ft=rst ff=unix fenc=utf8:
