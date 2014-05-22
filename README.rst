|build_status|

.. |build_status| image:: https://drone.io/bitbucket.org/solvcon/solvcon/status.png

SOLVCON is a collection of `Python <http://www.python.org>`__-based
conservation-law solvers that use the space-time `Conservation Element and
Solution Element (CESE) method <http://www.grc.nasa.gov/WWW/microbus/>`__.
SOLVCON targets at solving problems that can be formulated as a system of
first-order, linear or non-linear partial differential equations (PDEs).

Get the Code and the Dependencies
=================================

Please clone the development version from `BitBucket
<https://bitbucket.org/solvcon/solvcon>`__ (using `Mercurial
<http://mercurial.selenic.com/>`_)::

  hg clone https://bitbucket.org/solvcon/solvcon

SOLVCON has the following dependencies: `gcc <http://gcc.gnu.org/>`_ 4.3+,
`SCons <http://www.scons.org/>`_ 2+, `Python <http://www.python.org/>`_ 2.7,
`Cython <http://www.cython.org/>`_ 0.16+, `Numpy <http://www.numpy.org/>`_
1.5+, `LAPACK <http://www.netlib.org/lapack/>`_, `NetCDF
<http://www.unidata.ucar.edu/software/netcdf/index.html>`_ 4+, `SCOTCH
<http://www.labri.fr/perso/pelegrin/scotch/>`_ 6.0+, `Nose
<https://nose.readthedocs.org/en/latest/>`_ 1.0+, `gmsh
<http://geuz.org/gmsh/>`_ 2.5+, and `VTK <http://vtk.org/>`_ 5.6+.  You can
install them by running the scripts ``aptget.*.sh`` (Debian/Ubuntu) or
``conda.sh`` (`Miniconda <http://conda.pydata.org/miniconda.html>`__/`Anaconda
<https://store.continuum.io/cshop/anaconda/>`__) provided in the ``contrib/``
directory.

Build
=====

The binary part of SOLVCON should be built with SCons_::

  scons scmods

After worth, it can be built with `distutils
<https://docs.python.org/2/distutils/>`__::

  python setup.py build_ext --inplace

SOLVCON needs not explicit installation.  Setting the environment variables
``$PATH`` and ``$PYTHONPATH`` is sufficient.

Building document requires `Sphinx <http://sphinx.pocoo.org/>`_ 1.1.2+,
`Sphinxcontrib issue tracker
<http://sphinxcontrib-issuetracker.readthedocs.org/>`__ 0.11, and `graphviz
<http://www.graphviz.org/>`_ 2.28+.  Once the binary of SOLVCON is built, the
following commands can build the document:

::

  cd doc
  make html

The built document will be available at ``doc/build/html/``.

Run Tests
=========

Tests should be run with Nose_::

  nosetests

Another set of tests are collected in ``ftests/`` directory, and can be run
with::

  nosetests ftests/*

Some tests in ``ftests/`` involve remote procedure call (RPC) that uses `ssh
<http://www.openssh.com/>`_.  You need to set up the public key authentication
to properly run them.

(Not Recommended) Build Dependencies from Source
================================================

A hard way to install the dependencies is to build everything from source with
the scripts provided in the ``ground/`` directory::

  cd ground
  ../contrib/get
  make all
  cd ..
  source opt/etc/scvars.sh

A directory ``opt/`` will be created for the binaries.  The last line will
enable the runtime environment.  It also export an environment variable
``SCROOT`` that points to ``opt/``.
  
If we don't even have a compatible gcc_, scripts in the ``soil/`` directory can
be used::

  cd soil
  ../contrib/get
  make
  cd ..
  source opt/etc/scgccvars.sh

``$SCROOT/etc/scvars.sh`` and ``$SCROOT/etc/scgccvars.sh`` must be separately
sourced.  The two sets of packages reside in different directories.

.. vim: set ft=rst ff=unix fenc=utf8:
