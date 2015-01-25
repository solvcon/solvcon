|build_status|

.. |build_status| image:: https://drone.io/bitbucket.org/solvcon/solvcon/status.png

SOLVCON is a collection of `Python <http://www.python.org>`__-based
conservation-law solvers that use the space-time `Conservation Element and
Solution Element (CESE) method <http://www.grc.nasa.gov/WWW/microbus/>`__.  It
targets at solving problems that can be formulated as a system of first-order,
linear or non-linear partial differential equations (PDEs).

Install
=======

Clone the `hg <http://mercurial.selenic.com/>`_ repository from
https://bitbucket.org/solvcon/solvcon::

  $ hg clone https://bitbucket.org/solvcon/solvcon

SOLVCON needs the following packages: `gcc <http://gcc.gnu.org/>`_ 4.3+ (clang
on OSX works as well), `SCons <http://www.scons.org/>`_ 2+, `Python
<http://www.python.org/>`_ 2.7, `Cython <http://www.cython.org/>`_ 0.16+,
`Numpy <http://www.numpy.org/>`_ 1.5+, `LAPACK
<http://www.netlib.org/lapack/>`_, `NetCDF
<http://www.unidata.ucar.edu/software/netcdf/index.html>`_ 4+, `SCOTCH
<http://www.labri.fr/perso/pelegrin/scotch/>`_ 6.0+, `Nose
<https://nose.readthedocs.org/en/latest/>`_ 1.0+, `Paramiko
<https://github.com/paramiko/paramiko>`_ 1.14+, `boto
<http://boto.readthedocs.org/>`_ 2.29.1+, `gmsh <http://geuz.org/gmsh/>`_ 2.5+,
and `VTK <http://vtk.org/>`_ 5.6+.

In the ``contrib/`` directory, you can find the scripts for installing these
dependencies:

- ``aptget.*.sh`` for Debian/Ubuntu
- ``conda.sh`` for `Miniconda
  <http://conda.pydata.org/miniconda.html>`__/`Anaconda
  <https://store.continuum.io/cshop/anaconda/>`__

The binary part of SOLVCON should be built with SCons_::

  $ scons scmods

Then add the installation path to the environment variables ``$PATH`` and
``$PYTHONPATH``.

Additional build and tests:

- Building document requires `Sphinx <http://sphinx.pocoo.org/>`_ 1.1.2+,
  `Sphinxcontrib issue tracker
  <http://sphinxcontrib-issuetracker.readthedocs.org/>`__ 0.11, and `graphviz
  <http://www.graphviz.org/>`_ 2.28+.  Once the binary of SOLVCON is built, the
  following commands can build the document::

    $ cd doc
    $ make html

  The built document will be available at ``doc/build/html/``.

- Unit tests should be run with Nose_::

    $ nosetests

- Another set of tests are collected in ``ftests/`` directory, and can be run
  with::

    $ nosetests ftests/*

  Some tests in ``ftests/`` involve remote procedure call (RPC) that uses `ssh
  <http://www.openssh.com/>`_.  You need to set up the public key
  authentication to properly run them.

.. vim: set ft=rst ff=unix fenc=utf8:
