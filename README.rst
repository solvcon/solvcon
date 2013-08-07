|build_status|

.. |build_status| image:: https://drone.io/bitbucket.org/solvcon/solvcon/status.png

SOLVCON is a collection of `Python <http://www.python.org>`__-based
conservation-law solvers that use the space-time `Conservation Element and
Solution Element (CESE) method <http://www.grc.nasa.gov/WWW/microbus/>`__.
SOLVCON targets at solving problems that can be formulated as a system of
first-order, linear or non-linear partial differential equations (PDEs)

Install
=======

Please use the development version in the Mercurial_ repository::

  hg clone https://bitbucket.org/solvcon/solvcon

Released source tarballs can be downloaded from
https://bitbucket.org/solvcon/solvcon/downloads, but the development version is
recommended.

Prerequisites
+++++++++++++

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

The following command will install the dependencies on Debian jessie::

    sudo apt-get install build-essential gcc gfortran scons \
    liblapack-pic liblapack-dev libnetcdf-dev libnetcdfc7 netcdf-bin \
    libscotch-dev libscotchmetis-dev libscotch-5.1 \
    python2.7 python2.7-dev cython python-numpy python-nose gmsh python-vtk \
    python-pygraphviz python-sphinx python-sphinxcontrib.issuetracker \
    mercurial

On Ubuntu 12.04LTS please use::

    sudo apt-get install build-essential gcc gfortran scons \
    liblapack-pic liblapack-dev libnetcdf-dev libnetcdf6 netcdf-bin \
    libscotch-dev libscotchmetis-dev libscotch-5.1 \
    python2.7 python2.7-dev cython python-numpy python-nose gmsh python-vtk \
    python-pygraphviz python-sphinx python-sphinxcontrib.issuetracker \
    mercurial

Note: For Debian 6.x (squeeze), you need also ``apt-get install
python-profiler`` for the Python built-in profiler.

Build
+++++

The binary part of SOLVCON should be built with SCons_::

  cd $SCSRC
  scons

where ``$SCSRC`` indicates the root directory of unpacked source tree.

The source tarball supports distutils and can built alternatively::

  python setup.py build_ext --inplace

SOLVCON is designed to work without explicit installation.  You can simply set
the environment variable ``$PYTHONPATH`` to point to the source code, i.e.,
``$SCSRC``.  Note that the binary code is needed to be compiled.

.. note::

  The Python runtime will search the paths in the environment variable
  ``$PYTHONPATH`` for Python modules.  See
  http://docs.python.org/tutorial/modules.html#the-module-search-path for
  detail.

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