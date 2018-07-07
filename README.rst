SOLVCON implements conservation-law solvers that use the space-time
`Conservation Element and Solution Element (CESE) method
<http://www.grc.nasa.gov/WWW/microbus/>`__.

|travis_status| |gitlab_status| |rtd_status|

.. |travis_status| image:: https://travis-ci.org/solvcon/solvcon.svg?branch=master
  :target: https://travis-ci.org/solvcon/solvcon
  :alt: Travis-CI Status

.. |gitlab_status| image:: https://gitlab.com/solvcon/solvcon/badges/master/build.svg
  :target: https://gitlab.com/solvcon/solvcon/pipelines
  :alt: Gitlab-CI Status

.. |rtd_status| image:: https://readthedocs.org/projects/solvcon/badge/?version=latest
  :target: http://doc.solvcon.net/en/latest/
  :alt: Documentation Status

Install
=======

Clone from https://github.com/solvcon/solvcon::

  $ git clone https://github.com/solvcon/solvcon

SOLVCON needs the following packages: A C/C++ compiler supporting C++11, `cmake
<https://cmake.org>`_ 3.7+, `pybind11 <https://github.com/pybind/pybind11>`_
Git master, `Python <http://www.python.org/>`_ 3.6+, `Cython
<http://www.cython.org/>`_ 0.16+, `Numpy <http://www.numpy.org/>`_ 1.5+,
`LAPACK <http://www.netlib.org/lapack/>`_, `NetCDF
<http://www.unidata.ucar.edu/software/netcdf/index.html>`_ 4+, `SCOTCH
<http://www.labri.fr/perso/pelegrin/scotch/>`_ 6.0+, `Nose
<https://nose.readthedocs.org/en/latest/>`_ 1.0+, `Paramiko
<https://github.com/paramiko/paramiko>`_ 1.14+, `boto
<http://boto.readthedocs.org/>`_ 2.29.1+, and `gmsh <http://geuz.org/gmsh/>`_
3+.  Support for `VTK <http://vtk.org/>`_ is to be enabled for conda
environment.

To install the dependency, run the scripts ``contrib/conda.sh`` and
``contrib/build-pybind11-in-conda.sh`` (they use `Anaconda
<https://www.anaconda.com/download/>`__).

The development version of SOLVCON only supports local build::

  $ python setup.py build_ext --inplace

Test the build::

  $ nosetests --with-doctest
  $ nosetests ftests/gasplus/*

Building document requires `Sphinx <http://sphinx.pocoo.org/>`_ 1.3.1+, `pstake
<http://pstake.readthedocs.org/>`_ 0.3.4+, and `graphviz
<http://www.graphviz.org/>`_ 2.28+.  Use the following command::

  $ make -C doc html

The document will be available at ``doc/build/html/``.
