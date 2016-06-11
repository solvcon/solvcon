SOLVCON implements conservation-law solvers that use the space-time
`Conservation Element and Solution Element (CESE) method
<http://www.grc.nasa.gov/WWW/microbus/>`__.

|travis_status| |rtd_status|

.. |travis_status| image:: https://travis-ci.org/solvcon/solvcon.svg?branch=master
  :target: https://travis-ci.org/solvcon/solvcon
  :alt: Travis-CI Status

.. |rtd_status| image:: https://readthedocs.org/projects/solvcon/badge/?version=latest
  :target: http://doc.solvcon.net/en/latest/
  :alt: Documentation Status

Install
=======

Clone from https://github.com/solvcon/solvcon::

  $ git clone https://github.com/solvcon/solvcon

SOLVCON needs the following packages: A C/C++ compiler supporting C++14,
`pybind11 <https://github.com/pybind/pybind11>`_ Git master, `Python
<http://www.python.org/>`_ 2.7/3.5, `six <https://pypi.python.org/pypi/six/>`_
1.10.0, `Cython <http://www.cython.org/>`_ 0.16+, `Numpy
<http://www.numpy.org/>`_ 1.5+, `LAPACK <http://www.netlib.org/lapack/>`_,
`NetCDF <http://www.unidata.ucar.edu/software/netcdf/index.html>`_ 4+, `SCOTCH
<http://www.labri.fr/perso/pelegrin/scotch/>`_ 6.0+, `Nose
<https://nose.readthedocs.org/en/latest/>`_ 1.0+, `Paramiko
<https://github.com/paramiko/paramiko>`_ 1.14+, `boto
<http://boto.readthedocs.org/>`_ 2.29.1+, `gmsh <http://geuz.org/gmsh/>`_ 2.5+,
and `VTK <http://vtk.org/>`_ 5.6+.

A script at ``contrib/conda.sh`` is provided to install the dependency with
`Anaconda <https://store.continuum.io/cshop/anaconda/>`__.

The following command builds and installs SOLVCON::

  $ python setup.py install

Additional notes:

- Unit tests need to be run with local build::

    $ python setup.py build_ext --inplace
    $ nosetests --with-doctest

- Building document requires `Sphinx <http://sphinx.pocoo.org/>`_ 1.3.1+,
  `pstake <http://pstake.readthedocs.org/>`_ 0.3.4+, `Sphinxcontrib issue
  tracker <http://sphinxcontrib-issuetracker.readthedocs.org/>`__ 0.11, and
  `graphviz <http://www.graphviz.org/>`_ 2.28+.  Once the binary of SOLVCON is
  built, the following commands can build the document::

    $ make -C doc html

  The document will be available at ``doc/build/html/``.
