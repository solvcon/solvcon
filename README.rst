SOLVCON is a collection of conservation-law solvers that use the space-time
`Conservation Element and Solution Element (CESE) method
<http://www.grc.nasa.gov/WWW/microbus/>`__.

Install
=======

Clone from https://github.com/solvcon/solvcon::

  $ git clone https://github.com/solvcon/solvcon

SOLVCON needs the following packages: A C/C++ compiler supporting C++14,
`Python <http://www.python.org/>`_ 2.7/3.4, `six
<https://pypi.python.org/pypi/six/>`_ 1.10.0, `pybind11
<https://github.com/pybind/pybind11>`_ Git master, `Cython
<http://www.cython.org/>`_ 0.16+, `Numpy <http://www.numpy.org/>`_ 1.5+,
`LAPACK <http://www.netlib.org/lapack/>`_, `NetCDF
<http://www.unidata.ucar.edu/software/netcdf/index.html>`_ 4+, `SCOTCH
<http://www.labri.fr/perso/pelegrin/scotch/>`_ 6.0+, `Nose
<https://nose.readthedocs.org/en/latest/>`_ 1.0+, `Paramiko
<https://github.com/paramiko/paramiko>`_ 1.14+, `boto
<http://boto.readthedocs.org/>`_ 2.29.1+, `gmsh <http://geuz.org/gmsh/>`_ 2.5+,
and `VTK <http://vtk.org/>`_ 5.6+.

A script at ``contrib/conda.sh`` is provided to install the dependency with
`Anaconda <https://store.continuum.io/cshop/anaconda/>`__.

The following command builds and installs SOLVCON::

  $ python setup.py install

After installed you can run the unit tests::

  $ python -c 'import solvcon as sc; sc.test()'

Additional build and tests:

- Building document requires `Sphinx <http://sphinx.pocoo.org/>`_ 1.3.1+,
  `pstake <http://pstake.readthedocs.org/>`_ 0.3.4+, `Sphinxcontrib issue
  tracker <http://sphinxcontrib-issuetracker.readthedocs.org/>`__ 0.11, and
  `graphviz <http://www.graphviz.org/>`_ 2.28+.  Once the binary of SOLVCON is
  built, the following commands can build the document::

    $ cd doc
    $ make html

  The built document will be available at ``doc/build/html/``.

- When building SOLVCON locally::

    $ python setup.py build_ext --inplace

  You can run unit tests with Nose_::

    $ nosetests

- Another set of tests are collected in ``ftests/`` directory, and can be run
  with::

    $ nosetests ftests/*

  Some tests in ``ftests/`` involve remote procedure call (RPC) that uses `ssh
  <http://www.openssh.com/>`_.  You need to set up the public key
  authentication to properly run them.

.. vim: set ft=rst ff=unix fenc=utf8:
