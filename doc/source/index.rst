============================
Solvers of Conservation Laws
============================

SOLVCON is a collection of `Python <http://www.python.org>`__-based
conservation-law solvers that use the space-time `Conservation Element and
Solution Element (CESE) method <http://www.grc.nasa.gov/WWW/microbus/>`__.
SOLVCON targets at solving problems that can be formulated as a system of
first-order, linear or non-linear partial differential equations (PDEs)
[Lax73]_:

.. math::

  \dpd{\bvec{u}}{t}
  + \sum_{\iota=1}^3 \mathrm{A}^{(\iota)}(\bvec{u})\dpd{\bvec{u}}{x_{\iota}}
  = \bvec{s}(\bvec{u})

where :math:`\bvec{u}` is the unknown vector, :math:`\mathrm{A}^{(1)}`,
:math:`\mathrm{A}^{(2)}`, and :math:`\mathrm{A}^{(3)}` the Jacobian matrices,
and :math:`\bvec{s}` the source term.  SOLVCON is designed to be a software
framework to house various solvers [Chen11]_.  The design of SOLVCON is also
discussed in [Chen11]_ and you can use it to cite the software.

- Visit the project page https://bitbucket.org/solvcon/solvcon
- Report bugs and request features at
  https://bitbucket.org/solvcon/solvcon/issues?status=new&status=open
- Ask questions in our `mailing list
  <http://groups.google.com/group/solvcon>`__: solvcon@googlegroups.com

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
- `SCOTCH <http://www.labri.fr/perso/pelegrin/scotch/>`_ 6.0+
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

The following command will install the dependencies on Debian jessie:

.. literalinclude:: ../../contrib/aptget.debian.jessie.sh

On Ubuntu 12.04LTS please use:

.. literalinclude:: ../../contrib/aptget.ubuntu.12.04LTS.sh

Note: For Debian 6.x (squeeze), you need also ``apt-get install
python-profiler`` for the Python built-in profiler.

SOLVCON can also run on Mac OS X 10.9.

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

Manually Build Prerequisites (Optional)
+++++++++++++++++++++++++++++++++++++++

On a cluster or a supercomputer, it is impossible for a user to use package
managers (e.g., apt or yum) to install the prerequisites.  It is also
time-consuming to ask support people to install those packages.  Building the
required software manually is the most feasible approach to get the
prerequisites.  SOLVCON provides a suite of scripts and makefiles to facilitate
the tedious process.

The ``$SCSRC/ground`` directory contains scripts to build most of the software
that SOLVCON depends on.  The ``$SCSRC/contrib/get`` script downloads the
source packages to be built.  By default, the ``$SCSRC/ground/Makefile`` file
does not make large packages related to visualization, e.g., VTK.
Visualization packages must be manually built by specifying the target
``vislib``.  The built files will be automatically installed into the path
specified by the ``$SCROOT`` environment variable, which is set to
``$HOME/opt/scruntime`` by default.  The ``$SCROOT/bin/scvars.sh`` script will
be created to export necessary environment variables for the installed
software, and the ``$SCROOT`` environment variable itself.

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

Documentation
=============

.. toctree::
  :maxdepth: 2

  tutorial

Architecture Reference
++++++++++++++++++++++

.. toctree::
  :maxdepth: 3

  architecture
  inout
  system_modules

Application Reference
+++++++++++++++++++++

.. toctree::
  :maxdepth: 2

  app_linear

Development Support
+++++++++++++++++++

.. toctree::
  :maxdepth: 1

  python_style
  plan
  verification

Other Resources
+++++++++++++++

- Papers and presentations:

  - :doc:`pub_app`
  - `PyCon US 2011 talk
    <http://us.pycon.org/2011/schedule/presentations/50/>`__: `slides
    <http://solvcon.net/slide/PyCon11_yyc.pdf>`__ and `video
    <http://pycon.blip.tv/file/4882902/>`__
  - Yung-Yu Chen, David Bilyeu, Lixiang Yang, and Sheng-Tao John Yu,
    "SOLVCON: A Python-Based CFD Software Framework for Hybrid
    Parallelization",
    *49th AIAA Aerospace Sciences Meeting*,
    January 4-7 2011, Orlando, Florida.
    `AIAA Paper 2011-1065
    <http://pdf.aiaa.org/preview/2011/CDReadyMASM11_2388/PV2011_1065.pdf>`_
- The CESE method:

  - The CE/SE working group: http://www.grc.nasa.gov/WWW/microbus/
  - The CESE research group at OSU: http://cfd.solvcon.net/research.html
  - Selected papers:

    - Sin-Chung Chang, "The Method of Space-Time Conservation Element and
      Solution Element -- A New Approach for Solving the Navier-Stokes and
      Euler Equations", *Journal of Computational Physics*, Volume 119, Issue
      2, July 1995, Pages 295-324.  `doi: 10.1006/jcph.1995.1137
      <http://dx.doi.org/10.1006/jcph.1995.1137>`_
    - Xiao-Yen Wang, Sin-Chung Chang, "A 2D Non-Splitting Unstructured
      Triangular Mesh Euler Solver Based on the Space-Time Conservation Element
      and Solution Element Method", *Computational Fluid Dynamics Journal*,
      Volume 8, Issue 2, 1999, Pages 309-325.
    - Zeng-Chan Zhang, S. T. John Yu, Sin-Chung Chang, "A Space-Time
      Conservation Element and Solution Element Method for Solving the Two- and
      Three-Dimensional Unsteady Euler Equations Using Quadrilateral and
      Hexahedral Meshes", *Journal of Computational Physics*, Volume 175, Issue
      1, Jan. 2002, Pages 168-199.  `doi: 10.1006/jcph.2001.6934
      <http://dx.doi.org/10.1006/jcph.2001.6934>`_
- :doc:`link`
- :doc:`link_other`

Bibliography
============

.. [Lax73] Peter D. Lax, *Hyperbolic Systems of Conservation Laws and the
  Mathematical Theory of Shock Waves*, Society for Industrial Mathematics,
  1973.  `ISBN 0898711770
  <http://www.worldcat.org/title/hyperbolic-systems-of-conservation-laws-and-the-mathematical-theory-of-shock-waves/oclc/798365>`__.

.. [Chen11] Yung-Yu Chen, *A Multi-Physics Software Framework on Hybrid
  Parallel Computing for High-Fidelity Solutions of Conservation Laws*, Ph.D.
  Thesis, The Ohio State University, United States, Aug. 2011. (`OhioLINK
  <http://rave.ohiolink.edu/etdc/view?acc_num=osu1313000975>`__)

Appendices
==========

Copyright Notice
++++++++++++++++

.. include:: ../../COPYING

Release History
+++++++++++++++

.. toctree::
   :maxdepth: 2

   history

Indices and Tables
++++++++++++++++++

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. vim: set spell ft=rst ff=unix fenc=utf8:
