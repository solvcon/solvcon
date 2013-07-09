========
Tutorial
========

The goal of SOLVCON is to help code developers to focus on the numerical
algorithms.  These computing code can be written in C or other high-speed
languages and interfaced with SOLVCON.  SOLVCON has a general work flow that
support mesh loaders (`Gmsh <http://www.geuz.org/gmsh/>`__, FLUENT Gambit (R),
and `CUBIT <http://cubit.sandia.gov/>`__), MPI, and VTK.  These supportive
functionalities are ready to help developing problem solvers.

Set up the Environment
======================

Assume:

- SOLVCON is compiled without problems.  See :doc:`install` for more
  information.
- The compiled SOLVCON is located at ``$SCSRC``.
- You are using bash.

Usually we don't install SOLVCON into the OS, but use environment variable to
enable it, so that it's easier to modify the source code::

  export PYTHONPATH=$SCSRC:$PYTHONPATH

And then the following command::

  python -c "import solvcon; print solvcon"

should show you the correct location of SOLVCON.

There are various examples located at ``$SCSRC/examples``.  To follow the
examples, you need to:

- Install `ParaView <http://www.paraview.org/>`__.  On a Debian/Ubuntu, you can
  do it by executing::

    sudo apt-get install paraview

- Obtain example data.  You can do it by executing::

    scons --get-scdata

  in ``$SCSRC``.

More information of the verification examples can be found in
:doc:`verification`.

Configuration
=============

SOLVCON will find each of the solvcon.ini files from current working directory
toward the root, and use their settings.  Three settings are recognized in
``[SOLVCON]`` section:

- ``APPS``: Equivelent to the environment variable ``SOLVCON_APPS``.
- ``LOGFILE``: Equivelent to the environment variable ``SOLVCON_LOGFILE``.
- ``PROJECT_DIR``: Equivelent to the environment variable
  ``SOLVCON_PROJECT_DIR``.  Can be set to empty, which indicates the path where
  the configuration file locates.

The configurable environment variables:

- ``SOLVCON_PROJECT_DIR``: the directory holds the applications.
- ``SOLVCON_LOGFILE``: filename for solvcon logfile.
- ``SOLVCON_APPS``: names of the available applications, seperated with
  semi-colon.  There should be no spaces.
- ``SOLVCON_FPDTYPE``: a string for the numpy dtype object for floating-point.
  The default fpdtype to be used is float64 (double).
- ``SOLVCON_INTDTYPE``: a string for the numpy dtype object for integer.  The
  default intdtype to be used is int32.
- ``SOLVCON_MPI``: flag to use MPI.

Mesh Generation (to Be Added)
=============================

Before solving any PDE, you need to define the discretized spatial domain of
the problem by generating the mesh.

Example Solver (in Progress)
============================

To achieve high-performance in SOLVCON, the implementation of a numerical
method is divided into two parts: (i) a solver class and (ii) an algorithm
class.  A solver class is responsible for providing the API and managing
memory, while an algorithm class is responsible for number-crunching in C.
Users usually only see the solver class.  Intensive calculation is delegated to
the algorithm class from the solver class.  

.. note::

  For a PDE-solving method, code written in Python is in general two orders of
  magnitude slower than that written in C or Fortran.  And Cython code is still
  a bit (percentages or times) slower than C code.  Hence, in reality, we need
  to write C code for speed.

Two modules, :py:mod:`solvcon.fake_solver` and
:py:mod:`solvcon.fake_algorithm`, are put in SOLVCON to exemplify the
delegation structure by using a dummy numerical method.

.. py:module:: solvcon.fake_solver

The :py:mod:`solvcon.fake_solver` module contains the
:py:class:`FakeSolver` class that defines the API for the
dummy numerical method.

.. py:class:: FakeSolver

  This class represents the Python side of the numerical method.  It
  instantiates a :py:class:`solvcon.fake_algorithm.FakeAlgorithm` object.
  Computation-intensive tasks are delegated to the algorithm object.

  .. py:method:: create_alg

    Create a :py:class:`solvcon.fake_algorithm.FakeAlgorithm` object and return it.

  .. py:attribute:: MMNAMES

    An ordered registry for all names of methods to be called by a marcher.  Any
    methods to be called by a marcher should be registered into it.

  The following six methods are for the numerical methods.  They are registered
  into :py:attr:`MMNAMES` by the present order.

  .. py:method:: update

    Update the present solution arrays with the next solution arrays.

  .. py:method:: calcsoln

    Calculate the ``soln`` array.

  .. py:method:: ibcsoln

    Interchange BC for the ``soln`` array.

  .. py:method:: calccfl

    Calculate the CFL number.

  .. py:method:: calcdsoln

    Calculate the ``dsoln`` array.

  .. py:method:: ibcdsoln

    Interchange BC for the ``dsoln`` array.

.. py:module:: solvcon.fake_algorithm

The :py:mod:`solvcon.fake_algorithm` module contains the
:py:class:`FakeAlgorithm` that interfaces to the number-crunching C code.

.. py:class:: FakeAlgorithm

  This class represents the C side of the numerical method.  It wraps two C
  functions :c:func:`sc_fake_algorithm_calc_soln` and
  :c:func:`sc_fake_algorithm_calc_dsoln`.

  .. py:method:: setup_algorithm(svr)

    A :py:class:`FakeAlgorithm` object shouldn't allocate memory.  Instead, a
    :py:class:`solvcon.fake_solver.FakeSolver` object should allocate the memory
    and pass the solver into the algorithm.

  .. py:method:: calc_soln

    Wraps the C functions :c:func:`sc_fake_algorithm_calc_soln`.  Do the work
    delegated from :py:meth:`solvcon.fake_solver.FakeSolver.calcsoln`.

  .. py:method:: calc_dsoln

    Wraps the C functions :c:func:`sc_fake_algorithm_calc_dsoln`.  Do the work
    delegated from :py:meth:`solvcon.fake_solver.FakeSolver.calcdsoln`.
