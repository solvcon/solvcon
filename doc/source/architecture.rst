============
Architecture
============

SOLVCON divides a PDE solver into 5 layers.  At the top is the application
layer.  In this layer, users write scripts to drive all the underneath
functionalities to deliver the solutions and analysis.  The driving script can
be considered as the replacement of the traditional input files, but allows the
users to write code that alters the behaviors of the system.


.. figure:: _static/stack.png
  :width: 700px
  :alt: Architecture of SOLVCON
  :align: center

  Architecture of SOLVCON

.. py:module:: solvcon

Basic Constructs
================

.. py:module:: solvcon.mesh

.. py:class:: Mesh

  This class represents the data set of unstructured meshes of mixed elements.

.. py:module:: solvcon.mesh_solver

.. py:class:: MeshSolver(blk, **kw)

  This is the base class for all solver classes that use
  :py:class:`solvcon.mesh.Mesh`.

.. py:class:: MeshCase(**kw)

  This is the base class for all simulation cases that use
  :py:class:`MeshSolver` (and in turn :py:class:`solvcon.mesh.Mesh`).

  init() and run() are the two primary methods responsible for the
  execution of the simulation case object.  Both methods accept a keyword
  parameter ``level`` which indicates the run level of the run:

  - run level 0: fresh run (default),
  - run level 1: restart run,
  - run level 2: initialization only.

A Dummy Solver
==============

To achieve high-performance in SOLVCON, the implementation of a numerical
method is divided into two parts: (i) a solver class and (ii) an algorithm
class.  A solver class is responsible for providing the API and managing
memory, while an algorithm class is responsible for number-crunching in C.
Users usually only see the solver class.  Intensive calculation is delegated to
the algorithm class from the solver class.  Two modules,
:py:mod:`solvcon.fake_solver` and :py:mod:`solvcon.fake_algorithm`, are put in
SOLVCON to exemplify the delegation structure by using a dummy numerical
method.

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

.. vim: set spell ff=unix fenc=utf8 ft=rst:
