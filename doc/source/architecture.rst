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

A Simple Dummy Solver
=====================

A ``*_solver`` module and a ``*_algorithm`` module are in companion.  The
``*_solver`` module lays out how a numerical method to be applied to a specific
group of physical problems.  The ``*_algorithm`` is responsible to implement
the efficient number-crunching code of the numerical method.  A ``*_algorithm``
module is usually implemented by using Cython, and calls underlying C code.

The modules :py:mod:`solvcon.fake_solver` and :py:mod:`solvcon.fake_algorithm`
are an example to this structure.  :py:mod:`solvcon.fake_algorithm` wraps two C
functions :c:func:`sc_fake_algorithm_calc_soln` and
:c:func:`sc_fake_algorithm_calc_dsoln`, and then create a class
:py:class:`solvcon.fake_algorithm.FakeAlgorithm`.  In
:py:mod:`solvcon.fake_solver`, the class
:py:class:`solvcon.fake_solver.FakeSolver` uses its own data to instantiate a
:py:mod:`solvcon.fake_algorithm.FakeAlgorithm` object for number-crunching.

.. vim: set spell ff=unix fenc=utf8 ft=rst:
