=======
Solvers
=======

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

.. autoclass:: solvcon.fake_algorithm.FakeAlgorithm
  :members:
  :special-members:

.. automodule:: solvcon.fake_solver
  :members:
  :special-members:

.. vim: set spell ff=unix fenc=utf8 ft=rst:
