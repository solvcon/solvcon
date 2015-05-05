=======================================
Gas Dynamics Parcel (Under Development)
=======================================

.. py:module:: solvcon.parcel.gas

This parcel :py:mod:`solvcon.parcel.gas` is for gas dynamics.  It currently
uses the Euler equations, but will be updated to use the Navier-Stokes
equations.

.. toctree::

  euler
  oblique_shock_reflection

Simulation Settings
===================

.. py:method:: register_arrangement

  This is an alias to the instance method
  :py:meth:`GasCase.register_arrangement`, which inherits from the class
  :py:class:`solvcon.MeshCase`.  See
  :py:meth:`solvcon.MeshCase.register_arrangement` for implementation detail.

.. py:class:: GasCase(**kw)

  See :py:class:`case.GasCase` for implementation detail.

.. py:class:: GasSolver(blk, **kw)

  See :py:class:`solver.GasSolver` for implementation detail.

Boundary-Condition Treatments
=============================

.. py:class:: solvcon.parcel.gas.GasNonrefl

  See :py:class:`boundcond.GasNonrefl` for implementaion detail.

.. py:class:: solvcon.parcel.gas.GasWall

  See :py:class:`boundcond.GasWall` for implementation detail.

.. py:class:: solvcon.parcel.gas.GasInlet

  See :py:class:`boundcond.GasInlet` for implementation detail.

Callbacks
=========

.. py:class:: ProbeHook

  See :py:class:`probe.ProbeHook` for implementation detail.

.. py:class:: DensityInitAnchor

  See :py:class:`physics.DensityInitAnchor` for implementation detail.

.. py:class:: PhysicsAnchor

  See :py:class:`physics.PhysicsAnchor` for implementation detail.

.. py:class:: MeshInfoHook

  See :py:class:`inout.MeshInfoHook` for implementation detail.

.. py:class:: ProgressHook

  See :py:class:`inout.ProgressHook` for implementation detail.

.. py:class:: FillAnchor

  See :py:class:`inout.FillAnchor` for implementation detail.

.. py:class:: CflHook

  See :py:class:`inout.CflHook` for implementation detail.

.. py:class:: PMarchSave

  See :py:class:`inout.PMarchSave` for implementation detail.

Internal Refenrence
===================

.. toctree::
  :maxdepth: 2

  internal/simulation
  internal/boundcond
  internal/callback
