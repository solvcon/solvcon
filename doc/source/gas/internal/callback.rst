========
Callback
========

Useful callbacks are included in:

1. :py:mod:`solvcon.parcel.gas.probe`
2. :py:mod:`solvcon.parcel.gas.physics`
3. :py:mod:`solvcon.parcel.gas.inout`

:py:mod:`~solvcon.parcel.gas.probe`
===================================

.. py:module:: solvcon.parcel.gas.probe

.. autoclass:: ProbeHook

  .. inheritance-diagram:: ProbeHook

:py:mod:`~solvcon.parcel.gas.physics`
=====================================

.. py:module:: solvcon.parcel.gas.physics

.. autoclass:: DensityInitAnchor

  .. inheritance-diagram:: DensityInitAnchor

.. autoclass:: PhysicsAnchor

  .. inheritance-diagram:: PhysicsAnchor

:py:mod:`~solvcon.parcel.gas.inout`
===================================

.. py:module:: solvcon.parcel.gas.inout

.. autoclass:: MeshInfoHook

  .. inheritance-diagram:: MeshInfoHook

.. autoclass:: ProgressHook

  .. inheritance-diagram:: ProgressHook

.. autoclass:: FillAnchor

  .. inheritance-diagram:: FillAnchor

.. autoclass:: CflAnchor

  .. inheritance-diagram:: CflAnchor

  .. automethod:: __init__

.. autoclass:: CflHook

  .. inheritance-diagram:: CflHook

.. autoclass:: MarchSaveAnchor

  .. inheritance-diagram:: MarchSaveAnchor

.. autoclass:: PMarchSave

  .. inheritance-diagram:: PMarchSave
