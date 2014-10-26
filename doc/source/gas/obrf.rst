====================================================
Reflection of Oblique Shock Wave (Under Development)
====================================================

.. py:module:: solvcon.parcel.gas.oblique_shock

Consider an oblique shock wave hits a solid wall and reflects.  The system will
have two oblique shock waves and the relations of oblique shock will be applied
twice to obtain the flow properties in the total 3 zones.

The given data include:

1. The upstream (zone 1) Mach number :math:`M_1` and other flow properties
   (density, pressure, and temperature).
2. The first oblique shock angle :math:`\beta` (between zone 1 and 2) or the
   flow deflection angle :math:`\theta` (across zone 1/2 and zone 2/3).
   
When :math:`M_1` and one of the angle :math:`\theta` or :math:`\beta` are
given, the other angle can be calculated from
:py:meth:`ObliqueShockRelation.calc_flow_angle` or
:py:meth:`ObliqueShockRelation.calc_shock_angle`, respectively.

.. attention::

  I NEED A SCHEMATIC FOR THE REFLECTION PROBLEM.

Relations across Oblique Shock
==============================

Methods of calculating the shock relations are organized in the class
:py:class:`ObliqueShockRelation`.

.. attention::

  I NEED A SCHEMATIC FOR THE OBLIQUE SHOCK RELATIONS.

The oblique shock wave defines a rotated coordinate system :math:`(n, t)`,
where :math:`\hat{n}` is the unit vector normal to the oblique shock, and
:math:`\hat{t}` is the unit vector tangential to the shock.

.. autoclass:: ObliqueShockRelation

  .. autoinstanceattribute:: gamma

Density: :py:meth:`~ObliqueShockRelation.calc_density_ratio`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. automethod:: ObliqueShockRelation.calc_density_ratio

Pressure: :py:meth:`~ObliqueShockRelation.calc_pressure_ratio`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. automethod:: ObliqueShockRelation.calc_pressure_ratio

Temperature: :py:meth:`~ObliqueShockRelation.calc_temperature_ratio`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. automethod:: ObliqueShockRelation.calc_temperature_ratio

Mach Number: :py:meth:`~ObliqueShockRelation.calc_dmach`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. automethod:: ObliqueShockRelation.calc_dmach

Normal Mach Number: :py:meth:`~ObliqueShockRelation.calc_normal_dmach`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. automethod:: ObliqueShockRelation.calc_normal_dmach

Flow Angle: :py:meth:`~ObliqueShockRelation.calc_flow_angle`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. automethod:: ObliqueShockRelation.calc_flow_angle

.. automethod:: ObliqueShockRelation.calc_flow_tangent

Shock Angle: :py:meth:`~ObliqueShockRelation.calc_shock_angle`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. automethod:: ObliqueShockRelation.calc_shock_angle

.. automethod:: ObliqueShockRelation.calc_shock_tangent

.. automethod:: ObliqueShockRelation.calc_shock_tangent_aux

Numerical Simluation
====================

.. attention::

  TO BE WRITTEN:

  - Meshing.
  - Setting boundary-condition treatments.
  - Field initialization.
  - Physical variables calculation.
  - Data probing.
  - Field data output.
