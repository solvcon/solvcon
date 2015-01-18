====================================================
Reflection of Oblique Shock Wave (Under Development)
====================================================

.. py:module:: solvcon.parcel.gas.oblique_shock

This example solves a reflecting oblique shock wave, as shown in Figure
:num:`fig-reflection`.  The system consists of two oblique shock waves, which
separate the flow into three zones.  The incident shock results from a wedge.
The second reflects from a plane wall.  Flow properties in all the three zones
can be calculated with the following data:

1. The upstream (zone 1) Mach number :math:`M_1` and the flow properties
   density, pressure, and temperature.
2. The first oblique shock angle :math:`\beta_1` (between zone 1 and 2) or the
   flow deflection angle :math:`\theta` (across zone 1/2 and zone 2/3).  Only
   one of the angle is needed.  The other one can be calculated from the given
   one and :math:`M_1`.  The calculation detail is in
   :py:meth:`ObliqueShockRelation.calc_flow_angle` and
   :py:meth:`ObliqueShockRelation.calc_shock_angle`.

SOLVCON will be set up to solve this problem, and the simulated results will be
compared with the analytical solution.  The relation of flow properties across
each oblique shock can be analytically obtained [Anderson03]_.

.. _fig-reflection:

.. figure:: reflection.png
  :align: center

  Oblique shock reflected from a wall

  :math:`M_{1,2,3}` are the Mach number in the corresponding zone 1, 2, and 3.
  :math:`\theta` is the flow deflection angle.  :math:`\beta_{1,2}` are the
  oblique shock angle behind the first and the second zone, respectively.

Relations across Oblique Shock
==============================

An oblique shock is resulted from a sudden change of flow direction, as shown
in Figure :num:`fig-oblique-shock`.

.. _fig-oblique-shock:

.. figure:: oblique_shock.png
  :align: center

  Oblique shock wave by a wedge

  :math:`M` is Mach number.  :math:`\theta` is the flow deflection angle.
  :math:`\beta` is the oblique shock angle.

The notation and derivations from Section 4.3 *Oblique Shock Relations* of
[Anderson03]_ are used.  :math:`\square_1` denotes upstream properties and
:math:`\square_2` denotes downstream properties.  Two important angles are
defined:

1. :math:`\beta`: The angle of the oblique shock wave deflected from the
   upstream is :math:`\beta`; the shock angle.
2. :math:`\theta`: The angle of the flow behind the shock wave deflected from
   the upstream is :math:`\theta`; the flow angle.

Methods of calculating the shock relations are organized in the class
:py:class:`ObliqueShockRelation`.  Derivation of the relation uses a rotated
coordinate system :math:`(n, t)` defined by the oblique shock, where
:math:`\hat{n}` is the unit vector normal to the shock, and :math:`\hat{t}` is
the unit vector tangential to the shock.  Figure :num:`fig-oblique-relation` is
useful in the derivation of the relations.

.. _fig-oblique-relation:

.. figure:: oblique_relation.png
  :align: center

  Properties across an oblique shock

  The flow properties in the upstream zone of the oblique shock are :math:`v_1,
  M_1, \rho_1, p_1, T_1`.  Those in the downstream zone of the shock are
  :math:`v_2, M_2, \rho_2, p_2, T_2`.

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
