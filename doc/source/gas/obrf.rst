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
compared with the analytical solution.  The :ref:`shock relation
<sec-oblique-shock-relation>` needs to be applied multiple times to obtain the
flow properties in all the three zones.

.. _fig-reflection:

.. figure:: reflection.png
  :align: center

  Oblique shock reflected from a wall

  :math:`M_{1,2,3}` are the Mach number in the corresponding zone 1, 2, and 3.
  :math:`\theta` is the flow deflection angle.  :math:`\beta_{1,2}` are the
  oblique shock angle behind the first and the second zone, respectively.

.. _sec-oblique-shock-relation:

:py:class:`ObliqueShockRelation`
================================

An oblique shock results from a sudden change of direction of supersonic flow.
The relations of density (:math:`\rho`), pressure (:math:`p`), and temprature
(:math:`T`) across the shock can be obtained analytically [Anderson03]_.  In
addition, two angles are defined:

1. The angle of the oblique shock wave deflected from the upstream is
   :math:`\beta`; the shock angle.
2. The angle of the flow behind the shock wave deflected from the upstream is
   :math:`\theta`; the flow angle.

See Figure :num:`fig-oblique-shock` for the illustration of the two angles.

.. _fig-oblique-shock:

.. figure:: oblique_shock.png
  :align: center

  Oblique shock wave by a wedge

  :math:`M` is Mach number.  :math:`\theta` is the flow deflection angle.
  :math:`\beta` is the oblique shock angle.

Methods of calculating the shock relations are organized in the class
:py:class:`ObliqueShockRelation`.  To obtain the relations of density
(:math:`\rho`), pressure (:math:`p`), and temprature (:math:`T`), the control
volume across the shock is emplyed, as shown in Figure
:num:`fig-oblique-relation`.  In the figure and in
:py:class:`ObliqueShockRelation`, subscript 1 denotes upstream properties and
subscript 2 denotes downstream properties.  Derivation of the relation uses a
rotated coordinate system :math:`(n, t)` defined by the oblique shock, where
:math:`\hat{n}` is the unit vector normal to the shock, and :math:`\hat{t}` is
the unit vector tangential to the shock.  But in this document we won't go into
the detail.

.. _fig-oblique-relation:

.. figure:: oblique_relation.png
  :align: center

  Properties across an oblique shock

  The flow properties in the upstream zone of the oblique shock are :math:`v_1,
  M_1, \rho_1, p_1, T_1`.  Those in the downstream zone of the shock are
  :math:`v_2, M_2, \rho_2, p_2, T_2`.

.. autoclass:: ObliqueShockRelation

  .. autoinstanceattribute:: gamma
    :annotation:

:py:class:`ObliqueShockRelation` provides three methods to calculate the ratio
of flow properties across the shock.  :math:`M_1` and :math:`\beta` are
required arguments:

- :math:`\rho`: :py:meth:`~ObliqueShockRelation.calc_density_ratio`
- :math:`p`: :py:meth:`~ObliqueShockRelation.calc_pressure_ratio`
- :math:`T`: :py:meth:`~ObliqueShockRelation.calc_temperature_ratio`

With :math:`M_1` available, the shock angle :math:`\beta` can be calculated
from the flow angle :math:`\theta`, or vice versa, by using the following two
methods:

- :math:`\beta`: :py:meth:`ObliqueShockRelation.calc_shock_angle`
- :math:`\theta`: :py:meth:`ObliqueShockRelation.calc_flow_angle`

The following method calculates the downstream Mach number, with the upstream
Mach number :math:`M_1` and either of :math:`\beta` or :math:`\theta` supplied:

- :math:`M_2`: :py:meth:`~ObliqueShockRelation.calc_dmach`

Numerical Simluation
====================

Reference to the Methods of :py:class:`ObliqueShockRelation`
============================================================

Listing of all methods:

- :py:meth:`~ObliqueShockRelation.calc_density_ratio`
- :py:meth:`~ObliqueShockRelation.calc_pressure_ratio`
- :py:meth:`~ObliqueShockRelation.calc_temperature_ratio`
- :py:meth:`~ObliqueShockRelation.calc_dmach`
- :py:meth:`~ObliqueShockRelation.calc_normal_dmach`
- :py:meth:`~ObliqueShockRelation.calc_flow_angle`
- :py:meth:`~ObliqueShockRelation.calc_flow_tangent`
- :py:meth:`~ObliqueShockRelation.calc_shock_angle`
- :py:meth:`~ObliqueShockRelation.calc_shock_tangent`
- :py:meth:`~ObliqueShockRelation.calc_shock_tangent_aux`

.. automethod:: ObliqueShockRelation.calc_density_ratio

.. automethod:: ObliqueShockRelation.calc_pressure_ratio

.. automethod:: ObliqueShockRelation.calc_temperature_ratio

.. automethod:: ObliqueShockRelation.calc_dmach

.. automethod:: ObliqueShockRelation.calc_normal_dmach

.. automethod:: ObliqueShockRelation.calc_flow_angle

.. automethod:: ObliqueShockRelation.calc_flow_tangent

.. automethod:: ObliqueShockRelation.calc_shock_angle

.. automethod:: ObliqueShockRelation.calc_shock_tangent

.. automethod:: ObliqueShockRelation.calc_shock_tangent_aux

.. attention::

  TO BE WRITTEN:

  - Meshing.
  - Setting boundary-condition treatments.
  - Field initialization.
  - Physical variables calculation.
  - Data probing.
  - Field data output.
