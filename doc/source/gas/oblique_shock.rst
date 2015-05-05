:orphan:

.. py:module:: solvcon.parcel.gas.oblique_shock

======================
Oblique Shock Relation
======================

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

.. pstake:: oblique_shock.tex
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

.. pstake:: oblique_relation.tex
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

:math:`\rho`
  :py:meth:`~ObliqueShockRelation.calc_density_ratio`

:math:`p`
  :py:meth:`~ObliqueShockRelation.calc_pressure_ratio`

:math:`T`
  :py:meth:`~ObliqueShockRelation.calc_temperature_ratio`

With :math:`M_1` available, the shock angle :math:`\beta` can be calculated
from the flow angle :math:`\theta`, or vice versa, by using the following two
methods:

:math:`\beta`
  :py:meth:`ObliqueShockRelation.calc_shock_angle`

:math:`\theta`
  :py:meth:`ObliqueShockRelation.calc_flow_angle`

The following method calculates the downstream Mach number, with the upstream
Mach number :math:`M_1` and either of :math:`\beta` or :math:`\theta` supplied:

:math:`M_2`
  :py:meth:`~ObliqueShockRelation.calc_dmach`

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
