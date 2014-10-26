====================================================
Reflection of Oblique Shock Wave (Under Development)
====================================================

.. py:module:: solvcon.parcel.gas.oblique_shock

Consider an oblique shock wave hits a solid wall and reflects.  The system will
have two oblique shock waves and the relations of oblique shock will be applied
twice to obtain the flow properties in the totally 3 zones.

The given data include:

1. The upstream Mach number :math:`M_1`.
2. Either the first oblique shock angle :math:`\beta`, or the flow deflection
   angle :math:`\theta`.

Either of the angle :math:`\theta` or :math:`\beta` can be calculated from
:py:meth:`~ObliqueShockRelation.calc_flow_angle` or
:py:meth:`~ObliqueShockRelation.calc_shock_angle`, respectively, if :math:`M_1` and the
other angle is known.

Therefore :math:`\theta` and the Mach number in the second zone :math:`M_2` are
determined by applying the oblique shock relations for the first oblique shock.
Because in the third zone, the flow will be in parallel to the wall behind the
reflected oblique shock, the :math:`\beta`\ -:math:`\theta`\ -:math:`M`
relation is used to determine the second oblique shock angle :math:`\beta'`.

Relations across Oblique Shock
==============================

Methods of calculating the shock relations are organized in the class
:py:class:`ObliqueShockRelation`.

.. autoclass:: ObliqueShockRelation

  .. autoinstanceattribute:: gamma

Define the angle of the oblique shock wave deflected from the upstream is
:math:`\beta`, and the angle of the flow behind the shock wave deflected from
the upstream is :math:`\theta`.  The oblique shock wave defines a rotated
coordinate system :math:`(n, t)`, where :math:`\hat{n}` is the unit vector
normal to the oblique shock, and :math:`\hat{t}` is the unit vector tangential
to the shock.  Consequently, because :math:`M = v/a`, where :math:`a` is the
speed of sound, the Mach number corresponding to the normal components of the
velocity across the oblique shock can be written as:

.. math::

  M_{n1} = M_1\sin\beta, \quad M_{n2} = M_2\sin(\beta-\theta)

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
