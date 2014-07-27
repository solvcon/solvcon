===================================
The CESE Method (Under Development)
===================================

The space-time `Conservation Element and Solution Element (CESE) method
<http://www.grc.nasa.gov/WWW/microbus/>`__ is a numerical method designed for
solving linear and nonlinear first-order hyperbolic partial differential
equations (PDEs).  The method was originally developed for solving aerodynamic
problems [Chang95]_.

Reliability
===========

A classic example to verify whether an CFD algorithm is well-developed and
robust is the Sod shock tube problem.
The Sod shock tube problem is named after Gary A. Sod who proposed and
investigate the problem heavily in 1978 [Sod78]_.
In the following we are going to introduce the Sod tube problem in detail.

Analytic solution
+++++++++++++++++

In short,

.. math::

  \text{Sod shocktube problem} = \text{shock tube problem} + \text{Sod's initial condition}

A shock tube problem is a well-defined problem and has an analytic solution.
This is a good benchmark to compare different CFD algorithm results.

Form the point of physics and mathematics view, we will say

.. math::

  \text{shock tube problem} = \text{Riemann problem} + \text{Eular equations in gas dynamic}

where Riemann problem takes over the mathematics part and
Eular equations domain the physics part.

Riemann problem
---------------

The nonlinear hyperbolic system of PDEs :eq:`riemannproblem.pde`
and the piecewise-defined function :eq:`riemannproblem.piecewise`
define the Riemann problem.

.. math::
  :label: riemannproblem.pde

  \dpd{\bvec{U}}{t}
  + \dpd{\bvec{F(\bvec{U})}}{x}
  = 0


.. math::
  :label: riemannproblem.piecewise

  \bvec{U} \defeq \left(\begin{array}{c}
    \rho_L \\ u_L \\ p_L
  \end{array}\right)
  \text{ for }
  x <= 0
  \text{ and }
  \bvec{U} \defeq \left(\begin{array}{c}
    \rho_R \\ u_R \\ p_R
  \end{array}\right)
  \text{ for }
  x > 0

Eular equations in gas dynamic
------------------------------

Eular equations are one of the hyperbolic systems of PDEs
:eq:`riemannproblem.pde`. They represent mass conservation
:eq:`eular.gasdyn.mass`, momentum conservation :eq:`eular.gasdyn.momentum`,
and energy conservation :eq:`eular.gasdyn.energy`.

.. math::
  :label: eular.gasdyn.mass

  \dpd{\rho}{t} + \dpd{{\rho}{v}}{x} = 0

.. math::
  :label: eular.gasdyn.momentum

  \dpd{\rho{v}}{t} + \dpd{(p+\rho{v^2})}{x} = 0

.. math::
  :label: eular.gasdyn.energy

  \dpd{(\frac{p}{\gamma-1} + \frac{\rho{v^2}}{2})}{t}
  + \dpd{(\frac{\gamma}{\gamma-1}pv+\frac{1}{2}\rho{v^3})}{x}
  = 0

If

.. math::
  :label: eular.gasdyn.u

  \bvec{U}
  =
  \left(\begin{array}{c}
    u_1 \\ u_2 \\ u_3
  \end{array}\right)
  \defeq
  \left(\begin{array}{c}
    \rho_1 \\ \rho_2 \\ \rho_3
  \end{array}\right)

.. math::
  :label: eular.gasdyn.f

  \bvec{F}
  =
  \left(\begin{array}{c}
    f_1 \\ f_2 \\ f_3
  \end{array}\right)
  \defeq
  \left(\begin{array}{c}
    {\rho}{v} \\ {(p+\rho{v^2})} \\ {(\frac{\gamma}{\gamma-1}pv+\frac{1}{2}\rho{v^3})}
  \end{array}\right)

Equation :eq:`eular.gasdyn.mass`, :eq:`eular.gasdyn.momentum` and
:eq:`eular.gasdyn.energy` could be written as `riemannproblem.pde`. 

1D Sod's shock tube problem
---------------------------

In :eq:`riemannproblem.piecewise`, if we introduce Sod's conditions in
the one-dimension(1D) shock tube problem.

.. math::
  :label: sod.conditions

  \bvec{U} 
  \defeq
  \left(\begin{array}{c}
    1 \\ 0 \\ 1
  \end{array}\right)
  \defeq
  \bvec{U_L}
  \text{ for }
  x <= 0
  \text{ and }
  \bvec{U}
  \defeq
  \left(\begin{array}{c}
    0.125 \\ 0 \\ 0.1
  \end{array}\right)
  \defeq
  \bvec{U_R}
  \text{ for }
  x > 0
  \text{at } t=0

and :math:`\bvec{U}` and :math:`\bvec{F}` obey Eular equations,
this is called Sod's shock tube problem. The physical image could be
there is a diaphragm, which ideal gas with the status :math:`\bvec{U_L}`
in the left-hand side of the diaphragm, ideal gas with the status
:math:`\bvec{U_R}` in the right-hand side. How does the status evolve
after the diaphragm disappears all of a sudden, say at :math:`t>0`

We describe the Sod shock tube at :math:`t>0` in "5 zones".
From the left (:math:`x<0`) to the right (:math:`x>0`) of the diaphragm.

* Region 1

  * There is no boundary of the tube,so the status is always :math:`\bvec{U_L}`

* Region 2

  * The status is linear combination of the sound in the region 2 and
    the rarefaction wave. And the status is continuous from the region 1
    to the region 3. For example, the velocity in the region 3, :math:`u_3`,
    continues to decrease to be the velocity in the region 1,
    :math:`u_1=0`.

* Region 3
  
  * In the shock "pocket", there is "no more shock" and the hyperbolic
    PDE :eq:`riemannproblem.pde` told us :math:`u_3=u_4=\text{Reimann-invariants}`.
    Together with Rankine-Hugoniot conditions, we know :math:`p_3=p_4` and
    the density is discontinuous.

* Region 4

  * Because of the expansion of the shock, there is shock discontinuity.
    The discontinuity status could be determined by Rankine-Hugoniot conditions
    [Wesselling01]_.

* Region 5

  * There is no boundary of the tube,so the status is always :math:`\bvec{U_R}`


============
Bibliography
============

.. [Sod78] Sod, G. A., "A Survey of Several Finite Difference Methods for
  Systems of Nonlinear Hyperbolic Conservation Laws", *J. Comput. Phys.*,
  27: 1–31.
.. [Wesselling01] Pieter Wesseling, "Principles of Computational Fluid Dynamics"

.. vim: set spell ft=rst ff=unix fenc=utf8:
