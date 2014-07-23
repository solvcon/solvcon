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

  \text{shock tube problem} = \text{Riemann problem} + \text{Eular equations}

where Riemann problem takes over the mathematics part and
Eular equations domain the physics part.

Riemann problem
---------------

The nonlinear hyperbolic system of PDEs :eq:`riemannproblem.pde`
and the picewise-defined function :eq:`riemannproblem.piecewise`
define the Riemann problem.

.. math::
  :label: riemannproblem.pde

  \dpd{\bvec{U}}{t}
  + \dpd{\bvec{F(\bvec{U})}}{x}
  = 0


.. math::
  :label: riemannproblem.piecewise

  \bvec{U} \defeq \left(\begin{array}{c}
    \rho_L \\ u_L
  \end{array}\right)
  \text{ for }
  x <= 0
  \text{ and }
  \bvec{U} \defeq \left(\begin{array}{c}
    \rho_R \\ u_R
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

============
Bibliography
============

.. [Sod78] Sod, G. A., "A Survey of Several Finite Difference Methods for
  Systems of Nonlinear Hyperbolic Conservation Laws", *J. Comput. Phys.*,
  27: 1–31.

.. vim: set spell ft=rst ff=unix fenc=utf8:
