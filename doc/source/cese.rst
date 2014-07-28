===================================
The CESE Method (Under Development)
===================================

The space-time `Conservation Element and Solution Element (CESE) method
<http://www.grc.nasa.gov/WWW/microbus/>`__ is a numerical method designed for
solving linear and nonlinear first-order hyperbolic partial differential
equations (PDEs).  The method was originally developed for solving aerodynamic
problems [Chang95]_.

Verification by the Euler Equations
===================================

A classic example to verify whether a CFD algorithm the Sod shock tube problem
[Sod78]_.  We will introduce this problem in what follows.

Sod's Shock Tube Problem
++++++++++++++++++++++++

In short, a shock tube problem is a Riemann problem with the Euler equations.
This is a good benchmark to compare different CFD algorithm results.

The Euler equations consist of conservation of mass (Eq.
:eq:`eular.gasdyn.mass`), of momentum (Eq. :eq:`eular.gasdyn.momentum`), and of
energy (Eq. :eq:`eular.gasdyn.energy`).

.. math::
  :label: eular.gasdyn.mass

  \dpd{\rho}{t} + \dpd{{\rho}{v}}{x} = 0

.. math::
  :label: eular.gasdyn.momentum

  \dpd{\rho{v}}{t} + \dpd{(p+\rho{v^2})}{x} = 0

.. math::
  :label: eular.gasdyn.energy

  \dpd{}{t}\left(\frac{p}{\gamma-1} + \frac{\rho{v^2}}{2}\right)
  + \dpd{}{x}\left(\frac{\gamma}{\gamma-1}pv+\frac{1}{2}\rho{v^3}\right)
  = 0

By defining

.. math::
  :label: eular.gasdyn.u

  \bvec{u}
  =
  \left(\begin{array}{c}
    u_1 \\ u_2 \\ u_3
  \end{array}\right)
  \defeq
  \left(\begin{array}{c}
    \rho \\ \rho v \\
    \rho\left(\frac{1}{\gamma-1}\frac{p}{\rho} + \frac{v^2}{2}\right)
  \end{array}\right)

.. math::
  :label: eular.gasdyn.f

  \bvec{f}
  =
  \left(\begin{array}{c}
    f_1 \\ f_2 \\ f_3
  \end{array}\right)
  \defeq
  \left(\begin{array}{c}
    u_2 \\ (\gamma-1)u_3 - \frac{\gamma-3}{2}\frac{u_2^2}{u_1} \\
    \gamma\frac{u_2u_3}{u_1} - \frac{\gamma-1}{2}\frac{u_2^3}{u_1^2}
  \end{array}\right)

we can rewrite Eqs. :eq:`eular.gasdyn.mass`, :eq:`eular.gasdyn.momentum`, and
:eq:`eular.gasdyn.energy` in a general form for nonlinear hyperbolic PDEs:

.. math::
  :label: riemannproblem.pde

  \dpd{\bvec{u}}{t} + \dpd{\bvec{f}(\bvec{u})}{x} = 0

The initial condition of the Riemann problem is defined as:

.. math::
  :label: riemannproblem.piecewise

  \bvec{u} = \left(\begin{array}{c}
    \rho_L \\ u_L \\ p_L
  \end{array}\right)
  \text{ for }
  x <= 0
  \text{ and }
  \bvec{u} = \left(\begin{array}{c}
    \rho_R \\ u_R \\ p_R
  \end{array}\right)
  \text{ for }
  x > 0

By using Eq. :eq:`riemannproblem.piecewise`, Sod's initial conditions can be
set as:

.. math::
  :label: sod.conditions

  \bvec{u} 
  =
  \left(\begin{array}{c}
    1 \\ 0 \\ 1
  \end{array}\right)
  \defeq \bvec{u}_L
  \text{ for }
  x <= 0
  \text{ and }
  \bvec{u}
  =
  \left(\begin{array}{c}
    0.125 \\ 0 \\ 0.1
  \end{array}\right)
  \defeq \bvec{u}_R
  \text{ for }
  x > 0
  \text{at } t=0

We divide the solution of the problem in "5 zones".  From the left
(:math:`x<0`) to the right (:math:`x>0`) of the diaphragm.

- Region I

  - There is no boundary of the tube.  The status is always :math:`\bvec{u}_L`.

- Region II

  - Rarefaction wave.  The status is continuous from the region 1 to the region
    3.
  
- Region III
  
  - In the shock "pocket", there is "no more shock" and the hyperbolic PDE
    :eq:`riemannproblem.pde` told us :math:`u_{\mathrm{III}}=u_{\mathrm{IV}}`
    are Riemann invariants.  Together with Rankine-Hugoniot conditions, we know
    :math:`p_{\mathrm{III}}=p_{\mathrm{IV}}` and the density is not continuous.

- Region IV

  - Because of the expansion of the shock, there is shock discontinuity.
    The discontinuity status could be determined by Rankine-Hugoniot conditions
    [Wesselling01]_.

- Region V

  - There is no boundary of the tube, so the status is always
    :math:`\bvec{u}_R`

To derive the analytic solution, we will begin from the region 4 to get
:math:`\bvec{u}_{\mathrm{IV}}`, then :math:`\bvec{u}_{\mathrm{III}}` and
finally :math:`\bvec{u}_{\mathrm{II}}`.

============
Bibliography
============

.. [Sod78] Sod, G. A., "A Survey of Several Finite Difference Methods for
  Systems of Nonlinear Hyperbolic Conservation Laws", *J. Comput. Phys.*,
  27: 1–31.
.. [Wesselling01] Pieter Wesseling, "Principles of Computational Fluid Dynamics"

.. vim: set spell ft=rst ff=unix fenc=utf8:
