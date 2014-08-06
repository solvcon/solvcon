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

To derive the analytic solution, we will begin from the region
:math:`\mathrm{IV}` to get
:math:`\bvec{u}_{\mathrm{IV}}`,
then :math:`\bvec{u}_{\mathrm{III}}` and
finally :math:`\bvec{u}_{\mathrm{II}}`.


Derivation of :math:`\bvec{u}_{\mathrm{IV}}`
--------------------------------------------

Rankine-Hugoniot conditions gives that the jump conditions must hold
across a shock:

.. math::
  :label: u4u5.rh.1

  u_{shock}(\rho_{2} - \rho_{1}) = m_2 - m_1

.. math::
  :label: u4u5.rh.2

  u_{shock}(m_2 - m_1)
  = \frac{{m_2}^2}{\rho_2} + p_2 - \frac{{m_1}^2}{\rho_1} - p_1

.. math::
  :label: u4u5.rh.3

  u_{shock}(\rho_{2} E_2 - \rho_{1} E_1) = m_{2} H_{2} - m_{1} H_{1}

If this is a stationary shock, :math:`u_{shock} = 0`.
:eq:`u4u5.rh.1` tells us :math:`m_2 = m_1`.

Because :math:`u_{shock} = 0` and :eq:`u4u5.rh.1`,
from :eq:`u4u5.rh.2` we get:

.. math::

  \frac{{m_2}^2}{\rho_2} + p_2 - \frac{{m_1}^2}{\rho_1} - p_1 = 0 \\
  \text{divided by } m_1
  \Rightarrow
  \frac{{m_2}^2}{\rho_{2}m_1} + \frac{p_2}{m_1} -
  \frac{{m_1}^2}{\rho_1{m_1}} - \frac{p_1}{m_1} = 0 \\
  \text{please remember } m_1 = m_2
  \Rightarrow
  \frac{m_{2}}{\rho_2} + \frac{p_2}{m_2} -
  \frac{m_{1}}{\rho_1} - \frac{p_1}{m_1}=0 \\
  \text{use } m_1 = \rho_{1}{u_1}, m_2 = \rho_{2}{u_2} 
  \Rightarrow
  u_2 + \frac{{\gamma}{p_2}}{\gamma{\rho_{2}{u_2}}} -
  u_1 - \frac{{\gamma}{p_1}}{\gamma{\rho_{1}{u_1}}} \\
  \text{because }
  c_1 = \sqrt{\frac{{\gamma}{p_1}}{\rho_1}},
  c_2 = \sqrt{\frac{{\gamma}{p_2}}{\rho_2}}
  \Rightarrow
  u_2 + \frac{{c_2}^2}{u_2} - u_1 - \frac{{c_1}^2}{u_1} = 0

Thus, we get

.. math::
  :label: u4u5.rh.2.1 

  u_1 - u_2 = \frac{{c_2}^2}{u_2} - \frac{{c_1}^2}{u_1}

Since :math:`u_{shock} = 0`, :math:`m_2 = m_1`
and :eq:`u4u5.rh.3`, we get :math:`H_1 = H_2`.
Use :math:`H=h+\frac{{c}^2}{2}`, namely
:math:`H_1=h_1+\frac{{c_1}^2}{2}` and :math:`H_2=h_2+\frac{{c_2}^2}{2}`,
and we could rewrite :math:`H_1 = H_2` as

.. math::

  & H_1 = h_1+\frac{{u_1}^2}{2} = h_2+\frac{{u_2}^2}{2} = H_2 \\
  & \text{Use } h = c_{p}T = \frac{c^2}{\gamma - 1} \\
  & \text{that is }
  \quad h_1 = c_{p}T_1 = \frac{{c_1}^2}{\gamma - 1} \quad
  h_2 = c_{p}T_2 = \frac{{c_2}^2}{\gamma - 1} \\
  \Rightarrow
  \quad & h_1 + \frac{{u_1}^2}{2}
  =  \frac{{c_1}^2}{\gamma - 1} + \frac{{u_1}^2}{2} \\
  \quad & h_2 + \frac{{u_2}^2}{2}
  =  \frac{{c_2}^2}{\gamma - 1} + \frac{{u_2}^2}{2} \\
  \Rightarrow
  \quad & \frac{{c_1}^2}{\gamma - 1} + \frac{{u_1}^2}{2}
  =  \frac{{c_2}^2}{\gamma - 1} + \frac{{u_2}^2}{2}

Assume :math:`u_1 > \text{sonic speed} c_{*} > u_2`. Because of continuity,
there must be a point with the speed
:math:`u_{*}` equal to the sound speed :math:`c_{*}` which satisfies:

.. math::

  \frac{{c_1}^2}{\gamma - 1} + \frac{{u_{*}}^2}{2} =
  \frac{{c_1}^2}{\gamma - 1} + \frac{{c_{*}}^2}{2} =
  \frac{(\gamma-1)+2}{2(\gamma-1)}{c_{*}} = 
  \frac{(\gamma+1)}{2(\gamma-1)}{c_{*}}

And

.. math::
  :label: u4u5.rh.3.1

  \frac{{c_1}^2}{\gamma - 1} + \frac{{u_1}^2}{2} =
  \frac{{c_2}^2}{\gamma - 1} + \frac{{u_2}^2}{2} = 
  \frac{(\gamma+1)}{2(\gamma-1)}{c_{*}}

Now let's try to get :math:`c_{*}`
represented by :math:`u_{1}` and :math:`u_{2}`.
Because of :eq:`u4u5.rh.3.1`

.. math::

  & \frac{{c_1}^2}{\gamma - 1} + \frac{{u_1}^2}{2}
  = \frac{(\gamma+1)c_{*}}{2(\gamma-1)} \\
  & \frac{{c_2}^2}{\gamma - 1} + \frac{{u_2}^2}{2}
  = \frac{(\gamma+1)c_{*}}{2(\gamma-1)} \\
  \text{multipled by } \frac{(2\gamma-1)}{\gamma{u_1}}
  & \text{ and multipled by } \frac{(2\gamma-1)}{\gamma{u_2}}
  \text{ seperately} \\
  \Rightarrow
  & \frac{2{c_1}^2}{\gamma{u_1}} + \frac{{u_1}(\gamma-1)}{\gamma}
  = \frac{(\gamma+1)c_{*}}{\gamma{u_1}} \\
  & \frac{2{c_2}^2}{\gamma{u_2}} + \frac{{u_2}(\gamma-1)}{\gamma}
  = \frac{(\gamma+1)c_{*}}{\gamma{u_2}} \\
  \Rightarrow
  & \frac{2{c_1}^2}{\gamma{u_1}} =
  \frac{(\gamma+1)c_{*}}{\gamma{u_1}} - \frac{{u_1}(\gamma-1)}{\gamma} \\
  & \frac{2{c_2}^2}{\gamma{u_2}} =
  \frac{(\gamma+1)c_{*}}{\gamma{u_2}} - \frac{{u_2}(\gamma-1)}{\gamma} \\
  \Rightarrow
  & \frac{{c_1}^2}{\gamma{u_1}}
  = \frac{(\gamma+1)c_{*}}{2\gamma{u_1}} - \frac{{u_1}(\gamma-1)}{2\gamma}
  = [\frac{(\gamma+1)c_{*}}{\gamma-1}+{u_1}^2]
  (\frac{(\gamma-1)}{2\gamma{u_1})}) \\
  & \frac{{c_2}^2}{\gamma{u_2}}
  = \frac{(\gamma+1)c_{*}}{2\gamma{u_2}} - \frac{{u_2}(\gamma-1)}{2\gamma}
  = [\frac{(\gamma+1)c_{*}}{\gamma-1}+{u_2}^2]
  (\frac{(\gamma-1)}{2\gamma{u_2})}) \\
  \Rightarrow
  & \frac{{c_1}^2}{\gamma{u_1}} - \frac{{c_2}^2}{\gamma{u_2}}
  = \frac{(\gamma+1)c_{*}}{2\gamma{u_1}} - \frac{{u_1}(\gamma-1)}{2\gamma}
  - \frac{(\gamma+1)c_{*}}{2\gamma{u_2}} + \frac{{u_2}(\gamma-1)}{2\gamma}

please recall :eq:`u4u5.rh.2.1`, thus

.. math::

  u_2 - u_1
  & = \frac{(\gamma+1)c_{*}}{2\gamma{u_1}}
  - \frac{{u_1}(\gamma-1)}{2\gamma}
  - \frac{(\gamma+1)c_{*}}{2\gamma{u_2}}
  + \frac{{u_2}(\gamma-1)}{2\gamma} \\
  \Rightarrow
  & c_{*}(\frac{(\gamma+1)}{2\gamma{u_1}}
  - \frac{(\gamma+1)}{2\gamma{u_2}})
  = u_2\frac{\gamma+1}{2\gamma}
  + u_1\frac{\gamma+1}{2\gamma} \\
  \Rightarrow
  & c_{*}(\frac{1}{u_1}-\frac{1}{u_2}) = u_2 - u_1 \\
  \Rightarrow
  & c_{*} = {u_1}{u_2}

The relation

.. math::
  :label: prandtl.meyer.relation

  c_{*} = {u_1}{u_2}

is called Prandel-Meyer relation.
It means the flow at one side of a shock must be supersonic,
and the other side must be subsonic.

============
Bibliography
============

.. [Sod78] Sod, G. A., "A Survey of Several Finite Difference Methods for
  Systems of Nonlinear Hyperbolic Conservation Laws", *J. Comput. Phys.*,
  27: 1–31.
.. [Wesselling01] Pieter Wesseling,
  "Principles of Computational Fluid Dynamics"

.. vim: set spell ft=rst ff=unix fenc=utf8:
