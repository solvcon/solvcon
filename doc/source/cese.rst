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

.. math::
  :label: riemannproblem

  \bvec{w} \defeq \left(\begin{array}{c}
    \rho \\ u
  \end{array}\right)
