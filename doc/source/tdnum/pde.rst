========================================
What Are Partial Differential Equations?
========================================

A partial differential equation (PDE) is an equation that contains partial
derivatives of the unknown :math:`u`.  For example,

.. math::

  a_1\dpd[2]{u}{x} + a_2\dmd{u}{2}{x}{}{t}{} + a_3\dpd[2]{u}{t}
  + a_4\dpd{u}{x} + a_5\dpd{u}{t} + a_6u + a_0 = 0

where :math:`u = u(x, t)`.  For convenience, sometimes subscripts
(:math:`\square_x, \square_t`) are used to denote partial derivatives.  With
the convention we can simplify the notation of the above PDE:

.. math::
  :label: tdnum.pde.sub

  a_1u_{xx} + a_2u_{xt} + a_3u_{tt} + a_4u_x + a_5u_t + a_6u + a_0 = 0

The *order* of a PDE is determined by its highest-order derivative.  For
example, Eq. :eq:`tdnum.pde.sub` is second order.  A PDE is *linear* if every
term behaves linearly in the equation.  The linearity can be defined after
temporarily excluding the constant term from the equation (:math:`a_0` in
:eq:`tdnum.pde.sub`):

1. If :math:`u_1` and :math:`u_2` are both a solution of the modified equation,
   then :math:`u_1 + u_2` is also a solution.
2. If :math:`u_1` is a solution of the modified equation, then :math:`cu_1` is
   also a solution where :math:`c` is a constant. 

Otherwise it's nonlinear.  (Another way to define the linearity is to use a
linear operator, which is not discussed here.)  If :eq:`tdnum.pde.sub` is
linear and :math:`a_0` is indeed zero, the equation is a *homogeneous* linear
PDE, and the principle of superposition applies.  For example, let
:math:`\hat{u}` and :math:`\tilde{u}` be solution of

.. math::

  a u_{xx} + b u_{xt} + c u_{tt} = 0

Thus :math:`a \hat{u}_{xx} + b \hat{u}_{xt} + c \hat{u}_{tt} = 0` and :math:`a
\tilde{u}_{xx} + b \tilde{u}_{xt} + c \tilde{u}_{tt} = 0` hold true.  We have

.. math::

  a (\hat{u}_{xx}+\tilde{u}_{xx}) + b (\hat{u}_{xt}+\tilde{u}_{xt})
  + c (\hat{u}_{tt}+\tilde{u}_{tt}) = 0

In general a PDE can have two or more independent variables.  For the sake of
conciseness, the discussion is focused on equations of two independent
variables.

.. _tdnum.moc:

Method of Characteristics
=========================

A linear PDE

.. math::
  :label: tdnum.linear2d

  A u_x + B u_y = F

where :math:`A = A(x,y)`, :math:`B = B(x,y)`, and :math:`u = u(x,y)` can be
solved by using the method of characteristics, which reduces the PDE of two
independent variables to an ordinary differential equation (ODE).  First
consider the homogeneous part of :eq:`tdnum.linear2d`

.. math::
  :label: tdnum.linearhomo2d

  A u_x + B u_y = 0

In the two-dimensional space :math:`(x, y)`, :eq:`tdnum.linearhomo2d` can be
written as

.. math::

  (A, B)\cdot(u_x, u_y) = 0 \quad\mbox{or}\quad (A, B)\cdot\nabla u = 0

It can be seen that along the direction :math:`(A, B)`, the solution of
:math:`u` doesn't change.  The vector :math:`(A, B)` can form a curve that is
called the *characteristic curve* of :eq:`tdnum.linear2d`, and its slope is

.. math::

  \dod{y}{x} = \frac{B}{A}

To solve the homogeneous equation :eq:`tdnum.linearhomo2d`, choose a coordinate
transformation

.. math::

  \left\{\begin{aligned}
    \xi &= \xi(x, y) \\
    \eta &= \eta(x, y)
  \end{aligned}\right.

Aided by the chain rule, we have

.. math::

  \left\{\begin{aligned}
    u_x &= u_{\xi}\xi_x + u_{\eta}\eta_x \\
    u_y &= u_{\xi}\xi_y + u_{\eta}\eta_y
  \end{aligned}\right.

Substitute the above into :eq:`tdnum.linearhomo2d` and get

.. math::

    (A\xi_x + B\xi_y) u_{\xi} + (A\eta_x + B\eta_y) u_{\eta} = 0

By requiring :math:`A\eta_x + B\eta_y = 0`, :math:`u_{\eta}` is eliminated from
the above equation.  Further, we choose to obtain the solution on the
characteristic curve on which :math:`\dif y/\dif x = B/A`, and then

.. math::

  \dod{\eta}{x} &= \eta_x + \eta_y\dod{y}{x} = \eta_x + \frac{B}{A}\eta_y
  = 0 \\
  \dod{\eta}{y} &= \eta_x\dod{x}{y} + \eta_y = \frac{A}{B}\eta_x + \eta_y
  = 0

(:math:`A` and :math:`B` are assumed to be non-zero.)  :math:`\xi` and
:math:`\eta` should be chosen so that the Jacobian determinant

.. math::

  J = \dpd{(\xi, \eta)}{(x, y)} = \left|\begin{array}{cc}
    \xi_x & \xi_y \\ \eta_x & \eta_y
  \end{array}\right| \neq 0

which means the coordinate transformation is non-degenerate.  Finally, the
original equation :eq:`tdnum.linear2d` is transformed to an ODE:

.. math::
  :label: tdnum.mocode

  (A\xi_x + B\xi_y) u_{\xi} = F

If :math:`A` and :math:`B` are constant, we have a straight *characteristic
line*.

.. admonition:: Alternative Way to the Characteristic Curve

  .. math::

    \dif u = u_x\dif x + u_y\dif y
    \Rightarrow u_x = \frac{\dif u - u_y \dif y}{\dif x}

  Substituting above into :eq:`tdnum.linear2d` gives

  .. math::

    & A \frac{\dif u - u_y\dif y}{\dif x} + B u_y = F \\
    \Rightarrow\quad & A (\dif u - u_y \dif y) + B u_y \dif x = F \dif x \\
    \Rightarrow\quad & A \dif u + (B \dif x - A \dif y)u_y = F \dif x

  If requiring
  
  .. math::
  
    B\dif x - A\dif y = 0 \quad\mbox{or}\quad \dod{y}{x} = \frac{B}{A}
  
  we have

  .. math::

    A \dif u = F \dif x \quad\mbox{or}\quad A \dod{u}{x} = F

  It is the same result as :eq:`tdnum.mocode` when we choose :math:`\xi = x`.

.. admonition:: Example
  :class: example

  Let :math:`A = a`, :math:`B = 1`, :math:`F = 0`

  .. math::

    u_y + a u_x = 0

  is a one-way, one-dimensional, scalar wave equation.  Let :math:`y = t` to
  make it in space-time:

  .. math::

    u_t + a u_x = 0

  Thus, along :math:`\frac{\dif t}{\dif x} = \frac{B}{A} = \frac{1}{a}`,
  :math:`a\dod{u}{x} = 0`, or simply :math:`\dod{u}{x} = 0`.  :math:`\dod{x}{t}
  = a` is the wave speed.

  Integrate along the characteristic line

  .. math::

    x = a t + x_0

  .. math::

    \dod{u}{t}(at+x_0, x) = \dpd{u}{x}\dod{x}{t} + \dpd{u}{t}\dod{t}{t}
      = u_x a + u_t = 0

  :math:`u` is constant along the characteristic line :math:`\dod{x}{t} = a`.

  The solution of :math:`u` in the space-time depends on the initial condition
  at :math:`x = x_0, t = 0`.

  .. math::

    u(x, t) = f(x_0) = f(x - at)

  where :math:`u(x, 0) = f(x_0)` is the initial condition.

  Across the characteristic lines the solution of :math:`u` can be
  discontinuous.  :math:`u_t` and :math:`u_x` can be undefined.

  .. TODO: add illustrative figures.

In the inviscid Burger's equation

.. math::

  u_t + u u_x = 0

the wave speed is :math:`u` itself.  The profile of :math:`u` would change in
the time-evolving solution of :math:`u`.

.. TODO: add illustrative figures.

.. _tdnum.o2pde:

Second-Order PDE
================

Consider a linear second-order PDE

.. math::
  :label: tdnum.pde2d2v

  A u_{xx} + B u_{xy} + C u_{yy} + D u_x + E u_y + F u + G = 0

where :math:`A, B, C, D, E, F, G` are functions of :math:`x` and :math:`y`.
Because the order of a PDE is determined by the leading order terms, we let
:math:`H = -(D u_x + E u_y+ F u + G)` and make :eq:`tdnum.pde2d2v` become

.. math::
  :label: tdnum.pde2d2v.lt

  A u_{xx} + B u_{xy} + C u_{yy} = H

We want to find the characteristic curve on which the equation
:eq:`tdnum.pde2d2v` can be simplified.  To do it, we define a non-degenerate
(Jacobian determinant :math:`J = \partial(\xi, \eta)/\partial(x, y) \neq 0`)
coordinate transformation (like what we did in :ref:`tdnum.moc`):

.. math::

  \left\{\begin{aligned}
    \xi &= \xi(x, y) \\
    \eta &= \eta(x, y)
  \end{aligned}\right.

Write and substitute

.. math::

  u_x &= u_{\xi}\xi_x + u_{\eta}\eta_x \\
  u_y &= u_{\xi}\xi_y + u_{\eta}\eta_y \\
  u_{xx} &= \dpd{(u_{\xi}\xi_x + u_{\eta}\eta_x)}{x} \\
  &= (u_{\xi\xi} \xi_x + u_{\xi\eta} \eta_x)\xi_x  + u_{\xi} \xi_{xx}
   + (u_{\eta\xi}\xi_x + u_{\eta\eta}\eta_x)\eta_x + u_{\eta}\eta_{xx} \\
  &= u_{\xi\xi}\xi_x^2 + 2u_{\xi\eta}\xi_x\eta_x + u_{\eta\eta}\eta_x^2
   + u_{\xi}\xi_{xx} + u_{\eta}\eta_{xx} \\
  u_{xy} &= \dpd{(u_{\xi}\xi_x + u_{\eta}\eta_x)}{y} \\
  &= (u_{\xi\xi} \xi_y + u_{\xi\eta} \eta_y)\xi_x  + u_{\xi} \xi_{xy}
   + (u_{\eta\xi}\xi_y + u_{\eta\eta}\eta_y)\eta_x + u_{\eta}\eta_{xy} \\
  &= u_{\xi\xi}\xi_x\xi_y + (\xi_x\eta_y + \xi_y\eta_x)u_{\xi\eta}
   + u_{\eta\eta}\eta_x\eta_y + u_{\xi}\xi_{xy} + u_{\eta}\eta_{xy} \\
  u_{yy} &= \dpd{(u_{\xi}\xi_y + u_{\eta}\eta_y)}{y} \\
  &= (u_{\xi\xi} \xi_y + u_{\xi\eta} \eta_y)\xi_y  + u_{\xi} \xi_{yy}
   + (u_{\eta\xi}\xi_y + u_{\eta\eta}\eta_y)\eta_y + u_{\eta}\eta_{yy} \\
  &= u_{\xi\xi}\xi_y^2 + 2u_{\xi\eta}\xi_y\eta_y + u_{\eta\eta}\eta_y^2
   + u_{\xi}\xi_{yy} + u_{\eta}\eta_{yy}

into :eq:`tdnum.pde2d2v.lt` and obtain

.. math::

  & A(u_{\xi\xi}\xi_x^2 + 2u_{\xi\eta}\xi_x\eta_x + u_{\eta\eta}\eta_x^2
   + u_{\xi}\xi_{xx} + u_{\eta}\eta_{xx}) + \\
  & B[u_{\xi\xi}\xi_x\xi_y + (\xi_x\eta_y + \xi_y\eta_x)u_{\xi\eta}
   + u_{\eta\eta}\eta_x\eta_y + u_{\xi}\xi_{xy} + u_{\eta}\eta_{xy}] + \\
  & C(u_{\xi\xi}\xi_y^2 + 2u_{\xi\eta}\xi_y\eta_y + u_{\eta\eta}\eta_y^2
   + u_{\xi}\xi_{yy} + u_{\eta}\eta_{yy})
  = H \\
  \Rightarrow\quad &
  \bar{A}u_{\xi\xi} + \bar{B}u_{\xi\eta} + \bar{C}u_{\eta\eta} = \bar{H}

where

.. math::

  \bar{A} &\defeq A\xi_x^2 + B\xi_x\xi_y + C\xi_y^2 \\
  \bar{B} &\defeq
    2A\xi_x\eta_x + B\xi_x\eta_y + B\xi_y\eta_x + 2C\xi_y\eta_y \\
  \bar{C} &\defeq A\eta_x^2 + B\eta_x\eta_y + C\eta_y^2

It can be shown by straight-forward algebra that

.. math::
  :label: tdnum.discriminant

  \bar{B}^2 - 4\bar{A}\bar{C} = (\xi_x\eta_y - \xi_y\eta_x)^2 (B^2 - 4AC)
  = J^2 (B^2 - 4AC)

Thus :math:`B^2 - 4AC` and :math:`\bar{B}^2 - 4\bar{A}\bar{C}` have the same
sign, because we require a non-degenerate coordinate transform (:math:`J =
\xi_x\eta_y - \xi_y\eta_x \ne 0`).  The expression :math:`B^2 - 4AC` or
:math:`\bar{B}^2 - 4\bar{A}\bar{C}` is called the *discriminant* for the
equation :eq:`tdnum.pde2d2v`.  Based on the discriminant, the second-order PDE
can be categorized into three classes:

- When :math:`B^2 - 4AC > 0`, equation :eq:`tdnum.pde2d2v` is *hyperbolic*.
- When :math:`B^2 - 4AC = 0`, equation :eq:`tdnum.pde2d2v` is *parabolic*.
- When :math:`B^2 - 4AC < 0`, equation :eq:`tdnum.pde2d2v` is *elliptic*.

.. admonition:: Aside

  Recall the quadratic equation (second-order polymonial with two variables)

  .. math::

    a x^2 + b xy + c y^2 + d x + e y + f = 0

  - :math:`b^2-4ac > 0` means the equation is hyperbolic; :math:`xy = k,
    \frac{x^2}{a^2} - \frac{y^2}{b^2} = k`.
  - :math:`b^2-4ac = 0` means the equation is parabolic; :math:`y^2 = 4p x`.
  - :math:`b^2-4ac < 0` means the equation is elliptic; :math:`\frac{x^2}{a^2}
    + \frac{y^2}{b^2} = k`

An important fact is that, because of :eq:`tdnum.discriminant`, after an
arbitrary coordinate transformation, the class of the PDE does not change!

To simplify :eq:`tdnum.pde2d2v`, we choose the coordinate transformation to
make :math:`\bar{A} = \bar{B} = 0`.  In this way, we write

.. math::
  :label: tdnum.2ocheqn

  A \zeta_x^2 + B \zeta_x\zeta_y + C \zeta_y^2 = 0

where :math:`\zeta` is used to denote :math:`\xi` or :math:`\eta`.  Equation
:eq:`tdnum.2ocheqn` will be used to find the characteristic curves, on which
:math:`\zeta(x, y) = \mbox{constant}`, and we can write

.. math::

  \dif \zeta = \zeta_x \dif x + \zeta_y \dif y = 0

From the above equation the slope of the characteristic curves can be found to be

.. math::

  \dod{y}{x} = -\frac{\zeta_x}{\zeta_y}

By using the above equation and dividing the both sides of :eq:`tdnum.2ocheqn`
by :math:`\zeta_y`, we have

.. math::

  A \left(\dod{y}{x}\right)^2 - B \left(\dod{y}{x}\right) + C = 0

Then the slope of the characteristic curves will be obtained by solving the
above equation, and it is

.. math::

  \dod{y}{x} = \frac{B \pm \sqrt{B^2 - 4AC}}{2A}

Canonical Form
==============

By following the analysis in :ref:`tdnum.o2pde`, we can further reduce the
second-order PDE with two independent variables to their representative
canonical form:

- Hyperbolic PDE:

  .. math::

    u_{\xi\eta} &= \tilde{H}(\xi, \eta, u, u_{\xi}, u_{\eta}) \\
    u_{\xi\xi} - u_{\eta\eta} &= \tilde{H}^*(\xi, \eta, u, u_{\xi}, u_{\eta})

- Parabolic PDE:

  .. math::

    u_{\xi\xi} = \tilde{H}(\xi, \eta, u, u_{\xi}, u_{\eta})

- Elliptic PDE:

  .. math::

    u_{\xi\xi} + u_{\eta\eta} = \tilde{H}(\xi, \eta, u, u_{\xi}, u_{\eta})

Hyperbolic PDEs
===============

For hyperbolic PDEs, we have

.. math::

  & \dod{y}{x} = \frac{B \pm \sqrt{B^2 - 4AC}}{2A} \\
  \Rightarrow\quad & \frac{B \pm \sqrt{B^2 - 4AC}}{2A}x - y = c

where :math:`c = \mbox{constant}`.  Then we can write the coordinate
transformation as

.. math::

  \arraycolsep=1.4pt\def\arraystretch{1.5}
  \left\{\begin{array}{rcl}
    \xi & = & \frac{B + \sqrt{B^2 - 4AC}}{2A}x - y \\
    \eta & = & \frac{B - \sqrt{B^2 - 4AC}}{2A}x - y \\
  \end{array}\right.

Because of :eq:`tdnum.2ocheqn`, :math:`\bar{A} = \bar{B} = 0`.  Equation
:eq:`tdnum.pde2d2v` is then reduced to a canonical form of the hyperbolic PDE:

.. math::

  \bar{B} u_{\xi\eta} = \bar{H} \;\mbox{or}\;
  u_{\xi\eta} = \tilde{H}(\xi, \eta, u, u_{\xi}, u_{\eta})

.. vim: set spell ft=rst ff=unix fenc=utf8:
