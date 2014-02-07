========================================
What Are Partial Differential Equations?
========================================

A partial differential equation (PDE) is an equation that contains partial
derivatives of the unknown :math:`u`.  For example,

.. math::
  :label: tdnum.pde.dpd

  a_1\dpd[2]{u}{x} + a_2\dmd{u}{2}{x}{}{t}{} + a_3\dpd[2]{u}{t}
  + a_4\dpd{u}{x} + a_5\dpd{u}{t} + a_6u + a_0 = 0

where :math:`u = u(x, t)`.  For convenience, sometimes subscripts are used to
denote partial derivatives, like :math:`\square_x, \square_t`.  With the
convention we can simplify the notation of the above PDE:

.. math::
  :label: tdnum.pde.sub

  a_1u_{xx} + a_2u_{xt} + a_3u_{tt} + a_4u_x + a_5u_t + a_6u + a_0 = 0

The *order* of a PDE is determined by its highest-order derivative.  For
example, Eq. :eq:`tdnum.pde.sub` is second order.  A PDE is *linear* if every
term behaves linearly in the equation.  The linearity can be defined after
temporarily excluding the constant term (:math:`a_0` in Eq.
:eq:`tdnum.pde.sub`) from the equation:

1. If :math:`u_1` and :math:`u_2` are both a solution of the modified equation,
   then :math:`u_1 + u_2` is also a solution.
2. If :math:`u_1` is a solution of the modified equation, then :math:`cu_1` is
   also a solution where :math:`c` is a constant. 

Otherwise it's nonlinear.  (Another way to define the linearity is to use
linear operator, which is not discussed here.)  If Eq. :eq:`tdnum.pde.sub` is
linear and :math:`a_0` is indeed zero, the equation is a *homogeneous* linear
PDE.  For a homogeneous linear PDE, the principle of superposition applies.
For example, let :math:`\hat{u}` and :math:`\tilde{u}` be solution of

.. math::

  a u_{xx} + b u_{xt} + c u_{tt} = 0

Thus :math:`a \hat{u}_{xx} + b \hat{u}_{xt} + c \hat{u}_{tt} = 0` and :math:`a
\tilde{u}_{xx} + b \tilde{u}_{xt} + c \tilde{u}_{tt} = 0` hold true.  We have

.. math::

  a (\hat{u}_{xx}+\tilde{u}_{xx}) + b (\hat{u}_{xt}+\tilde{u}_{xt})
  + c (\hat{u}_{tt}+\tilde{u}_{tt}) = 0

In general a PDE can have two or more independent variables, but for the sake
of conciseness, this chapter focuses on equations of two independent variables.
A differential equation of only one independent variable is ordinary and cannot
be a PDE.

Method of Characteristics for First-Order PDE
=============================================

In the following linear PDE, :math:`A = A(x,y)`, :math:`B = B(x,y)`,
:math:`F = F(x,y)`, and :math:`u = u(x,y)`:

.. math::
  :label: tdnum.linear2d

  A u_x + B u_y = F

.. math::

  & \dif u = u_x\dif x + u_y\dif y \\
  \Rightarrow\quad & u_x = \frac{\dif u - u_y \dif y}{\dif x}

Substituting above into Eq. :eq:`tdnum.linear2d` gives

.. math::

  & A \frac{\dif u - u_y\dif y}{\dif x} + B u_y = F \\
  \Rightarrow\quad & A (\dif u - u_y \dif y) + B u_y \dif x = F \dif x \\
  \Rightarrow\quad & A \dif u + (B \dif x - A \dif y)u_y = F \dif x

If :math:`B\dif x - A\dif y = 0` (or :math:`\frac{\dif y}{\dif x} =
\frac{B}{A}`), we have

.. math::

  A \dif u = F \dif x \,\mbox{, or}\, A \frac{\dif u}{\dif x} = F

Along the characteristic line :math:`\frac{\dif y}{\dif x} = \frac{B}{A}`,
:math:`A\frac{\dif u}{\dif x} = F`, and a PDE in :math:`(x, y)` domain then
becomes an ordinary differential equation (ODE).

If :math:`A` and :math:`B` are constant, the characteristic line is a straight
line, while in general it is a characteristic curve.

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

Second-Order PDE
================

.. math::
  :label: tdnum.pde2d2v

  A u_{xx} + B u_{xy} + C u_{yy} + D u_x + E u_y + F u + G = 0

For linear PDEs, :math:`A, B, C, D, E, F, G` could be functions of :math:`x`
and :math:`y`.

For nonlinear PDEs, :math:`A, B, C, D, E, F, G` are functions of :math:`u` and
its derivatives, e.g.,

.. math::

  A = A(u, u_x, u_y, u_{xx}, \ldots)

The order of a PDE is determined by the highest order of the derivatives in the
equation.  Thus Eq. :eq:`tdnum.pde2d2v` is a second-order PDE.

We let :math:`H = -(D u_x + E u_y+ F u + G)` and Eq. :eq:`tdnum.pde2d2v`
becomes

.. math::

  A u_{xx} + B u_{xy} + C u_{yy} = H

If the characteristic curves exit on the :math:`(x, y)` plane, the second-order
derivatives of :math:`u`, i.e., :math:`u_{xx}`, :math:`u_{xy}`, and
:math:`u_{yy}` are undefined across the characteristic curves (similar to the
idea of the one-dimensional PDE).

Along a characteristic curve, we let :math:`\tau` be the independent variable
which varies along the curve.  Along the characteristic curves, :math:`u =
u(\tau)`, :math:`u_x = u_x(\tau)`, and :math:`u_y = u_y(\tau)` are continuous.  

.. math::

  & \dod{u_x}{\tau} = u_{xx} x_{\tau} + u_{xy} y_{\tau} \\
  & \dod{u_y}{\tau} = u_{yx} x_{\tau} + u_{yy} y_{\tau}

We also have :math:`A u_{xx} + B u_{xy} + C u_{yy} = H`.  Together we have the
following matrix-vector form

.. math::

  \arraycolsep=1.4pt\def\arraystretch{2.2}
  \left(\begin{array}{ccc}
    x_{\tau} & y_{\tau} & 0 \\
    0 & x_{\tau} & y_{\tau} \\
    A & B & C
  \end{array}\right)
  \left(\begin{array}{c}
    u_{xx} \\ u_{xy} \\ u_{yy}
  \end{array}\right)
  = \left(\begin{array}{c}
    \dod{u_x}{\tau} \\ \dod{u_y}{\tau} \\ H
  \end{array}\right)

To solve for :math:`u_{xx}`, :math:`u_{xy}`, and :math:`u_{yy}`, we use the
Cramer's rule:

.. math::

  u_{xx} = \frac{\mathrm{D}_{xx}}{\mathrm{D}},
  u_{xy} = \frac{\mathrm{D}_{xy}}{\mathrm{D}},
  u_{yy} = \frac{\mathrm{D}_{yy}}{\mathrm{D}}

where

.. math::

  \mathrm{D} &= \left|\begin{array}{ccc}
    x_{\tau} & y_{\tau} & 0 \\
    0 & x_{\tau} & y_{\tau} \\
    A & B & C
  \end{array}\right|
  = C x_{\tau}^2 + A y_{\tau}^2 - B x_{\tau}y_{\tau}, \\
  \mathrm{D}_{xx} &=
  \arraycolsep=1.4pt\def\arraystretch{2.2}
  \left|\begin{array}{ccc}
    \dod{u_x}{\tau} & y_{\tau} & 0 \\
    \dod{u_y}{\tau} & x_{\tau} & y_{\tau} \\
    H & B & C
  \end{array}\right| = \ldots, \\
  \mathrm{D}_{xy} &=
  \arraycolsep=1.4pt\def\arraystretch{2.2}
  \left|\begin{array}{ccc}
    x_{\tau} & \dod{u_x}{\tau} & 0 \\
    0 & \dod{u_y}{\tau} & y_{\tau} \\
    A & H & C
  \end{array}\right| = \ldots, \\
  \mathrm{D}_{yy} &=
  \arraycolsep=1.4pt\def\arraystretch{2.2}
  \left|\begin{array}{ccc}
    x_{\tau} & y_{\tau} & \dod{u_x}{\tau} \\
    0 & x_{\tau} & \dod{u_y}{\tau} \\
    A & B & H
  \end{array}\right| = \ldots

Along the characteristic lines, :math:`u_{xx}`, :math:`u_{xy}`, and
:math:`u_{yy}` are undefined.  There is not viable solution for :math:`u_{xx}`,
:math:`u_{xy}`, and :math:`u_{yy}`, :math:`\Rightarrow \mathrm{D} = 0`,

.. math::

  & C \left(\dod{x}{\tau}\right)^2 + A \left(\dod{y}{\tau}\right)^2
  - B \dod{x}{\tau}\dod{y}{\tau} = 0 \\
  \Rightarrow\quad & A \left(\dod{y}{x}\right)^2
  - B \dod{y}{x} + C = 0

Let :math:`h \defeq \dod{y}{x}`, the slope of the characteristic curves is

.. math::

  h = \frac{B \pm \sqrt{B^2-4AC}}{2A}

.. NOTE: The above equation is corrected from the notes.

There are three cases:

1. :math:`B^2 - 4AC > 0`, there are two distinct real roots for :math:`h`.
2. :math:`B^2 - 4AC = 0`, there is only one real roots for :math:`h`.
3. :math:`B^2 - 4AC < 0`, there is no real roots for :math:`h`.

.. admonition:: Aside

  Recall the quadratic equation (second-order polymonial with two variables)

  .. math::

    a x^2 + b xy + c y^2 + d x + e y + f = 0

  - :math:`b^2-4ac > 0` means the equation is hyperbolic; :math:`xy = k,
    \frac{x^2}{a^2} - \frac{y^2}{b^2} = k`.
  - :math:`b^2-4ac = 0` means the equation is parabolic; :math:`y^2 = 4p x`.
  - :math:`b^2-4ac < 0` means the equation is elliptic; :math:`\frac{x^2}{a^2}
    + \frac{y^2}{b^2} = k`

  For the PDE:

  .. math::

    A u_{xx} + B u_{xy} + C u_{yy} + D u_x + E u_y + F = 0

  - :math:`B^2-4AC > 0` means the PDE is hyperbolic.
  - :math:`B^2-4AC = 0` means the PDE is parabolic.
  - :math:`B^2-4AC < 0` menas the PDE is elliptic.

Canonical Form of Second-Order PDEs
===================================

Perform coordinate transformation

.. math::

  (x, y) \rightarrow (\xi, \eta)

We obtain

.. math::

  u_x &= u_{\xi}\xi_x + u_{\eta}\eta_x \\
  u_y &= u_{\xi}\xi_y + u_{\eta}\eta_y \\
  u_{xx} &= \dpd{(u_{\xi}\xi_x + u_{\eta}\eta_x)}{x} \\
  &= (u_{\xi\xi} \xi_x + u_{\xi\eta} \eta_x)\xi_x  + u_{\xi} \xi_{xx}
   + (u_{\eta\xi}\xi_x + u_{\eta\eta}\eta_x)\eta_x + u_{\eta}\eta_{xx} \\
  &= u_{\xi\xi}\xi_x^2 + 2u_{\xi\eta}\xi_x\eta_x + u_{\eta\eta}\eta_x^2
   + u_{\xi}\xi_{xx} + u_{\eta}\eta_{xx} \\
  u_{yy} &= \dpd{(u_{\xi}\xi_y + u_{\eta}\eta_y)}{y} \\
  &= (u_{\xi\xi} \xi_y + u_{\xi\eta} \eta_y)\xi_y  + u_{\xi} \xi_{yy}
   + (u_{\eta\xi}\xi_y + u_{\eta\eta}\eta_y)\eta_y + u_{\eta}\eta_{yy} \\
  &= u_{\xi\xi}\xi_y^2 + 2u_{\xi\eta}\xi_y\eta_y + u_{\eta\eta}\eta_y^2
   + u_{\xi}\xi_{yy} + u_{\eta}\eta_{yy} \\
  u_{xy} &= \dpd{(u_{\xi}\xi_x + u_{\eta}\eta_x)}{y} \\
  &= (u_{\xi\xi} \xi_y + u_{\xi\eta} \eta_y)\xi_x  + u_{\xi} \xi_{xy}
   + (u_{\eta\xi}\xi_y + u_{\eta\eta}\eta_y)\eta_x + u_{\eta}\eta_{xy} \\
  &= u_{\xi\xi}\xi_x\xi_y + (\xi_x\eta_y + \xi_y\eta_x)u_{\xi\eta}
   + u_{\eta\eta}\eta_x\eta_y + u_{\xi}\xi_{xy} + u_{\eta}\eta_{xy}

Substitute into

.. math::

  &A u_{xx} + B u_{xy} + C u_{yy} = H \\
  \Rightarrow\quad &
  A(u_{\xi\xi}\xi_x^2 + 2u_{\xi\eta}\xi_x\eta_x + u_{\eta\eta}\eta_x^2
   + u_{\xi}\xi_{xx} + u_{\eta}\eta_{xx}) \\
  & + B[u_{\xi\xi}\xi_x\xi_y + (\xi_x\eta_y + \xi_y\eta_x)u_{\xi\eta}
   + u_{\eta\eta}\eta_x\eta_y + u_{\xi}\xi_{xy} + u_{\eta}\eta_{xy}] \\
  & + C(u_{\xi\xi}\xi_y^2 + 2u_{\xi\eta}\xi_y\eta_y + u_{\eta\eta}\eta_y^2
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

.. TODO: The notes have negative signs of the B in A and C bar, that I can't
   reproduce.

We have

.. math::

  \bar{B}^2 - 4\bar{A}\bar{C} = (B^2 - 4AC)(\xi_x\eta_y - \xi_y\eta_x)^2

Thus :math:`B^2 - 4AC` and :math:`\bar{B}^2 - 4\bar{A}\bar{C}` have the same
sign as long as :math:`\xi_x\eta_y - \xi_y\eta_x` is not zero.  After an
arbitrary coordinate transformation, the property of the PDE does not change!

.. TODO: The note have the square for the Jacobian, that I can't reproduce.

.. admonition:: Aside

  :math:`\xi_x\eta_y - \xi_y\eta_x = J = \dpd{(\xi\eta)}{(xy)}` is the Jacobian
  of the coordinate transformation.  For a non-singular coordinate
  transformation,

  .. math::

    J \ne 0, \, J^2 > 0

As discussed above, each class of the second-order PDE with two independent
variables can be reduced to a representative canonical form:

1. Hyperbolic PDE:

   .. math::

     u_{\xi\eta} &= \tilde{H}(\xi, \eta, u, u_{\xi}, u_{\eta}) \\
     u_{\xi\xi} - u_{\eta\eta} &= \tilde{H}^*(\xi, \eta, u, u_{\xi}, u_{\eta})

2. Parabolic PDE:

   .. math::

     u_{\xi\xi} = \tilde{H}(\xi, \eta, u, u_{\xi}, u_{\eta})

3. Elliptic PDE:

   .. math::

     u_{\xi\xi} + u_{\eta\eta} = \tilde{H}(\xi, \eta, u, u_{\xi}, u_{\eta})

Hyperbolic PDEs
===============

.. math::

  &B^2 - 4AC > 0 \\

  &Ah^2 - Bh + C = 0, \quad h = \dod{y}{x}
  \, \mbox{the slope of characteristic curves} \\

  h = \frac{B \pm \sqrt{B^2 - 4AC}}{2A} = \lambda_1, \lambda_2,
  \quad \mbox{two distince real roots}

Consider the specific coordinate transformation in the following

.. math::

  \def\arraystretch{2.2}
  \begin{array}{rcl|rcl}
    \dod{y}{x} &=& \lambda_1 & 
    \dod{y}{x} &=& \lambda_2 \\
    \dod{y}{x} &=& \lambda_1 \dod{\xi}{\xi} &&& \\
    \dod{\xi}{x} &=& \lambda_1 \dpd{\xi}{y} &&& \\
    \xi_x &=& \lambda_1 \xi_y &
    \eta_x &=& \lambda_2\eta_y
  \end{array}

recall that 

.. math::

  \bar{A} &= A \xi_x^2 - B \xi_x\xi_y + C \xi_y^2 \\
          &= A \lambda_1^2\xi_y^2 - B \lambda_1\xi_y^2 + C \xi_y^2 \\
          &= (A \lambda_1^2 - B \lambda_1 + C) \xi_y^2 = 0 \\
  \bar{C} &= A \eta_x^2 - B \eta_x\eta_y + C \eta_y^2 \\
          &= (A \lambda_2^2 - B \lambda_2 + C) \eta_y^2 = 0

Thus the PDE becomes

.. math::

  -\bar{B} u_{\xi\eta} = \bar{H}

or

.. math::

  u_{\xi\eta} = \chi(\xi, \eta, u, u_{\xi}, u_{\eta})

A canonical form of the hyperbolic PDE.

.. vim: set spell ft=rst ff=unix fenc=utf8:
