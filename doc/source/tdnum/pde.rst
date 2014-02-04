========================================
What Are Partial Differential Equations?
========================================

A partial differential equation (PDE) is an equation that contains partial
derivatives of the unknown :math:`u`.  When :math:`u = u(x, y, z, t)`, we have

.. math::

  \dpd{u}{t}, \dpd{u}{x}, \dpd{u}{y}, \dpd{u}{z},
  \dpd[2]{u}{x}, \dmd{u}{2}{x}{}{y}{}, \ldots

Sometimes subscripts are used to denote partial derivatives: :math:`u_t`,
:math:`u_x`, :math:`u_y`, :math:`u_z`, :math:`u_{xx}`, etc.  Now we can write
an arbitrary PDE like

.. math::

  a_1 u + a_2 u_x + a_3 u_{xx} + a_4 uu_{xy} + a_5 u_{tt} = a_0

In a linear PDE, :math:`u` and it derivatives appear in the equation linearly.
There is no product of :math:`u` and its derivatives, such as :math:`uu_x`,
:math:`(u_x)^2`, etc.

In a homogeneous PDE, there is no sink or source term in the equation.  In the
above example, :math:`a_0 = 0`.

For a linear and homogeneous PDE, we have the principle of superposition.
Individual solutions of the PDE can be added together to form a new solution.
For example, let :math:`\hat{u}` and :math:`\tilde{u}` be solution of

.. math::

  a u_{xx} + b u_{xy} + c u_{yy} = 0

Thus :math:`a \hat{u}_{xx} + b \hat{u}_{xy} + c \hat{u}_{yy} = 0` and :math:`a
\tilde{u}_{xx} + b \tilde{u}_{xy} + c \tilde{u}_{yy} = 0` hold true.  We have

.. math::

  a (\hat{u}_{xx}+\tilde{u}_{xx}) + b (\hat{u}_{xy}+\tilde{u}_{xy})
  + c (\hat{u}_{yy}+\tilde{u}_{yy}) = 0

First-Order PDE in Two Dimensions
=================================

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

Non-Linear One-Dimensional PDE
==============================

In the inviscid Burger's equation

.. math::

  u_t + u u_x = 0

the wave speed is :math:`u` itself.  The profile of :math:`u` would change in
the time-evolving solution of :math:`u`.

.. TODO: add illustrative figures.

Second-Order Linear PDEs with Two Independent Variables
=======================================================

.. math::
  :label: tdnum.pde2d2v

  A u_{xx} + B u_{xy} + C u_{yy} + D u_x + E u_y + F u + G = 0

For linear PDEs, :math:`A, B, C, D, E, F, G` could be functions of :math:`x`
and :math:`y`.

For nonlinear PDEs, :math:`A, B, C, D, E, F, G` are functions of :math:`u` and
it derivatives, e.g.,

.. math::

  A = A(u, u_x, u_y, u_{xx}, \ldots)

The order of a PDE is determined by the higher order of the derivatives in the
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

  & \mathrm{D} = \left|\begin{array}{ccc}
    x_{\tau} & y_{\tau} & 0 \\
    0 & x_{\tau} & y_{\tau} \\
    A & B & C
  \end{array}\right|
  = C x_{\tau}^2 + A y_{\tau}^2 - B x_{\tau}y_{\tau}, \\
  & \mathrm{D}_{xx} = \left|\begin{array}{ccc}
    \dod{u_x}{\tau} & y_{\tau} & 0 \\
    \dod{u_y}{\tau} & x_{\tau} & y_{\tau} \\
    H & B & C
  \end{array}\right| = \ldots, \\
  & \mathrm{D}_{xy} = \left|\begin{array}{ccc}
    x_{\tau} & \dod{u_x}{\tau} & 0 \\
    0 & \dod{u_y}{\tau} & y_{\tau} \\
    A & H & C
  \end{array}\right| = \ldots, \\
  & \mathrm{D}_{yy} = \left|\begin{array}{ccc}
    x_{\tau} & y_{\tau} & \dod{u_x}{\tau} \\
    0 & x_{\tau} & \dod{u_y}{\tau} \\
    A & B & H
  \end{array}\right| = \ldots

Along the characteristic lines, :math:`u_{xx}`, :math:`u_{xy}`, and
:math:`u_{yy}` are undefined.    There is not viable solution for
:math:`u_{xx}`, :math:`u_{xy}`, and :math:`u_{yy}`.

If :math:`\mathrm{D} = 0`,

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

    A u_{xx} + B u_{xy} + C u_{yy} + D u_x + E u_y + F = 0

  - :math:`B^2-4AC > 0` means the PDE is hyperbolic.
  - :math:`B^2-4AC = 0` means the PDE is parabolic.
  - :math:`B^2-4AC < 0` menas the PDE is elliptic.

.. vim: set spell ft=rst ff=unix fenc=utf8:
