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

.. vim: set spell ft=rst ff=unix fenc=utf8:
