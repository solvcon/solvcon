=================
Viscoelastic Wave
=================

.. py:module:: solvcon.parcel.vewave

Viscoelastic Model
==================

[Yang13]_

Mathematical Model
==================

For isothermal viscoelastic material, the model equations consist conservation
of mass and momentum as follows,

.. math::
  :label: vewave.model

  & \dpd{v_i}{t} - \frac{1}{\rho}\sigma_{ji,j} = 0 \\
  & \dpd{\sigma_{ij}}{t} - \delta_{ij}(G^{\psi}_e + \sum^L_{l=1}
    G^{\psi}_l)v_{k,k} + 
    2\delta_{ij}(G^{\mu}_e + \sum^L_{l=1}G^{\mu}_l)v_{k,k}
    - (G^{\mu}_e + \sum^L_{l=1}G^{\mu}_l)v_{i,j} 
    - (G^{\mu}_e + \sum^L_{l=1}G^{\mu}_l)v_{j,i} = \sum^L_{l=1}\gamma^l_{ij} \\
  & \dpd{\gamma^l_{ij}}{t} - \delta_{ij}(-\frac{G^{\psi}_l}{
    \tau_{\sigma l}})
    v_{k,k} + \delta_{ij}(-\frac{G^{\mu}_l}{\tau_{\sigma l}})
    v_{k,k} - (-\frac{G^{\mu}_l}{\tau_{\sigma l}})
    v_{i,j} - (-\frac{G^{\mu}_l}{\tau_{\sigma l}})
    v_{j,i} = -\frac{1}{\tau_{\sigma l}}\gamma^l_{ij}

where :math:`v_i` are the Cartesian component of the velocity, :math:`\rho` the
density, :math:`\sigma_{ij}` the stress tensor, :math:`\gamma_{ij}` the
internal variables, and :math:`\delta_{ij}` the Kronecker delta.  Subscripts
:math:`i, j, k = 1, 2, 3` are for the Cartesian tensors.
:math:`\square_{\square,\square}` denotes derivatives.  Einstein's summation
rule is applied to the Cartesian tensors.  :math:`G^{\psi}_e, G^{\psi}_l,
G^{\mu}_e, G^{\mu}_l,` and :math:`\tau_{\sigma l}` are the constants of the
standard linear solid model.

Jacobian Matrices
+++++++++++++++++

Equation :eq:`vewave.model` can be further organized to a matrix vector form as
follows,

.. math::
  :label: vewave.matvec

  \dpd{U}{t} + \dpd{E}{x_1} + \dpd{F}{x_2} + \dpd{G}{x_3} = S

where :math:`U` is the solution matrix and :math:`E, F`, and :math:`G` are flux
matrices.  By applying the chain rule to Eq. :eq:`vewave.matvec`, we can derive
the Jacobian matrices for the flux matrices:

.. math::
  :label: vewave.jacos

  \dpd{U}{t} + A\dpd{U}{x_1} + B\dpd{U}{x_2} + C\dpd{U}{x_3} = S

The Jacobian matrices (A, B, and C) can have a general form that is described 
by M:

.. math::
  :label: vewave.jacoM

  M = \left( \begin{array}{c|c|c}
      M_1 & M_2 & M_3 \\
      \hline
      M_4 & M_5 & M_6
      \end{array} \right)

For all Jacobian matrices, M1, M3, M5, and M6 are the same.
They are all zero two-dimensional matrices with size, 3 by 3, 3 by (6xL), 
(6+6xL) by 6, and (6+6xL) by 6xL, respectively.  The L is the number in 
:eq:`vewave.model` and is decided by the number of the SLS constants 
:math:`G^{\phi}` and :math:`G^{\mu}` are needed.
The difference of M2 and M4 between each Jacobian matrices are given below.

For Jacobian matrix A,

.. math::
  :label: vewave.jacoA

  & M_2 = \left( \begin{array}{cccccc}
        -\frac{1}{\rho} & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & -\frac{1}{\rho} \\
        0 & 0 & 0 & 0 & -\frac{1}{\rho} & 0 \\
        \end{array} \right) \\
  & M_4 = \left( \begin{array}{ccc}
        (-G^{\psi}_e-\sum^L_{l=1}G^{\psi}_l) & 0 & 0 \\\relax
        [2G^{\mu}_e-G^{\psi}_e+\sum^L_{l=1}(2G^{\mu}_l-G^{\psi}_l)] &
        0 & 0 \\\relax
        [2G^{\mu}_e-G^{\psi}_e+\sum^L_{l=1}(2G^{\mu}_l-G^{\psi}_l)] &
        0 & 0 \\
        %
        0 & 0 & 0 \\
        0 & 0 & (-G^{\mu}_e-\sum^L_{l=1}G^{\mu}_l) \\
        0 & (-G^{\mu}_e-\sum^L_{l=1}G^{\mu}_l) & 0 \\
        %
        (\frac{G^{\psi}_l}{\tau_{\sigma l}}+\frac{G^{\mu}_l}
          {\tau_{\sigma l}})
        & 0 & 0 \\
        (\frac{G^{\psi}_l}{\tau_{\sigma l}}-\frac{G^{\mu}_l}
          {\tau_{\sigma l}})
        & 0 & 0 \\
        (\frac{G^{\psi}_l}{\tau_{\sigma l}}-\frac{G^{\mu}_l}
          {\tau_{\sigma l}})
        & 0 & 0 \\
        %
        0 & 0 & 0 \\
        0 & 0 & \frac{G^{\mu}_l}{\tau_{\sigma l}} \\
        0 & \frac{G^{\mu}_l}{\tau_{\sigma l}} & 0
        \end{array} \right)

For Jacobian matrix B,

.. math::
  :label: vewave.jacoB

  & M_2 = \left( \begin{array}{cccccc}
        0 & 0 & 0 & 0 & 0 & -\frac{1}{\rho} \\
        0 & -\frac{1}{\rho} & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & -\frac{1}{\rho} & 0 & 0 \\
        \end{array} \right) \\
  & M_4 = \left( \begin{array}{ccc}
        0 & [2G^{\mu}_e-G^{\psi}_e+\sum^L_{l=1}
          (2G^{\mu}_l-G^{\psi}_l)] &
        0 \\
        0 & (-G^{\psi}_e-\sum^L_{l=1}G^{\psi}_l) & 0 \\
        0 & [2G^{\mu}_e-G^{\psi}_e+\sum^L_{l=1}
          (2G^{\mu}_l-G^{\psi}_l)] &
        0 \\
        %
        0 & 0 & (-G^{\mu}_e-\sum^L_{l=1}G^{\mu}_l) \\
        0 & 0 & 0 \\
        (-G^{\mu}_e-\sum^L_{l=1}G^{\mu}_l) & 0 & 0\\
        %
        0 & (\frac{G^{\psi}_l}{\tau_{\sigma l}}-\frac{G^{\mu}_l}{
          \tau_{\sigma l}})
        & 0 \\
        0 & (\frac{G^{\psi}_l}{\tau_{\sigma l}}+\frac{G^{\mu}_l}{
          \tau_{\sigma l}})
        & 0 \\
        0 & (\frac{G^{\psi}_l}{\tau_{\sigma l}}-\frac{G^{\mu}_l}{
          \tau_{\sigma l}})
        & 0 \\
        %
        0 & 0 & \frac{G^{\mu}_l}{\tau_{\sigma l}} \\
        0 & 0 & 0 \\
        \frac{G^{\mu}_l}{\tau_{\sigma l}} & 0 & 0
        \end{array} \right)

For Jacobian matrix C,

.. math::
  :label: vewave.jacoC

  & M_2 = \left( \begin{array}{cccccc}
        0 & 0 & 0 & 0 & -\frac{1}{\rho} & 0 \\
        0 & 0 & 0 & -\frac{1}{\rho} & 0 & 0 \\
        0 & 0 & -\frac{1}{\rho} & 0 & 0 & 0 \\
        \end{array} \right) \\
  & M_4 = \left( \begin{array}{ccc}
        0 & 0 &
        [2G^{\mu}_e-G^{\psi}_e+\sum^L_{l=1}
          (2G^{\mu}_l-G^{\psi}_l)] \\
        0 & 0 &
        [2G^{\mu}_e-G^{\psi}_e+\sum^L_{l=1}
          (2G^{\mu}_l-G^{\psi}_l)] \\
        0 & 0 & (-G^{\psi}_e-\sum^L_{l=1}G^{\psi}_l) \\
        %
        0 & (-G^{\mu}_e-\sum^L_{l=1}G^{\mu}_l) & 0 \\
        (-G^{\mu}_e-\sum^L_{l=1}G^{\mu}_l) & 0 & 0 \\
        0 & 0 & 0 \\
        %
        0 & 0 &
        (\frac{G^{\psi}_l}{\tau_{\sigma l}}-\frac{G^{\mu}_l}{
          \tau_{\sigma l}}) \\
        0 & 0 &
        (\frac{G^{\psi}_l}{\tau_{\sigma l}}-\frac{G^{\mu}_l}{
          \tau_{\sigma l}}) \\
        0 & 0 &
        (\frac{G^{\psi}_l}{\tau_{\sigma l}}+\frac{G^{\mu}_l}{
          \tau_{\sigma l}}) \\
        %
        0 & \frac{G^{\mu}_l}{\tau_{\sigma l}} & 0\\
        \frac{G^{\mu}_l}{\tau_{\sigma l}} & 0 & 0 \\
        0 & 0 & 0
        \end{array} \right)

The left hand side of the model equation Eq. :eq:`vewave.jacos` can be proved
as a hyperbolic system.  The method of proof is similar to the :doc:`app_bulk`.
The list of the eigenvalues is provided:

.. math::
  :label: vewave.eigValue

  \lambda_{1,2,3,4,5,6\cdots} = 
  \pm\sqrt{ar(k^2_1+k^2_2+k^2_3)},
  \pm\sqrt{br(k^2_1+k^2_2+k^2_3)},
  \pm\sqrt{br(k^2_1+k^2_2+k^2_3)},
  0,\cdots,

where :math:`r = \frac{1}{\rho}, a = G^{\psi}_e+\sum^L_{l=1}G^{\psi}_l`, and
:math:`b = G^{\mu}_e+\sum^L_{l=1}G^{\mu}_l`.  The :math:`k_1, k_2`, and
:math:`k_3` are the components of a direction vector, as used in
:doc:`app_bulk`.
