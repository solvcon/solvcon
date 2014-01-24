========================
Viscoelastic Wave Solver
========================

.. py:module:: solvcon.parcel.vewave

Mathematical Model
==================

For isothermal viscoelastical material, the model equations consist
conservation of mass and momentum as follows,

.. math::
  :label: vewave.model

  \dpd{v_i}{t} - \frac{1}{\rho}\sigma_{ji,j} = 0 \\
  \dpd{\sigma_{ij}}{t} - \delta_{ij}(G^{\psi}_e + \sum^L_{l=1}
    G^{\psi}_l)v_{k,k} + 
    2\delta_{ij}(G^{\mu}_e + \sum^L_{l=1}G^{\mu}_l)v_{k,k} - 
    (G^{\mu}_e + \sum^L_{l=1}G^{\mu}_l)v_{i,j} \nonumber\\ 
    -(G^{\mu}_e + \sum^L_{l=1}G^{\mu}_l)v_{j,i} = 
    \sum^L_{l=1}\gamma^l_{ij} \\
  \dpd{\gamma^l_{ij}}{t} - \delta_{ij}(-\frac{G^{\psi}_l}{
    \tau_{\sigma l}})
    v_{k,k} + \delta_{ij}(-\frac{G^{\mu}_l}{\tau_{\sigma l}})
    v_{k,k} - (-\frac{G^{\mu}_l}{\tau_{\sigma l}})
    v_{i,j} - (-\frac{G^{\mu}_l}{\tau_{\sigma l}})
    v_{j,i} = -\frac{1}{\tau_{\sigma l}}\gamma^l_{ij}

where :math:`v_1, v_2,` and :math:`v_3` the Cartesian component of the 
velocity, :math:`\rho` is the density, :math:`\sigma_{ij}` is the stress 
tensor, :math:`\delta` is the Delta function, :math:`G^{\psi}_e, G^{\psi}_l,
G^{\mu}_e, G^{\mu}_l,` and :math:`\tau_{\sigma l}` are constants of the
standard linear solid model.

Equation (:eq:`vewave.model`) can be further organized to a matrix vector
form as follows,

.. math::
  :label: vewave.matvec

  \dpd{U}{t} + \dpd{E}{x_1} + \dpd{F}{x_2} + \dpd{G}{x_3} = S,

where :math:`U` is the solution matrix and :math:`E, F`, and :math:`G` are flux
matrices.  Apply chain rule to Eq.(:eq:`vewave.matvec`), we can derive the 
Jacobian matrix for each flux matrix.  To be complete, the Jacobian matrices
are provided as follows,

.. math::
  :label: vewave.jacos

  \dpd{U}{t} + A\dpd{U}{x_1} + B\dpd{U}{x_2} + C\dpd{U}{x_3} = S

.. math::
  :label: vewave.jacoM

  M = \left( \begin{array}{c|c|c}
      M_1 & M_2 & M_3 \\
      \hline
      M_4 & M_5 & M_6
      \end{array} \right).

.. math::
  :label: vewave.jacoA

   M_2 = \left( \begin{array}{cccccc}
        -\frac{1}{\rho} & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & -\frac{1}{\rho} \\
        0 & 0 & 0 & 0 & -\frac{1}{\rho} & 0 \\
        \end{array} \right),
  \nonumber \\
  M_4 = \left( \begin{array}{ccc}
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
        \end{array} \right).

.. math::
  :label: vewave.jacoB

  M_2 = \left( \begin{array}{cccccc}
        0 & 0 & 0 & 0 & 0 & -\frac{1}{\rho} \\
        0 & -\frac{1}{\rho} & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & -\frac{1}{\rho} & 0 & 0 \\
        \end{array} \right),
  \nonumber \\
  M_4 = \left( \begin{array}{ccc}
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
        \end{array} \right).

.. math::
  :label: vewave.jacoC

  M_2 = \left( \begin{array}{cccccc}
        0 & 0 & 0 & 0 & -\frac{1}{\rho} & 0 \\
        0 & 0 & 0 & -\frac{1}{\rho} & 0 & 0 \\
        0 & 0 & -\frac{1}{\rho} & 0 & 0 & 0 \\
        \end{array} \right),
  \nonumber \\
  M_4 = \left( \begin{array}{ccc}
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
        \end{array} \right).

The left hand side of the model equation (Eq.~(:eq:`ve.jacos`)) can be proved
as a hyperbolic system.  The method of proof is similar to the "Second-Order 
Hydro-Acoustic Solver Based on Bulk Modulus".  Here we list the eigenvalues to 
complete this page.

.. math::
  :label: vewave.eigValue

  \lambda_{1,2,3,4,5,6\cdots} = 
  \pm\sqrt{ar(k^2_1+k^2_2+k^2_3)},
  \pm\sqrt{br(k^2_1+k^2_2+k^2_3)},
  \pm\sqrt{br(k^2_1+k^2_2+k^2_3)},
  0,\cdots,

where :math:`r = \frac{1}{\rho}, a = G^{\psi}_e+\sum^L_{l=1}G^{\psi}_l`, and
:math:`b = G^{\mu}_e+\sum^L_{l=1}G^{\mu}_l`.  The :math:`k_1, k_2`, and 
:math:`k_3` are the Euler angles, which are as same as in the "Second-Order 
Hydro-Acoustic Solver Based on Bulk Modulus".


This is the placeholder for formulations of viscoelastic wave solver.
References can be inserted like [VEWAVE14]_.

.. math::
  :label: vewave.comass

  \dpd{\rho}{t} + \sum_{i=1}^3 \dpd{\rho v_i}{x_i} = 0

Bibliography
============

.. [VEWAVE14] Jane Doe, "Snake Oil",
  *Jungle of Possibility*,
  Volume 106, Issue 2, Feb. 2014, Page 149-164. `doi:
  28825252 <http://dx.doi.org/28825252>`__.
