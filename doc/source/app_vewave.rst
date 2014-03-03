=====================================
Viscoelastic Wave (Under Development)
=====================================

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

  & \dpd{v_i}{t} - \frac{1}{\rho} \sum_{j=1}^3\dpd{\sigma_{ji}}{x_j} = 0 \\
  & \dpd{\sigma_{ij}}{t}
    - \delta_{ij} \left( G^{\psi}_e + \sum^L_{l=1} G^{\psi}_l \right)
      \sum_{k=1}^3 \dpd{v_k}{x_k}
    + \left( G^{\mu}_e + \sum^L_{l=1}G^{\mu}_l \right)
      \left(
      2 \delta_{ij} \sum_{k=1}^3 \dpd{v_k}{x_k}
      - \dpd{v_i}{x_j} - \dpd{v_j}{x_i} \right)
    = \sum^L_{l=1}\gamma^l_{ij} \\
  & \dpd{\gamma^l_{ij}}{t}
    + \delta_{ij} \frac{G^{\psi}_l - G^{\mu}_l}{\tau_{\sigma l}}
      \sum_{k=1}^3 \dpd{v_k}{x_k}
    + \frac{G^{\mu}_l}{\tau_{\sigma l}}
      \left( \dpd{v_i}{x_j} + \dpd{v_j}{x_i} \right)
    = -\frac{1}{\tau_{\sigma l}}\gamma^l_{ij}

where :math:`v_i` are the Cartesian component of the velocity, :math:`\rho` the
density, :math:`\sigma_{ij}` the stress tensor, :math:`\gamma_{ij}` the
internal variables, and :math:`\delta_{ij}` the Kronecker delta.  Subscripts
:math:`i, j, k = 1, 2, 3` are for the Cartesian tensors.  :math:`G^{\psi}_l,
G^{\mu}_e, G^{\mu}_l`, and :math:`\tau_{\sigma l}` are the constants of the
standard linear solid (SLS) model with :math:`l = 1, 2, \ldots, L`.  :math:`L`
is the number of the employed SLS model components.

Equation :eq:`vewave.model` can be further organized to a vector form:

.. math::
  :label: vewave.gevec

  \dpd{\bvec{u}}{t} + \sum_{k=1}^3 \dpd{\bvec{f}^{(k)}}{x_k} = \bvec{s}

where :math:`\bvec{u}` is the solution variable, :math:`\bvec{f}^{(1)}`,
:math:`\bvec{f}^{(2)}`, and :math:`\bvec{f}^{(3)}` flux functions, and
:math:`\bvec{s}` the source term.

Jacobian Matrices
+++++++++++++++++

By applying the chain rule to Eq.  :eq:`vewave.gevec`, we can derive the
Jacobian matrices:

.. math::
  :label: vewave.gemat

  \dpd{\bvec{u}}{t} + \sum_{k=1}^3 \mathrm{A}^{(k)} \dpd{\bvec{u}}{x_k}
  = \bvec{s}

where :math:`\mathrm{A}^{(1)}`, :math:`\mathrm{A}^{(2)}`, and
:math:`\mathrm{A}^{(3)}` are :math:`(9+6L)\times(9+6L)` are the Jacobian
matrices:

.. math::
  :label: vewave.jacos

  \mathrm{A}^{(i)} \defeq \dpd{\bvec{f}^{(i)}}{\bvec{u}}
  = \left( \begin{array}{c|c|c}
    \mathrm{0}_{3\times3} & \mathrm{C}^{(i)} & \mathrm{0}_{3\times(6L)} \\
    \hline
    \mathrm{B}^{(i)} & \mathrm{0}_{(6+6L)\times6} &
    \mathrm{0}_{(6+6L)\times(6L)}
  \end{array} \right), \quad i = 1, 2, 3

where

.. math::

  \mathrm{B}^{(i)} \defeq \left( \begin{array}{ccc}
    \left[ 2(G^{\mu}_e  + \sum^L_{l=1} G^{\mu}_l)
          - (G^{\psi}_e + \sum^L_{l=1} G^{\psi}_l) \right]
    \mathrm{M}^{(i)}
    - (G^{\psi}_e+\sum^L_{l=1}G^{\psi}_l) \mathrm{K}^{(i)}
    \\
    \frac{G^{\phi}_1 - G^{\mu}_1}{\tau_{\sigma 1}} \mathrm{M}^{(i)}
    + \frac{G^{\phi}_1}{\tau_{\sigma 1}} \mathrm{N}^{(i)}
    + \frac{G^{\mu}_1}{\tau_{\sigma 1}} \mathrm{K}^{(i)}
    \\
    \vdots \\
    \frac{G^{\phi}_L - G^{\mu}_L}{\tau_{\sigma 1}} \mathrm{M}^{(i)}
    + \frac{G^{\phi}_L}{\tau_{\sigma 1}} \mathrm{N}^{(i)}
    + \frac{G^{\mu}_L}{\tau_{\sigma 1}} \mathrm{K}^{(i)}
  \end{array} \right), \,
  \mathrm{C}^{(i)} \defeq -\frac{1}{\rho} {\mathrm{K}^{(i)}}^t,
  \quad i = 1, 2, 3

and

.. math::
  :label: vewave.dirMmat

  \mathrm{M}^{(1)} \defeq \left( \begin{array}{ccc}
    0 & 0 & 0 \\
    1 & 0 & 0 \\
    1 & 0 & 0 \\
    0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0
  \end{array} \right), \,
  \mathrm{M}^{(2)} \defeq \left( \begin{array}{ccc}
    0 & 1 & 0 \\
    0 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0
  \end{array} \right), \,
  \mathrm{M}^{(3)} \defeq \left( \begin{array}{ccc}
    0 & 0 & 1 \\
    0 & 0 & 1 \\
    0 & 0 & 0 \\
    0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0
  \end{array} \right)

.. math::
  :label: vewave.dirNmat

  \mathrm{N}^{(1)} \defeq \left( \begin{array}{ccc}
    1 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0
  \end{array} \right), \,
  \mathrm{N}^{(2)} \defeq \left( \begin{array}{ccc}
    0 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0
  \end{array} \right), \,
  \mathrm{N}^{(3)} \defeq \left( \begin{array}{ccc}
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 1 \\
    0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0
  \end{array} \right)

.. math::
  :label: vewave.dirKmat

  \mathrm{K}^{(1)} \defeq \left( \begin{array}{ccc}
    1 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 1 \\
    0 & 1 & 0
  \end{array} \right), \,
  \mathrm{K}^{(2)} \defeq \left( \begin{array}{ccc}
    0 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 1 \\
    0 & 0 & 0 \\
    1 & 0 & 0
  \end{array} \right), \,
  \mathrm{K}^{(3)} \defeq \left( \begin{array}{ccc}
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 1 \\
    0 & 1 & 0 \\
    1 & 0 & 0 \\
    0 & 0 & 0
  \end{array} \right)

:math:`\mathrm{B}^{(1)}`, :math:`\mathrm{B}^{(2)}`, and
:math:`\mathrm{B}^{(3)}` are :math:`(6+6L)\times3` matrices.
:math:`\mathrm{C}^{(1)}`, :math:`\mathrm{C}^{(2)}`, and
:math:`\mathrm{C}^{(3)}` are :math:`3\times6` matrices.

Hyperbolicity
+++++++++++++

The left hand side of the model equation Eq. :eq:`vewave.gemat` can be proved
as a hyperbolic system.  The method of proof is similar to the
:doc:`bulk/index`.  The list of the eigenvalues is provided:

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
:doc:`bulk/index`.
