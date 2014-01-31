===================================
Hydro-Acoustics (Under Development)
===================================

.. py:module:: solvcon.parcel.bulk

Mathematical Model
==================

The governing equations of the hydro-acoustic wave include the continuity
equation

.. math::
  :label: bulk.comass

  \dpd{\rho}{t} + \sum_{i=1}^3 \dpd{\rho v_i}{x_i} = 0

and the momentum equations

.. math::
  :label: bulk.comomentum

  \dpd{\rho v_j}{t}
  + \sum_{i=1}^3 \dpd{(\rho v_iv_j + \delta_{ij}p)}{x_i}
  = \dpd{}{x_j}\left(\lambda\sum_{k=1}^3\dpd{v_k}{x_k}\right)
  + \sum_{i=1}^3 \dpd{}{x_i}
                 \left[\mu \left( \dpd{v_i}{x_j}+\dpd{v_j}{x_i} \right)\right],
                 \quad j = 1, 2, 3

where :math:`\rho` is the density, :math:`v_1, v_2,` and :math:`v_3` the
Cartesian component of the velocity, :math:`p` the pressure,
:math:`\delta_{ij}, i, j = 1, 2, 3` the Kronecker delta, :math:`\lambda`
the second viscosity coefficien, :math:`\mu` the dynamic viscosity coefficient,
:math:`t` the time, and :math:`x_1, x_2`, and :math:`x_3` the Cartesian
coordinate axes.  Newtonian fluid is assumed.

The above four equations in Eqs. :eq:`bulk.comass` and :eq:`bulk.comomentum`
have five independent variables :math:`\rho, p, v_1, v_2`, and :math:`v_3`, and
hence are not closed without a constitutive relation.  In the :py:mod:`~.bulk`
package, the constitutive relation (or the equation of state) of choice is

.. math::

  K = \rho\dpd{p}{\rho}

where :math:`K` is a constant and the bulk modulus.  We chose to use the
density :math:`\rho` as the independent variable, and integrate the equation of
state to be

.. math::
  :label: bulk.eos

  p = p_0 + K \ln\frac{\rho}{\rho_0}

where :math:`p_0` and :math:`\rho_0` are constants.  Substituting Eq.
:eq:`bulk.eos` into Eq. :eq:`bulk.comomentum` gives

.. math::
  :label: bulk.comomentum_eos

  \dpd{\rho v_j}{t} + \sum_{i=1}^3\dpd{}{x_i}
  \left[\rho v_iv_j
      + \delta_{ij}\left(p_0 + K\ln\frac{\rho}{\rho_0}\right) \right]
  = \sum_{i=1}^3 \dpd{}{x_i}
                 \left[\delta_{ij} \lambda \sum_{k=1}^3 \dpd{v_k}{x_k}
                     + \mu \left( \dpd{v_i}{x_j}+\dpd{v_j}{x_i} \right)\right],
                 \quad j = 1, 2, 3

Jacobian Matrices
+++++++++++++++++

We proceed to analyze the advective part of the governing equations (Eqs.
:eq:`bulk.comass` and :eq:`bulk.comomentum_eos`).  Define the conservation
variables

.. math::
  :label: bulk.csvar

  \bvec{u} \defeq \left(\begin{array}{c}
    \rho \\ \rho v_1 \\ \rho v_2 \\ \rho v_3
  \end{array}\right)

and flux functions

.. math::
  :label: bulk.fluxf

  \bvec{f}^{(1)} \defeq \left(\begin{array}{c}
    \rho v_1 \\
    \rho v_1^2 + K\ln\frac{\rho}{\rho_0} + p_0 \\
    \rho v_1v_2 \\ \rho v_1v_3
  \end{array}\right), \quad
  \bvec{f}^{(2)} \defeq \left(\begin{array}{c}
    \rho v_2 \\ \rho v_1v_2 \\
    \rho v_2^2 + K\ln\frac{\rho}{\rho_0} + p_0 \\
    \rho v_2v_3
  \end{array}\right), \quad
  \bvec{f}^{(3)} \defeq \left(\begin{array}{c}
    \rho v_3 \\ \rho v_1v_3 \\ \rho v_2v_3 \\
    \rho v_3^2 + K\ln\frac{\rho}{\rho_0} + p_0
  \end{array}\right)

Aided by the definition of conservation variables in Eq. :eq:`bulk.csvar`, the
flux functions defined in Eq. :eq:`bulk.fluxf` can be rewritten with
:math:`u_1, u_2, u_3`, and :math:`u_4`

.. math::
  :label: bulk.fluxfu

  \bvec{f}^{(1)} = \left(\begin{array}{c}
    u_2 \\
    \frac{u_2^2}{u_1} + K\ln\frac{u_1}{\rho_0} + p_0 \\
    \frac{u_2u_3}{u_1} \\ \frac{u_2u_4}{u_1}
  \end{array}\right), \quad
  \bvec{f}^{(2)} = \left(\begin{array}{c}
    u_3 \\ \frac{u_2u_3}{u_1} \\
    \frac{u_3^2}{u_1} + K\ln\frac{u_1}{\rho_0} + p_0 \\
    \frac{u_3u_4}{u_1}
  \end{array}\right), \quad
  \bvec{f}^{(3)} = \left(\begin{array}{c}
    u_4 \\ \frac{u_2u_4}{u_1} \\ \frac{u_3u_4}{u_1} \\
    \frac{u_4^2}{u_1} + K\ln\frac{u_1}{\rho_0} + p_0
  \end{array}\right)

By using Eq. :eq:`bulk.csvar` and Eq. :eq:`bulk.fluxfu`, the left-hand-side of
the governing equations can be cast into the conservative form

.. math::
  :label: bulk.ge.csv

  \dpd{\bvec{u}}{t} + \sum_{i=1}^3\dpd{\bvec{f}^{(i)}}{x_i} = 0

Aided by using the chain rule, Eq. :eq:`bulk.ge.csv` can be rewritten in the
quasi-linear form

.. math::
  :label: bulk.ge.qlcsv

  \dpd{\bvec{u}}{t} + \sum_{i=1}^3\mathrm{A}^{(i)}\dpd{\bvec{u}}{x_i} = 0

where the Jacobian matrices :math:`\mathrm{A}^{(1)}, \mathrm{A}^{(2)}`, and
:math:`\mathrm{A}^{(3)}` are defined as

.. math::
  :label: bulk.jacogen

  \mathrm{A}^{(i)} \defeq \left(\begin{array}{cccc}
    \pd{f_1^{(i)}}{u_1} & \pd{f_1^{(i)}}{u_2} &
    \pd{f_1^{(i)}}{u_3} & \pd{f_1^{(i)}}{u_4} \\
    \pd{f_2^{(i)}}{u_1} & \pd{f_2^{(i)}}{u_2} &
    \pd{f_2^{(i)}}{u_3} & \pd{f_2^{(i)}}{u_4} \\
    \pd{f_3^{(i)}}{u_1} & \pd{f_3^{(i)}}{u_2} &
    \pd{f_3^{(i)}}{u_3} & \pd{f_3^{(i)}}{u_4} \\
    \pd{f_4^{(i)}}{u_1} & \pd{f_4^{(i)}}{u_2} &
    \pd{f_4^{(i)}}{u_3} & \pd{f_4^{(i)}}{u_4}
  \end{array}\right), \quad i = 1, 2, 3

Aided by using Eq. :eq:`bulk.fluxfu`, the Jacobian matrices defined in Eq.
:eq:`bulk.jacogen` can be written out as

.. math::
  :label: bulk.jaco.csvar

  \mathrm{A}^{(1)} = \left(\begin{array}{cccc}
    0 & 1 & 0 & 0 \\
    -\frac{u_2^2}{u_1^2} + \frac{K}{u_1} & 2\frac{u_2}{u_1} & 0 & 0 \\
    -\frac{u_2u_3}{u_1^2} & \frac{u_3}{u_1} & \frac{u_2}{u_1} & 0 \\
    -\frac{u_2u_4}{u_1^2} & \frac{u_4}{u_1} & 0 & \frac{u_2}{u_1}
  \end{array}\right), \quad
  \mathrm{A}^{(2)} = \left(\begin{array}{cccc}
    0 & 0 & 1 & 0 \\
    -\frac{u_2u_3}{u_1^2} & \frac{u_3}{u_1} & \frac{u_2}{u_1} & 0 \\
    -\frac{u_3^2}{u_1^2} + \frac{K}{u_1} & 0 & 2\frac{u_3}{u_1} & 0 \\
    -\frac{u_3u_4}{u_1^2} & 0 & \frac{u_4}{u_1} & \frac{u_3}{u_1}
  \end{array}\right), \quad
  \mathrm{A}^{(3)} = \left(\begin{array}{cccc}
    0 & 0 & 0 & 1 \\
    -\frac{u_2u_4}{u_1^2} & \frac{u_4}{u_1} & 0 & \frac{u_2}{u_1} \\
    -\frac{u_3u_4}{u_1^2} & 0 & \frac{u_4}{u_1} & \frac{u_3}{u_1} \\
    -\frac{u_4^2}{u_1^2} & 0 & 0 & 2\frac{u_4}{u_1}
  \end{array}\right)

Hyperbolicity
+++++++++++++

Hyperbolicity is a prerequisite for us to apply the space-time CESE method to a
system of first-order PDEs.  For the governing equations, Eqs.
:eq:`bulk.comass` and :eq:`bulk.comomentum_eos`, to be hyperbolic, the linear
combination of the three Jacobian matrices of their quasi-linear for must be
diagonalizable.  The spectrum of the linear combination must be all real, too
[Warming75]_, [Chen12]_.

To facilitate the analysis, we chose to use the non-conservative version of the
governing equations.  Define the non-conservative variables

.. math::
  :label: bulk.ncsvar

  \tilde{\bvec{u}} \defeq \left(\begin{array}{c}
    \rho \\ v_1 \\ v_2 \\ v_3
  \end{array}\right) =
  \left(\begin{array}{c}
    u_1 \\ \frac{u_2}{u_1} \\ \frac{u_3}{u_1} \\ \frac{u_4}{u_1}
  \end{array}\right)

Aided by using Eqs. :eq:`bulk.ncsvar` and :eq:`bulk.csvar`, the transformation
between the conservative variables and the non-conservative variables can be
done with the transformation Jacobian defined as

.. math::
  :label: bulk.noncstrans

  \mathrm{P} \defeq \dpd{\tilde{\bvec{u}}}{\bvec{u}} =
  \left(\begin{array}{cccc}
    \pd{\tilde{u}_1}{u_1} & \pd{\tilde{u}_1}{u_2} &
    \pd{\tilde{u}_1}{u_3} & \pd{\tilde{u}_1}{u_4} \\
    \pd{\tilde{u}_2}{u_1} & \pd{\tilde{u}_2}{u_2} &
    \pd{\tilde{u}_2}{u_3} & \pd{\tilde{u}_2}{u_4} \\
    \pd{\tilde{u}_3}{u_1} & \pd{\tilde{u}_3}{u_2} &
    \pd{\tilde{u}_3}{u_3} & \pd{\tilde{u}_3}{u_4} \\
    \pd{\tilde{u}_4}{u_1} & \pd{\tilde{u}_4}{u_2} &
    \pd{\tilde{u}_4}{u_3} & \pd{\tilde{u}_4}{u_4}
  \end{array}\right) = \left(\begin{array}{cccc}
    1 & 0 & 0 & 0 \\
    -\frac{u_2}{u_1^2} & \frac{1}{u_1} & 0 & 0 \\
    -\frac{u_3}{u_1^2} & 0 & \frac{1}{u_1} & 0 \\
    -\frac{u_4}{u_1^2} & 0 & 0 & \frac{1}{u_1}
  \end{array}\right) = \left(\begin{array}{cccc}
    1 & 0 & 0 & 0 \\
    -\frac{v_1}{\rho} & \frac{1}{\rho} & 0 & 0 \\
    -\frac{v_2}{\rho} & 0 & \frac{1}{\rho} & 0 \\
    -\frac{v_3}{\rho} & 0 & 0 & \frac{1}{\rho}
  \end{array}\right)

Aided by writing Eq. :eq:`bulk.csvar` as

.. math::

  \bvec{u} = \left(\begin{array}{c}
    \tilde{u}_1 \\
    \tilde{u}_1\tilde{u}_2 \\ \tilde{u}_1\tilde{u}_3 \\ \tilde{u}_1\tilde{u}_4
  \end{array}\right)

the inverse matrix of :math:`\mathrm{P}` can be obtained

.. math::
  :label: bulk.cstrans

  \mathrm{P}^{-1} = \dpd{\bvec{u}}{\tilde{\bvec{u}}} =
  \left(\begin{array}{cccc}
    1 & 0 & 0 & 0 \\
    \tilde{u}_2 & \tilde{u}_1 & 0 & 0 \\
    \tilde{u}_3 & 0 & \tilde{u}_1 & 0 \\
    \tilde{u}_4 & 0 & 0 & \tilde{u}_1
  \end{array}\right) = \left(\begin{array}{cccc}
    1 & 0 & 0 & 0 \\
    v_1 & \rho & 0 & 0 \\
    v_2 & 0 & \rho & 0 \\
    v_3 & 0 & 0 & \rho
  \end{array}\right)

and :math:`\mathrm{P}^{-1}\mathrm{P} = \mathrm{P}\mathrm{P}^{-1} =
\mathrm{I}_{4\times4}`.

The transformation matrix :math:`\mathrm{P}` can be used to cast the
conservative equations, Eq. :eq:`bulk.ge.qlcsv`, into non-conservative ones.
Pre-multiplying :math:`\pd{\tilde{\bvec{u}}}{\bvec{u}}` to Eq.
:eq:`bulk.ge.qlcsv` gives

.. math::
  :label: bulk.ge.qlncsv

  \dpd{\tilde{\bvec{u}}}{t} +
  \sum_{i=1}^3\tilde{\mathrm{A}}^{(i)}\dpd{\tilde{\bvec{u}}}{x_i} = 0

where

.. math::
  :label: bulk.jaco.ncsvar

  \tilde{\mathrm{A}}^{(1)} \defeq
    \mathrm{P}\mathrm{A}^{(1)}\mathrm{P}^{-1}, \quad
  \tilde{\mathrm{A}}^{(2)} \defeq
    \mathrm{P}\mathrm{A}^{(2)}\mathrm{P}^{-1}, \quad
  \tilde{\mathrm{A}}^{(3)} \defeq
    \mathrm{P}\mathrm{A}^{(3)}\mathrm{P}^{-1}

To help obtaining the expression of :math:`\tilde{\mathrm{A}}^{(1)},
\tilde{\mathrm{A}}^{(2)}`, and :math:`\tilde{\mathrm{A}}^{(3)}`, we substitute
Eq. :eq:`bulk.csvar` into Eq. :eq:`bulk.jaco.csvar` and get

.. math::
  :label: bulk.jaco.ovar

  \mathrm{A}^{(1)} = \left(\begin{array}{cccc}
    0 & 1 & 0 & 0 \\
    -v_1^2 + \frac{K}{\rho} & 2v_1 & 0 & 0 \\
    -v_1v_2 & v_2 & v_1 & 0 \\
    -v_1v_3 & v_3 & 0 & v_1
  \end{array}\right), \quad
  \mathrm{A}^{(2)} = \left(\begin{array}{cccc}
    0 & 0 & 1 & 0 \\
    -v_1v_2 & v_2 & v_1 & 0 \\
    -v_2^2 + \frac{K}{\rho} & 0 & 2v_2 & 0 \\
    -v_2v_3 & 0 & v_3 & v_2
  \end{array}\right), \quad
  \mathrm{A}^{(3)} = \left(\begin{array}{cccc}
    0 & 0 & 0 & 1 \\
    -v_1v_3 & v_3 & 0 & v_1 \\
    -v_2v_3 & 0 & v_3 & v_2 \\
    -v_3^2 + \frac{K}{\rho} & 0 & 0 & 2v_3
  \end{array}\right)

The Jacobian matrices in Eq. :eq:`bulk.jaco.ncsvar` can be spelled out with the
expressions in Eqs. :eq:`bulk.noncstrans`, :eq:`bulk.cstrans`, and
:eq:`bulk.jaco.ovar`

.. math::
  :label: bulk.jaco.ncsvar.out

  \tilde{\mathrm{A}}^{(1)} = \left(\begin{array}{cccc}
    v_1 & \rho & 0 & 0 \\
    \frac{K}{\rho^2} & v_1 & 0 & 0 \\
    0 & 0 & v_1 & 0 \\
    0 & 0 & 0 & v_1
  \end{array}\right), \quad
  \tilde{\mathrm{A}}^{(2)} = \left(\begin{array}{cccc}
    v_2 & 0 & \rho & 0 \\
    0 & v_2 & 0 & 0 \\
    \frac{K}{\rho^2} & 0 & v_2 & 0 \\
    0 & 0 & 0 & v_2
  \end{array}\right), \quad
  \tilde{\mathrm{A}}^{(3)} = \left(\begin{array}{cccc}
    v_3 & 0 & 0 & \rho \\
    0 & v_3 & 0 & 0 \\
    0 & 0 & v_3 & 0 \\
    \frac{K}{\rho^2} & 0 & 0 & v_3
  \end{array}\right)

Equation :eq:`bulk.ge.qlncsv` is hyperbolic where the linear combination of its
Jacobian matrices :math:`\tilde{\mathrm{A}}^{(1)}`,
:math:`\tilde{\mathrm{A}}^{(2)}`, and :math:`\tilde{\mathrm{A}}^{(3)}`

.. math::
  :label: bulk.jaco.ncsvar.lc

  \tilde{\mathrm{R}} \defeq \sum_{i=1}^3 k_i\tilde{\mathrm{A}}^{(i)}
  = \left(\begin{array}{cccc}
    \sum_{i=1}^3 k_iv_i & k_1\rho & k_2\rho & k_3\rho \\
    \frac{k_1K}{\rho^2} & \sum_{i=1}^3 k_iv_i & 0 & 0 \\
    \frac{k_2K}{\rho^2} & 0 & \sum_{i=1}^3 k_iv_i & 0 \\
    \frac{k_3K}{\rho^2} & 0 & 0 & \sum_{i=1}^3 k_iv_i
  \end{array}\right)

where :math:`k_1, k_2`, and :math:`k_3` are real and bounded.

The linearly combined Jacobian matrix can be used to rewrite the
three-dimensional governing equation, Eq. :eq:`bulk.ge.qlncsv`, into
one-dimensional space

.. math::
  :label: bulk.ge.qlncsv1d

  \dpd{\tilde{\bvec{u}}}{t} + \tilde{\mathrm{R}}\dpd{\tilde{\bvec{u}}}{y} = 0

where

.. math::
  :label: bulk.ge.y1d

  y \defeq \sum_{i=1}^3 k_ix_i

and aided by Eq. :eq:`bulk.jaco.ncsvar.lc` and the chain rule

.. math::

  \sum_{i=1}^3 \tilde{\mathrm{A}}^{(i)}
               \dpd{\tilde{\bvec{u}}}{x_i} =
  \sum_{i=1}^3 \tilde{\mathrm{A}}^{(i)}
               \dpd{\tilde{\bvec{u}}}{y} \dpd{y}{x_i} =
  \sum_{i=1}^3 k_{i} \tilde{\mathrm{A}}^{(i)}
               \dpd{\tilde{\bvec{u}}}{y} =
  \tilde{\mathrm{R}}\dpd{\tilde{\bvec{u}}}{y}

The eigenvalues of :math:`\tilde{\mathrm{R}}` can be found by solving the
polynomial equation :math:`\det(\tilde{\mathrm{R}} -
\lambda\mathrm{I}_{4\times4}) = 0` for :math:`\lambda`, and are

.. math::
  :label: bulk.eigval

  \lambda_{1, 2, 3, 4} = \phi, \phi,
                         \phi+\sqrt{\frac{K\psi}{\rho}},
                         \phi-\sqrt{\frac{K\psi}{\rho}}

where :math:`\phi \defeq \sum_{i=1}^3 k_iv_i`, and :math:`\psi \defeq
\sum_{i=1}^3 k_i^2`.  The corresponding eigenvector matrix is

.. math::
  :label: bulk.eigvecmat

  \mathrm{T} = \left(\begin{array}{cccc}
    0 & 0 &
    \sqrt{\frac{\rho^3\psi}{K}} & \sqrt{\frac{\rho^3\psi}{K}} \\
    k_3 &  0   & k_1 & -k_1 \\
    0   &  k_3 & k_2 & -k_2 \\
   -k_1 & -k_2 & k_3 & -k_3
  \end{array}\right)

The left eigenvector matrix is

.. math::
  :label: bulk.eigvecmatright

  \mathrm{T}^{-1} = \left(\begin{array}{cccc}
    0 & -\frac{k_1^2-\psi}{k_3\psi} & -\frac{k_1k_2}{k_3\psi} & -\frac{k_1}{\psi} \\
    0 & -\frac{k_1k_2}{k_3\psi} & -\frac{k_2^2-\psi}{k_3\psi} & -\frac{k_2}{\psi} \\
    \frac{1}{2\sqrt{\frac{\rho^3\psi}{K}}} &
    \frac{k_1}{2\psi} &  \frac{k_2}{2\psi} &  \frac{k_3}{2\psi} \\
    \frac{1}{2\sqrt{\frac{\rho^3\psi}{K}}} &
   -\frac{k_1}{2\psi} & -\frac{k_2}{2\psi} & -\frac{k_3}{2\psi}
  \end{array}\right)

Riemann Invariants
++++++++++++++++++

Aided by Eqs. :eq:`bulk.eigvecmat` and :eq:`bulk.eigvecmatright`,
:math:`\tilde{\mathrm{R}}` can be diagonalized as

.. math::
  :label: bulk.eigvalmat

  \mathrm{\Lambda} \defeq \left(\begin{array}{cccc}
    \lambda_1 & 0 & 0 & 0 \\
    0 & \lambda_2 & 0 & 0 \\
    0 & 0 & \lambda_3 & 0 \\
    0 & 0 & 0 & \lambda_4
  \end{array}\right) = \left(\begin{array}{cccc}
    \phi & 0 & 0 & 0 \\
    0 & \phi & 0 & 0 \\
    0 & 0 & \phi + \sqrt{\frac{K\psi}{\rho}} & 0 \\
    0 & 0 & 0 & \phi - \sqrt{\frac{K\psi}{\rho}}
  \end{array}\right) = \mathrm{T}^{-1}\tilde{\mathrm{R}}\mathrm{T}

Equation :eq:`bulk.eigvalmat` defines the eigenvalue matrix of
:math:`\tilde{\mathrm{R}}`.  Aach element in the diagonal of the eigenvalue
matrix is the eigenvalue listed in Eq. :eq:`bulk.eigval`.  Pre-multiplying Eq.
:eq:`bulk.ge.qlncsv1d` with :math:`\mathrm{T}^{-1}` gives

.. math::

  \mathrm{T}^{-1}\dpd{\tilde{\bvec{u}}}{t}
  + \mathrm{\Lambda}\mathrm{T}^{-1}\dpd{\tilde{\bvec{u}}}{y} = 0

Define the characteristic variables

.. math::
  :label: bulk.chvar

  \hat{\bvec{u}} \defeq \left(\begin{array}{c}
   -\frac{k_1^2-\psi}{k_3\psi}v_1 - \frac{k_1k_2}{k_3\psi}v_2 - \frac{k_1}{\psi}v_3 \\
   -\frac{k_1k_2}{k_3\psi}v_1 - \frac{k_2^2-\psi}{k_3\psi}v_2 - \frac{k_2}{\psi}v_3 \\
   -\sqrt{\frac{K}{\rho\psi}} + \frac{k_1}{2\psi}v_1 + \frac{k_2}{2\psi}v_2 + \frac{k_3}{2\psi}v_3 \\
   -\sqrt{\frac{K}{\rho\psi}} - \frac{k_1}{2\psi}v_1 - \frac{k_2}{2\psi}v_2 - \frac{k_3}{2\psi}v_3
  \end{array}\right)

such that

.. math::

  \mathrm{T}^{-1} = \dpd{\hat{\bvec{u}}}{\tilde{\bvec{u}}}

Then aided by the chain rule, we can write

.. math::
  :label: bulk.ge.char

  \dpd{\hat{\bvec{u}}}{t} + \mathrm{\Lambda}\dpd{\hat{\bvec{u}}}{y} = 0

The components of :math:`\hat{\bvec{u}}` defined in :eq:`bulk.chvar` are the
Riemann invariants.

Diffusion Term Treatment
========================

The momentum equation (Eq. :eq:`bulk.comomentum_eos`) contains the diffusion
term

.. math::

  \sum_{i=1}^3 \dpd{}{x_i}
               \left[\delta_{ij} \lambda \sum_{k=1}^3 \dpd{v_k}{x_k}
                   + \mu \left( \dpd{v_i}{x_j}+\dpd{v_j}{x_i} \right)\right],
               \quad j = 1, 2, 3

Define

.. math::
  :label: bulk.dfuvec

  \bvec{g}^{(1)} \defeq \left(\begin{array}{c}
    0 \\
    \lambda\sum_{k=1}^3\dpd{v_k}{x_k} + 2\mu\dpd{v_1}{x_1} \\
    \mu(\dpd{v_1}{x_2} + \dpd{v_2}{x_1}) \\
    \mu(\dpd{v_1}{x_3} + \dpd{v_3}{x_1})
  \end{array}\right), \quad
  \bvec{g}^{(2)} \defeq \left(\begin{array}{c}
    0 \\
    \mu(\dpd{v_2}{x_1} + \dpd{v_1}{x_2}) \\
    \lambda\sum_{k=1}^3\dpd{v_k}{x_k} + 2\mu\dpd{v_2}{x_2} \\
    \mu(\dpd{v_2}{x_3} + \dpd{v_3}{x_2})
  \end{array}\right), \quad
  \bvec{g}^{(3)} \defeq \left(\begin{array}{c}
    0 \\
    \mu(\dpd{v_3}{x_1} + \dpd{v_1}{x_3}) \\
    \mu(\dpd{v_3}{x_2} + \dpd{v_2}{x_3}) \\
    \lambda\sum_{k=1}^3\dpd{v_k}{x_k} + 2\mu\dpd{v_3}{x_3}
  \end{array}\right)
