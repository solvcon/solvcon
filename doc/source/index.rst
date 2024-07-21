=======
solvcon
=======

solvcon is a collection of conservation-law solvers that use the space-time
`Conservation Element and Solution Element (CESE) method
<https://yyc.solvcon.net/en/latest/cese/index.html>`__.  The equations to be
solved are formulated as:

.. math::

  \frac{\partial\mathbf{u}}{\partial t}
  + \sum_{k=1}^3 \mathrm{A}^{(k)}(\mathbf{u})
                 \frac{\partial\mathbf{u}}{\partial x_k}
  = 0

where :math:`\mathbf{u}` is the unknown vector and :math:`\mathrm{A}^{(1)}`,
:math:`\mathrm{A}^{(2)}`, and :math:`\mathrm{A}^{(3)}` are the Jacobian
matrices.

The code development has been moved in the repository
https://github.com/solvcon/modmesh.  The code remaining in the old repository
https://github.com/solvcon/solvcon will eventually be migrated.  The old
repository will be updated to include setups for problems and solutions.

.. toctree::
  :maxdepth: 1

  mesh
  nestedloop
  python_style
  history
  copying

.. vim: set spell ft=rst ff=unix fenc=utf8:
