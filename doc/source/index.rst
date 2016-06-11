=======
SOLVCON
=======

SOLVCON is a collection of conservation-law solvers that use the space-time
`Conservation Element and Solution Element (CESE) method
<http://www.grc.nasa.gov/WWW/microbus/>`__ [Chang95]_.  The equations to be
solved are formulated as:

.. math::

  \frac{\partial\mathbf{u}}{\partial t}
  + \sum_{k=1}^3 \mathrm{A}^{(k)}(\mathbf{u})
                 \frac{\partial\mathbf{u}}{\partial x_k}
  = 0

where :math:`\mathbf{u}` is the unknown vector and :math:`\mathrm{A}^{(1)}`,
:math:`\mathrm{A}^{(2)}`, and :math:`\mathrm{A}^{(3)}` are the Jacobian
matrices.

.. include:: ../../README.rst
  :start-line: 14

Documents
=========

.. toctree::
  :maxdepth: 2

  mesh
  nestedloop

Development
===========

- https://github.com/solvcon/solvcon/issues (issue tracker)
- solvcon@googlegroups.com (mailing list; its web interface:
  http://groups.google.com/group/solvcon)
- :doc:`python_style`
- :doc:`hidden_infrastructure`
- :doc:`hidden_applications`

References
==========

- :doc:`bibliography`
- :doc:`history`
- :doc:`copying`
- :doc:`link`
- :doc:`link_other`

.. vim: set spell ft=rst ff=unix fenc=utf8:
