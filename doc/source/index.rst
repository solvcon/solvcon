============================
Solvers of Conservation Laws
============================

SOLVCON is a collection of `Python <http://www.python.org>`__-based
conservation-law solvers that use the space-time `Conservation Element and
Solution Element (CESE) method <http://www.grc.nasa.gov/WWW/microbus/>`__
[Chang95]_.  SOLVCON targets at solving problems that can be formulated as a
system of first-order, linear or non-linear partial differential equations
(PDEs) [Lax73]_:

.. math::

  \dpd{\bvec{u}}{t}
  + \sum_{k=1}^3 \mathrm{A}^{(k)}(\bvec{u})\dpd{\bvec{u}}{x_k}
  = \bvec{s}(\bvec{u})

where :math:`\bvec{u}` is the unknown vector, :math:`\mathrm{A}^{(1)}`,
:math:`\mathrm{A}^{(2)}`, and :math:`\mathrm{A}^{(3)}` the Jacobian matrices,
and :math:`\bvec{s}` the source term.  SOLVCON is designed to be a software
framework to house various solvers.  The design of SOLVCON is discussed in
[Chen11]_.

- The project page https://bitbucket.org/solvcon/solvcon
- Report bugs and request features at
  https://bitbucket.org/solvcon/solvcon/issues?status=new&status=open
- Ask questions in our `mailing list
  <http://groups.google.com/group/solvcon>`__: solvcon@googlegroups.com

.. include:: ../../README.rst
  :start-line: 10

Documentation
=============

.. toctree::
  :maxdepth: 2

  tutorial

Applications
++++++++++++

.. toctree::
  :maxdepth: 2

  app_linear
  gas/index
  bulk/index
  app_vewave

Numerical Methods
+++++++++++++++++

.. toctree::
  :maxdepth: 3

  tdnum/index
  cese

Infrastructure
++++++++++++++

.. toctree::
  :maxdepth: 3

  architecture
  inout
  system_modules

Development Support
+++++++++++++++++++

.. toctree::
  :maxdepth: 1

  python_style
  verification

Appendices
==========

Other Resources
+++++++++++++++

- Papers and presentations:

  - :doc:`pub_app`
  - `PyCon US 2011 talk
    <http://us.pycon.org/2011/schedule/presentations/50/>`__: `slides
    <http://solvcon.net/slide/PyCon11_yyc.pdf>`__ and `video
    <http://pycon.blip.tv/file/4882902/>`__
  - Yung-Yu Chen, David Bilyeu, Lixiang Yang, and Sheng-Tao John Yu,
    "SOLVCON: A Python-Based CFD Software Framework for Hybrid
    Parallelization",
    *49th AIAA Aerospace Sciences Meeting*,
    January 4-7 2011, Orlando, Florida.
    `AIAA Paper 2011-1065
    <http://pdf.aiaa.org/preview/2011/CDReadyMASM11_2388/PV2011_1065.pdf>`_
- The CESE method:

  - The CE/SE working group: http://www.grc.nasa.gov/WWW/microbus/
  - The CESE research group at OSU: http://cfd.solvcon.net/research.html
  - Selected papers:

    - Sin-Chung Chang, "The Method of Space-Time Conservation Element and
      Solution Element -- A New Approach for Solving the Navier-Stokes and
      Euler Equations", *Journal of Computational Physics*, Volume 119, Issue
      2, July 1995, Pages 295-324.  `doi: 10.1006/jcph.1995.1137
      <http://dx.doi.org/10.1006/jcph.1995.1137>`_
    - Xiao-Yen Wang, Sin-Chung Chang, "A 2D Non-Splitting Unstructured
      Triangular Mesh Euler Solver Based on the Space-Time Conservation Element
      and Solution Element Method", *Computational Fluid Dynamics Journal*,
      Volume 8, Issue 2, 1999, Pages 309-325.
    - Zeng-Chan Zhang, S. T. John Yu, Sin-Chung Chang, "A Space-Time
      Conservation Element and Solution Element Method for Solving the Two- and
      Three-Dimensional Unsteady Euler Equations Using Quadrilateral and
      Hexahedral Meshes", *Journal of Computational Physics*, Volume 175, Issue
      1, Jan. 2002, Pages 168-199.  `doi: 10.1006/jcph.2001.6934
      <http://dx.doi.org/10.1006/jcph.2001.6934>`_
- :doc:`link`
- :doc:`link_other`

Copyright Notice
++++++++++++++++

.. include:: ../../COPYING

Release History
+++++++++++++++

.. toctree::
   :maxdepth: 2

   history

Contributors
++++++++++++

.. include:: ../../CREDITS.rst
  :start-line: 3

Bibliography
++++++++++++

.. include:: bibliography.rst_inc
  :start-line: 3

Indices and Tables
++++++++++++++++++

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. vim: set spell ft=rst ff=unix fenc=utf8:
