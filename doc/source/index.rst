=======
SOLVCON
=======

SOLVCON is a collection of `Python <http://www.python.org>`__-based
conservation-law solvers that use the space-time `Conservation Element and
Solution Element (CESE) method <http://www.grc.nasa.gov/WWW/microbus/>`__
[Chang95]_.  It targets at solving problems that can be formulated as a system
of first-order, linear or non-linear partial differential equations (PDEs)
[Lax73]_:

.. math::

  \dpd{\bvec{u}}{t}
  + \sum_{k=1}^3 \mathrm{A}^{(k)}(\bvec{u})\dpd{\bvec{u}}{x_k}
  = \bvec{s}(\bvec{u})

where :math:`\bvec{u}` is the unknown vector, :math:`\mathrm{A}^{(1)}`,
:math:`\mathrm{A}^{(2)}`, and :math:`\mathrm{A}^{(3)}` the Jacobian matrices,
and :math:`\bvec{s}` the source term.

.. include:: ../../README.rst
  :start-line: 10

Focus Applications
==================

.. toctree::
  :maxdepth: 2

  gas/index

Other Applications
==================

.. toctree::
  :maxdepth: 1

  app_linear
  bulk/index
  app_vewave

Infrastructure
==============

.. toctree::
  :maxdepth: 3

  architecture
  inout

Development Support
===================

- Issue tracker: https://bitbucket.org/solvcon/solvcon/issues
- Users' mailing list: solvcon@googlegroups.com (or
  http://groups.google.com/group/solvcon)
- :doc:`python_style`
- :doc:`jenkins`

References
==========

- :doc:`bibliography`
- :doc:`history`
- Papers and presentations:

  - :doc:`pub_app`
  - Yung-Yu Chen,
    *A Multi-Physics Software Framework on Hybrid Parallel Computing for
    High-Fidelity Solutions of Conservation Laws*,
    Ph.D. Thesis, The Ohio State University, United States, Aug. 2011.
    (`OhioLINK <http://rave.ohiolink.edu/etdc/view?acc_num=osu1313000975>`__)
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

.. rubric:: Copyright Notice

.. include:: ../../COPYING

.. rubric:: Contributors

.. include:: ../../CREDITS.rst
  :start-line: 3

.. vim: set spell ft=rst ff=unix fenc=utf8:
