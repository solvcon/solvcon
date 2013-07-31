============================
Solvers of Conservation Laws
============================

SOLVCON is a collection of `Python <http://www.python.org>`__-based
conservation-law solvers that use the space-time `Conservation Element and
Solution Element (CESE) method <http://www.grc.nasa.gov/WWW/microbus/>`__.
SOLVCON targets at problems that can be formulated as a system of first-order,
linear or non-linear partial differential equations (PDEs) [Lax73]_:

.. math::
  :label: e:consform

  \dpd{\bvec{u}}{t}
  + \sum_{\iota=1}^3 \mathrm{A}^{(\iota)}(\bvec{u})\dpd{\bvec{u}}{x_{\iota}}
  = \bvec{s}(\bvec{u})

where :math:`\bvec{u}` is the unknown vector, :math:`\mathrm{A}^{(1)}`,
:math:`\mathrm{A}^{(2)}`, and :math:`\mathrm{A}^{(3)}` the Jacobian matrices,
and :math:`\bvec{s}` the source term.

- Get the source from https://bitbucket.org/solvcon/solvcon
- Report bugs and request features at
  https://bitbucket.org/solvcon/solvcon/issues?status=new&status=open
- Ask questions in our `mailing list
  <http://groups.google.com/group/solvcon>`__: solvcon@googlegroups.com

This software uses `BSD license
<http://opensource.org/licenses/BSD-3-Clause>`__.

Documentation
=============

Introduction
++++++++++++

.. toctree::
  :maxdepth: 2

  install
  tutorial
  verification

Applications
++++++++++++

.. toctree::
  :maxdepth: 2

  app_linear

Reference
+++++++++

.. toctree::
  :maxdepth: 3

  architecture
  inout
  system_modules

Control & Management
++++++++++++++++++++

.. toctree::
  :maxdepth: 1

  python_style
  plan

Resources
=========

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

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Release History
===============

.. toctree::
   :maxdepth: 2

   history

.. rubric:: Footnotes

.. [Lax73] Peter D. Lax, *Hyperbolic Systems of Conservation Laws and the
  Mathematical Theory of Shock Waves*, Society for Industrial Mathematics,
  1973.  `ISBN 0898711770
  <http://www.worldcat.org/title/hyperbolic-systems-of-conservation-laws-and-the-mathematical-theory-of-shock-waves/oclc/798365>`__.

.. vim: set spell ft=rst ff=unix fenc=utf8:
