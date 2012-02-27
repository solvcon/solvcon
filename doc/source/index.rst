=============================
SOLVCON -- SOLVer CONstructor
=============================

Supportive functionalities, e.g., mesh loading, result writing, visualizing,
etc., are usually the tedious and error-prone part of coding up a PDE solver.
It takes a lot of efforts to develop the functionalities, and more efforts to
maintain them.  As the result, compared to the supportive functionalities, the
lines of code written for the core numerical methods of a PDE solver are fairly
few.  The productivity of PDE-solver developers will be rocket-boosted is they
do not need to worry about the supportive functionalities.

Unfortunately, no matter how costly developing the supportive functionalities
is, it cannot be avoided.  A PDE solver without a mesh loader or a mesh
generator is not applicable at all.  PDE solvers need supportive
functionalities to deliver results.

To resolve this dilemmatic issue, we designed `SOLVCON <http://solvcon.net/>`_
to host supportive functionalities and to provide a `software framework
<http://en.wikipedia.org/wiki/Software_framework>`__ to develop
high-performance, massively parallelized PDE solvers.  Generally speaking, PDE
solvers are computer programs consisting of two levels of loops: The outer loop
and the inner loops.  Computer code of both the supportive functionalities and
the numerical methods can be wrapped around the fundamental two-loop structure.
As a result, supportive functionalities can be segregated from the core
numerical algorithms.  The reusability gained by using SOLVCON can
significantly save the efforts of the development of PDE solvers.

This is especially true for solvers of hyperbolic PDEs or conservation laws,
i.e., for solving systems of first-order PDEs in the quasi-linear form:

.. math::

  \dpd{\bvec{u}}{t}
  + \sum_{\iota=1}^3 \mathrm{A}^{(\iota)}(\bvec{u})\dpd{\bvec{u}}{x_{\iota}}
  = \bvec{s}(\bvec{u}).

In the context of numerical solutions of conservation laws, the outer loop is
used to perform time-marching, and is usually called the *temporal loop*.
Within the outer temporal loop, the inner loops are used to sweep over the
discretized spatial domain.  Therefore, the inner loops are called the *spatial
loops*.  While there is only one outer temporal loop, there usually are many
inner spatial loops to perform different numerical calculations.

Using SOLVCON calls for the fundamental understanding of the basic, macroscopic
two-loop structure of PDE solvers.  The fundamental structure makes no
assumption for the computer architecture nor the numerical method employed.
The macroscopic abstraction allows the developed PDE solvers to be as
high-performance as possible, at the price of being unable to use some of the
built-in functionalities.

Contents
========

.. toctree::
   :maxdepth: 2

   install
   tutorial

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

History
=======

.. include:: ../../HISTORY.rst
.. vim: set ft=rst ff=unix fenc=utf8:
