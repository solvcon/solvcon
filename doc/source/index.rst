=====================
SOLVCON documentation
=====================

SOLVCON_ is a framework to write explicit and time-accurate simulation codes
for PDEs with the unstructured mesh.  As a framework, SOLVCON_ provides:

#. A data structure for two- and three-dimensional unstructured mesh of mixed
   elements.

#. Unstructured mesh importers.

#. Simulation data writers in VTK_ legacy format.

#. An organized and flexible system to write pre- and post-processing codes
   (the **Hooks**).

#. Predefined and automated domain-decomposition logic.

#. IPC (Inter-Process Communication) and RPC (Remote Procedure Call).  

.. _SOLVCON: http://solvcon.net/
.. _VTK: http://www.vtk.org/

.. note:: SOLVCON_ is not a framework that applies to all kinds of scientific
   code, but it's general enough to help programs which fit in the category.
   You can use SOLVCON_ for **Computational Fluid Dynamics** (CFD),
   **Computational Mechanics**, **Computational Electromagnetics**, or any
   other fields that solve PDEs.

By using SOLVCON_, you are able to concentrate in implementing the essential
numerical algorithm in one-, two- or three-dimensional space.  You don't need
to worry about how to parse and load an unstructured mesh, where to specify the
initial conditions in your code, or how to implement the boundary conditions
for your solver.  SOLVCON_ provides the guidelines for all the components that
a simulation code for PDEs needed.

If you don't even know anything about or have no experience in implementing a
scientific code, then SOLVCON_ is a good resource for you to get start.

Contents
========

.. toctree::
   :maxdepth: 2

   install
   concept

Get Solvcon
===========

SOLVCON_ hasn't been released.  Ask the author (`Yung-Yu Chen`_) for detail.

.. _Yung-Yu Chen: mailto: yyc@seety.org

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. vim: set ft=rst ff=unix fenc=utf8:
