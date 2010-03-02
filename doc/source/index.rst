=====================
Solvcon documentation
=====================

Solvcon_ is a framework to write explicit and time-accurate simulation codes
for PDEs with the unstructured mesh.  As a framework, Solvcon_ provides:

1. A data structure for two- and three-dimensional mixed-type unstructured
   meshes.

2. Unstructured mesh importers (currently Fluent Gambit (tm) neutral file
   format is implemented).

3. Simulation data writers in VTK_ legacy format.

4. An organized and flexible system to write pre- and post-processing codes
   (the **Hooks**).

5. Predefined and automated domain-decomposition logic.

6. IPC (Inter-Process Communication) and RPC (Remote Procedure Call).  

.. _Solvcon: http://cfd.eng.ohio-state.edu/~yungyuc/solvcon/
.. _VTK: http://www.vtk.org/

.. note:: Solvcon_ is not a framework that applies to all kinds of scientific
   code, but it's general enough to help programs which fit in the category.
   You can use Solvcon_ for **Computational Fluid Dynamics** (CFD),
   **Computational Mechanics**, **Computational Electromagnetics**, or any
   other fields that solve PDEs.

By using Solvcon_, you are able to concentrate in implementing the essential
numerical algorithm in one-, two- or three-dimensional space.  You don't need
to worry about how to parse and load an unstructured mesh, where to specify the
initial conditions in your code, or how to implement the boundary conditions
for your solver.  Solvcon_ provides the guidelines for all the components that
a simulation code for PDEs needed.

If you don't even know anything about or have no experience in implementing a
scientific code, then Solvcon_ is a good resource for you to get start.

Contents
========

.. toctree::
   :maxdepth: 2

   install
   concept

Get Solvcon
===========

Solvcon_ hasn't been released.  Ask the developer (`Yung-Yu Chen`_) for detail.

.. _Yung-Yu Chen: mailto: yyc@seety.org

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. vim: set ft=rst ff=unix fenc=utf8:
