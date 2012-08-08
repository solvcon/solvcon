========
Tutorial
========

SOLVCON is a Python library for developing high-performance,
massively-parallelized PDE solvers.  By default, SOLVCON provides a series of
solvers that use the space-time `Conservation Element and Solution Element
(CESE) <http://www.grc.nasa.gov/WWW/microbus/>`__ method.  Problems of
compressible flows and stress waves in solids can be solved by using the
stocked solvers.

The goal of SOLVCON is to help code developers to focus on the numerical
algorithms.  These computing cores can be written in C or any high-speed
language (Fortran, CUDA, C++, etc.; you name it) and interfaced with SOLVCON.
SOLVCON has a general work flow that includes things like mesh loaders (`Gmsh
<http://www.geuz.org/gmsh/>`__, FLUENT Gambit (R), and `CUBIT
<http://cubit.sandia.gov/>`__), MPI, and VTK.  Users of SOLVCON can just take
the supportive functionalities and jump into the physics and numerics.

For solving for conservation laws and most PDEs, the computer codes usually
contains two levels of loops.  An outer loop is used to perform time-marching,
and is usually called the *temporal loop*.  Within the outer temporal loop,
there are multiple inner loops to sweep over the discretized spatial domain.
The inner loops are called the *spatial loops*.  This is the well-known
*two-loop structure* of PDE solvers and is absorbed into the SOLVCON work flow.
`Inversion of control (IoC)
<http://en.wikipedia.org/wiki/Inversion_of_control>`__ is used to expose the
work flow to the code developers.

The key functionalities of SOLVCON will be introduced in this document.

Set up the Environment
======================

Assume:

- SOLVCON is compiled without problems.  See the document of :doc:`install` for
  more information.
- The compiled SOLVCON is located at ``$SCSRC``.
- You are using bash.

The following setting will enables SOLVCON::

  export PYTHONPATH=$SCSRC:$PYTHONPATH

And then the following command::

  python -c "import solvcon; print solvcon"

should show you where SOLVCON is imported from.

There are various examples located at ``$SCSRC/examples``.  To follow the
examples, you need to:

- Install `ParaView <http://www.paraview.org/>`__.  On a Debian/Ubuntu, you can
  do it by executing::

    sudo apt-get install paraview

- Obtain example data.  You can do it by executing::

    scons --get-scdata

  in ``$SCSRC``.

Configuration
=============

SOLVCON will find each of the solvcon.ini files from current working directory
toward the root, and use their settings.  Three settings are recognized in
``[SOLVCON]`` section:

- ``APPS``: Equivelent to the environment variable ``SOLVCON_APPS``.
- ``LOGFILE``: Equivelent to the environment variable ``SOLVCON_LOGFILE``.
- ``PROJECT_DIR``: Equivelent to the environment variable
  ``SOLVCON_PROJECT_DIR``.  Can be set to empty, which indicates the path where
  the configuration file locates.

The configurable environment variables:

- ``SOLVCON_PROJECT_DIR``: the directory holds the applications.
- ``SOLVCON_LOGFILE``: filename for solvcon logfile.
- ``SOLVCON_APPS``: names of the available applications, seperated with
  semi-colon.  There should be no spaces.
- ``SOLVCON_FPDTYPE``: a string for the numpy dtype object for floating-point.
  The default fpdtype to be used is float64 (double).
- ``SOLVCON_INTDTYPE``: a string for the numpy dtype object for integer.  The
  default intdtype to be used is int32.
- ``SOLVCON_MPI``: flag to use MPI.

Mesh Generation (to Be Added)
=============================

Before solving any PDE, you need to define the discretized spatial domain of
the problem by generating the mesh.

Driving Scripts (to Be Corrected)
=================================

The simplest example: located at ``$SCSRC/examples/euler/hbnt/``.

Objective:

#. Understand the concept of "driving script" (programmable input file).
#. Perform simulation with SOLVCON.

Course:

#. Run the code::

     $ PYTHONPATH=../../.. ./go run

#. Simulation results are stored in the sub-directory ``result/``.  Use
   ParaView to load the VTK files.
#. Code organization:

   - First line: indicating it is a Python script.
   - Second line: indicating encoding of the file.
   - Line 4-18: comments for copyright information.
   - Line 20-27: docstring in the form of `string literal
     <http://docs.python.org/reference/lexical_analysis.html#string-literals>`_.
   - Line 29: module-level import for arrangement decorator.
   - Line 31-99: main body for instantiating the ``Case`` object in the form
     of a Python function/callable; "creation function".
   - Line 101-112: `decorated
     <http://en.wikipedia.org/wiki/Decorator_pattern>`_ arrangement
     (function).
   - Line 114-116: invoking command-line interface of SOLVCON.
#. Customization goes into the creation function:

   - Specify BC: line 54-59.
   - Feed parameter to Case: line 60-64.
#. SOLVCON modules to hack:

   - ``solvcon.boundcond``
   - ``solvcon.case``
   - ``solvcon.solver``

The Hook System (to Be Corrected)
=================================

Located at ``$SCSRC/examples/euler/obrefl/``.

Objective:

- Use the programmability of input file for properties specification.
- Understand the Hook system for custom post-processing.

Question:

- Where is the creation function?

Course:

#. Run and inspect the simulation.
#. Change the flow properties in line 263-275 and see the difference.

   - Utility code is organized as a class in line 52-164, for calculating shock
     properties.
#. How to extend SOLVCON by using Hook, i.e., line 166-244, 318-320.
#. SOLVCON modules to hack:

   - ``solvcon.hook``
   - ``solvcon.kerpak.euler``

Change Physical Model (to Be Corrected)
=======================================

Located at ``$SCSRC/examples/elastic/grpv/``.

Objective:

- Change the physical model.
- Understand the Anchor system for parallel processing.

Questions:

#. What is the path of the mesh file used in this simulation?
#. What is the equivalent code of line 123-125 in the previous two examples?

Course:

#. Run and inspect the simulation.
#. Note the difference of line 144.  It uses a different calculator to the
   Euler solver.
#. Line 76-89, 135-142 uses the Anchor system to insert source term.
#. Line 35-74 calculate the source value.
#. SOLVCON modules to hack:

   - ``solvcon.anchor``
   - ``solvcon.kerpak.elastic``

Output Control (to Be Corrected)
================================

Located at ``$SCSRC/examples/visout/pvtk/``.

Objective:

- Parallel run.
- Specify the variables to output.

Questions:

#. Guess what problem is it?
#. Where is the code for sequential VTK output in legacy format?

Course:

#. Run the simulation in parallel by following the docstring.
#. Inspect the solution.
#. Line 90-102 specifies three kinds of quantities:

   - Negative integer for array.
   - Zero for scalar.
   - Positive value for vector.
#. Try to turn off some of the variables by commenting out.

   - Before rerun the simulation, clean the ``result/`` directory.
#. SOLVCON sub-package to hack:

   - ``solvcon.io``
