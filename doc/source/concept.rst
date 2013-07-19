:orphan:

=======================
Concepts behind SOLVCON
=======================

SOLVCON_ is a software framework.  What does that mean?

.. _SOLVCON: http://solvcon.net/

A software framework dictates how you can write a software program.  Since
SOLVCON_ is targeting to simulation codes, it guides, or requires, programmers
to write the scientific codes in a specific way.  Therefore, when using 
SOLVCON_, you can say that we lose some of our freedom in coding.

.. note:: SOLVCON_ is designed for scientific codes that solve PDEs in a
   time-accurate way, and uses the unstructured mesh.  When I mention
   *scientific codes* or *simulation codes* later, I mean the specific category
   of codes.

Why doing so?  It's all about **supportive tasks**.  There are always
supportive tasks to be done before one can actually make a scientific code to 
work.  For the solving methods with a grid, such as `FDM
<http://en.wikipedia.org/wiki/Finite_difference_method>`_, `FDTD
<http://en.wikipedia.org/wiki/FDTD>`_, `FEM
<http://en.wikipedia.org/wiki/Finite_element_method>`_, `FVM
<http://en.wikipedia.org/wiki/Finite_volume_method>`_, and `CESE
<http://www.grc.nasa.gov/WWW/microbus/>`_, the spatial domain is discretized
into small pieces (the mesh), and various boundary conditions are applied to
the boundaries.  In order to use the numerical method to solve the problem
described by the PDEs in the spatial domain, you have to:

1. Construct and/or load the mesh into memory for your solver.

2. Build the mapping between the boundaries to the boundary conditions.

3. Initialize the computing domain with certain initial conditions, before you
   can march the solver.  Often, the conditions are not trivial.

4. Post-process or store the results.  Usually, you want to do it after or at
   the time of solving.
  
5. Parallelize the code using domain-decomposition technique.  Not rarely, a
   serial code is not fast enough, or just can't model big problems.

Your precious time has to be spent in these merely supportive, but non-trivial
tasks, because you can not function the numerical algorithm before finishing
them.  A lot of scientific codes have a significant portion, usually more than
half, and sometimes 80%, dedicated to these supportive functionalities.

The supportive coding is important, but not so important as the numerical
algorithms or the physical problems, for computational physicists.  We
definitely want to concentrate our brain cells in the problems, rather than in
the off-topic coding.  However, these supportive tasks not only occupy our
working hours, but also the way people implement them is usually inefficient
and hard to maintain.  A lot of the implementations involve low-speed
intermediate temporary files and lengthy ad-hoc shell scripts.  The non-ideal
practices make our codes to be less useful.  For example, it's easy to find
codes can run only on the environment on which they are developed, or the
post-processing portion takes longer time to finish than solving.

There should be a better way to finish the tedious tasks.  SOLVCON_ is designed
to free you from these headaches, for the specific category of scientific
codes.  It saves you a lot of coding, and provides a lot of functionalities to
your code.  Those are what we trade some of our freedom in coding for.

Stereotyped Simulation
======================

I start the discussion of the concepts of SOLVCON_ from what a simulation code
usually looks like.

A scientific code usually has a kernel, which implements the numerical method
as the solving algorithm.  The solving kernel is the inner-most part of the
code, and it iterates over all the spatial elements (cells).  To solve the PDEs
time-accurately, you have to time-march the solving kernel.  There will be a
loop (called marching loop, time loop or temporal loop) that do the
time-marching.

The cell loop (or called space loop or spatial loop) and the marching loop are
the main structure of the simulation code.  They are the most essential parts
to a simulation.  All other things are designed to support these two loops.

Before one can loop over the cells, the structure of the discretized spatial
domain has to be know.  At this point, the code need to load or create the mesh
that describe the domain, where the governing equations are applied to.

All PDEs need initial conditions and boundary conditions and then can be
solved.  Usually the PDEs describe fields, so that the initial conditions are
nothing more then how we initialize the solution variables.  The boundary
conditions are more complicated, since they have to be applied time to time in
the marching loop.

Mathematically, with enough conditions, the PDEs can be solved.  Numerically,
the problem can be solved only after a right logic to execute the solver.  With
all the defined loops, mesh, initial conditions, and boundary conditions, there
needs a wrapping logic to conduct the simulation in a right way.  Also in the
wrapping logic, we tell the code when and how to do the post-processing, or the
preparation of the post-processing, so that the solution can be analyzed.

Hence, we know that the following components, if they can be decoupled, compose
the simulation code:

* **Mesh**.  It defines the computing domain, including the connectivity and 
  geometry of fundamental elements.

* **Solving kernel**.  It implements the cell loop, and is the essential part 
  of the solving numerical algorithm.

* **Boundary-condition treatments**.  It couples with the solving kernel in a 
  certain way.

* **Logics for initial conditions**.  It initialize the fields to be simulated.

* **Conductor for the simulation**.  The marching loop should be implemented in
  this part, and other supportive tasks such as input, output, and just-in-time
  post-processing are connected to the marching loop.

Of course, if one wants to write parallel code, there are more things to do,
but the structure is pretty much the same as the described.

Model of SOLVCON
================

SOLVCON_ abstracts the common structure that almost all the simulation codes
share, and makes it a framework.  That is, instead of construct the must-have
skeleton by your own, you get it from SOLVCON_.  All you need is to know where
to put what code in.

For example, there are more than 7,000 lines of well-organized code in
SOLVCON_, and you need to write less than 2,000 lines of code for one- and
two-dimensional stress wave simulation and corresponding post-processing.  Then
you enjoy automatically available features like parallelization, mixed-typed
unstructured mesh, and input and output components.

The essential part of SOLVCON_ is the mesh definition, rather than anything
else such as the solving algorithm.  It makes good sense for people having
experience in implementing their own solver.  The most time-consuming part of
the implementation could be to figure out how to correctly manipulate the
unstructured connectivities for various types of fundamental elements.  Only
after one has a right definition of the spatial domain, the numerical algorithm
can be correctly tested.  SOLVCON_ defines the connectivities for seven
different elements in two- and three-dimensional space, including triangles,
quadrilaterals, tetrahedra, prisms, pyramids, and hexahedra.  All the
fundamental elements can be mixed up (in the same rank of spatial domain, that
is, you can not mix elements in two-dimensional space with others in
three-dimensional space).  The mesh is defined in :class:`solvcon.block.Block`.

For the real solving work, SOLVCON_ puts the cell loop and the marching loop in
:mod:`solvcon.solver` and :mod:`solvcon.case`, respectively.  The two
sub-packages contains definitions for one- and multi-dimensional solvers and
cases.  The solvers care only for the cell loop, while the cases need to
connect the marching loop with various supportive tasks.  The
boundary-condition treatments are invoked from the solvers, since they are
usually applied to the solution in each time-marching.  All the boundary
conditions are subclasses of :class:`solvcon.boundcond.BC`.

Because most of the simulation codes pursue the utmost performance, the cell
loop is usually implemented with FORTRAN or C/C++, and called from a wrapper
method in the Python solver classes.

The cases (defined in :mod:`solvcon.case`) are versatile, since they not only
define the marching loop, but also conduct how the simulation should run.  A
case (instance) contains all the needed information about a simulation.  These
information includes, for example, the mesh to be loaded, the solver to be
used, the boundary-condition mappings, the related parameters, and many others.

In order to assist the cases to manage various supportive of specific pre- and
post-processing tasks, a family of classes based on :class:`Hook` are defined
also in :mod:`solvcon.case`.  The programmer can implement optional features as
hooks, so that they can be plug-and-play to the simulation cases.  For example,
there is :class:`solvcon.case.core.ProgressHook` pre-defined, which reports the
progress of a simulation case to the terminal.  There are also other
pre-defined hooks you can use out-of-box.

In order to perform a simulation and make the result analyzable, usually the
programmer need to implement two different customized hooks of
:class:`Initializer` and :class:`Calculator`, for initializing the solution
fields and post-processing, respectively.

Since the internal of the one- and multi-dimensional meshes are very different,
the solver, cases, and hooks class hierarchies contain both one- and
multi-dimensional versions.  There are some shared features across the two
categories.  Sometimes there are general hooks can be applied to both kinds of
cases, such as :class:`solvcon.case.core.ProgressHook`.  However, usually you
have to implement the logics for either categories, since they are so
different.

Hierarchical Structure
======================

The following list roughly demonstrates the structure of SOLVCON_:

* :mod:`solvcon` -- The top-level namespace.

  * :mod:`block` -- Definition of the multi-dimensional unstructured mesh.

  * :mod:`solver` -- Framework to implement cell loop for the solving kernel.

  * :mod:`case` -- Simulation case definition and the hook framework.

  * :mod:`boundcond` -- Framework for boundary-condition treatments.

  * :mod:`dependency` -- Helpers to load and use external dynamically linked
    libraries.

  * :mod:`io` -- Input and output facilities.

  * :mod:`helper` -- Miscellaneous helper facilities.

  * :mod:`rpc` -- Inter-process communication and remote procedure call.

  * :mod:`domain` -- Domain-decomposition logic.

  * :mod:`conf` -- Configuration information for the runtime.

  * :mod:`gendata` -- Some internal generic data structure.

Below the top-level namespace, there are more than ten sub-packages or
sub-modules within the top package :mod:`solvcon`.  The modules listed are
ordered by how much a programmer needs to know about them.

The programmer should be very familiar with the first 4 modules: :mod:`block`,
:mod:`solver`, :mod:`case`, and :mod:`boundcond`, because they define the main 
structure of the simulation code.  You should understand the APIs in these 4
modules, and then subclass the base classes.

The next 3 modules: :mod:`dependency`, :mod:`io`, and :mod:`helper` should also
be useful in your program, since they are the utility modules.  Next, modules
:mod:`rpc` and :mod:`domain` are for parallelization through domain
decomposition.  The rest modules :mod:`conf` and :mod:`gendata` are mostly used
internally and usually you don't need to touch them.

How to Organize Your Simulation Code
====================================

SOLVCON_ is a framework, not the solver itself.  That is, SOLVCON_ is a tool or
a library that helps you to create your solving code.  Since SOLVCON_ is
written (mostly) in the Python programming language, your code will be a Python
program as well.

Justification for Python
++++++++++++++++++++++++

There are a lot of advantages to build the simulation code using a high-level
language such as Python.  One big advantage of using Python as the "driver" of
your simulation is that, you don't need to design an input file anymore!
Because of the scripting ability of Python, you don't need to "compile" Python
code into the executable form before running.  You run it on the fly.  The
source itself can be fed to the Python runtime (VM) and runs.  That is, the
simulation code itself acts as the input file.  Whenever you want to change any
of the parameter, you can directly make the change and run.  No compilation is
needed.

This is not to say you write everything in Python in the simulation code.
Python is a dynamic language, and by the nature it is way too slow for
implementing the numerical algorithm that hogs computing power.  Python is just
unsuitable to "squeeze" all the performance out of the hardware.  In order to
gain the wanted performance, the solving kernel usually has to be implement in
FORTRAN.

.. note:: If you don't have experience or preference in any of the number
   crunching languages, I would like to suggest you to start coding the
   number-crunching part in FORTRAN 90/95.  Not FORTRAN 77, C, nor C++,
   although any language should be fine.  For some people it sounds weird, but
   FORTRAN provides really good facilities for implementing numerical
   algorithms involving spatial meshes.

   When you program in FORTRAN 90/95 with Python, it is good to stay away from 
   the fancy module things provided by the language.  Usually you don't need it 
   when used with Python.  Avoiding them can save you from a lot of headaches.

   If you want to write a code that makes use of special hardware such as GPUs,
   FORTRAN might not be ideal.  There are different considerations.

Usually a dynamic language such as Python is not considered to be used in
implementation of a scientific code, just because it's not fast enough.
However, the languages suit number crunching are too primitive to write an
easy-to-use and flexible framework for general problems or physical models.  For
the balance, to mix Python with another number crunching programming language,
usually FORTRAN, is a reasonable take, and the result turns out to be very
good.  Codes developed using SOLVCON_ are just as fast as their pure-FORTRAN
counterpart, and sometimes even faster.

Big Picture
+++++++++++

There is an entry point for every program on the earth.  The entry point for
the simulation code using SOLVCON_ would be a **driving script** written in
Python.  The driving script dictates how to run the simulation code, and is
responsible for all the setting-up and finalizing things.

A simple driving script would look like this:

.. highlight:: python
   :linenothreshold: 5

::

   from tolkien import case as casemd
   @casemd.TolkienCase.register_simulation
   def cstest(casename=None, meshname='middleearth_22k.neu.gz',
       th=0.0, ph=0.0, mtrlname='Dust', core_width=0.02, valin=1.0,
       time_increment=7.5e-7, nsteps=20, psteps=1, ssteps=10,
       **kw
   ):
       from solvcon.boundcond import bctregy
       bcmap = {
           'left': (bctregy.TolkienNonrefl, {}),
           'right': (bctregy.TolkienNonrefl, {}),
       }
       case = casemd.TolkienCase(basedir=basedir, basefn=casename,
           fn_neu=meshname, bcmap=bcmap, steps_run=nsteps,
           time_increment=time_increment,
           **kw
       )
       case.execution.runhooks.append(casemd.Init(case,
           core_width=core_width, valin=valin, mtrlname=mtrlname, th=th, ph=ph,
       ))
       case.execution.runhooks.append(casemd.Calc(case, psteps=ssteps))
       return case
   def main():
       import sys
       from solvcon.case.core import simulations
       simulations[sys.argv[1]](submit=False)
   if __name__ == '__main__':
       main()

The function :func:`cstest` is called an **arrangement** in the code, because
it arranges a simulation case, and finally returns the case object which is set
up.  The decorator in the second line will push the arrangement into a
dictionary-like registry singleton located at
:const:`solvcon.case.core.simulations`.

Once an arrangement is registered, you can access it from the registry, as the
driver script does in line 26.  There will be a generated wrapper which is
responsible for calling the relevant methods of the case to initialize and run
the case itself.

The driving script import a package named :mod:`tolkien`, and it is the place
which you should put your definition of simulation case classes and solver in.
It usually has a structure similar to:

* :mod:`tolkien` -- Top-level namespace.

  * :mod:`solver` -- Define the solver for the physical model by subclassing a
    base solver class in :mod:`solvcon.solver`.

  * :mod:`boundcond` --  Define the corresponding boundary conditions.

  * :mod:`case` -- Define the customized simulation case by subclassing a base
    case class in :mod:`solvcon.case`.

Also, there is usually code written in FORTRAN to serve as the kernel of the
solver classes in the module :mod:`tolkien.solver`.  You can use SCons_ to
build the FORTRAN code into a dynamically-linked library and load it by the aid
of :mod:`solvcon.dependency`.

.. _SCons: http://www.scons.org/

Pros and Cons
+++++++++++++

To organize a simulation code in this way gives us very high flexibility to
manage the simulation code.  You can reuse all the code in your simulation
package (in the previous case, it's :mod:`tolkien`).  Since the driving script
is a Python script, you have full control over it to do anything.  If you
really want a traditional input file, nothing stops you.

You can use the way how :mod:`solvcon` is structured to organize your
simulation code.  It makes good sense since you are using SOLVCON_.  However,
it's also OK for you to take other way to organize your code.  The only thing
you need to do is to make use of SOLVCON_.

No matter how you use SOLVCON_, it would save you from a lot of coding for the
features it provides.

Although SOLVCON_ is convenient, it does impose limitation to how you can write
your code.  The entry point has to be a Python script.  Sometimes it's
cumbersome, but usually there are workarounds.  SOLVCON_ would be a good tool,
but it's not a free lunch.

.. vim: set ft=rst ff=unix fenc=utf8 ai:
