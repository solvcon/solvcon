:orphan:

===============================
Other Software for Solving PDEs
===============================

There has been a number of great projects dedicated to solving general partial
differential equations.  You may want to check them out when evaluating whether
SOLVCON is the right tool for you.  This is not a comprehensive list of
existing software.

- FEniCS_/DOLFIN: a PDE-solving tool writtin in C++ with a Python_ interface,
  developed at Simula_ Research Laboratory.  FEniCS/DOLFIN is based on finite
  element method (FEM).
- FiPy_: a PDE solver written in Python_ at National Institute of Standards and
  Technology (NIST_).  FiPy is based on projction method with finite volume
  (FV) formulation.
- hpFEM/Hermes_: a C++ library for FEM and hp-FEM/hp-DG solvers with
  hp-adaptive algorithms, developed at University of Nevada, Reno.
- hpGEM_: a C++ software framework for discontinuous Galerkin (DG) method
  developed at `University of Twente <http://www.math.utwente.nl/nacm/>`_.
- Kestrel_: a parallelized CFD solver for high-resolution solutions of gas
  dynamics, and is constructed by using Python_.
- SfePy_: a FEM solver for PDEs, written in Python_ and C/FORTRAN.  SfePy
  stresses on mixing languages.
- Sundance_: a FEM solver for PDEs, written in C++.  Sundance uses Trilinos_
  for parallel computing.

.. _Python: http://www.python.org/

.. _FEniCS: http://www.fenicsproject.org/
.. _Simula: http://simula.no/
.. _FiPy: http://www.ctcms.nist.gov/fipy/
.. _NIST: http://www.nist.gov/
.. _Hermes: http://hpfem.org/hermes/
.. _hpGEM: http://wwwhome.math.utwente.nl/~hpgemdev/
.. _Kestrel: http://pdf.aiaa.org/preview/2010/CDReadyMASM10_1812/PV2010_511.pdf
.. _SfePy: http://sfepy.org/
.. _Sundance: http://www.math.ttu.edu/~kelong/Sundance/html/index.html

Additionally, there are other more general tools of which the purpose is to
help building PDE solvers.

- Hypre_: a library for solving large and sparse linear systems in parallel.
- PETSc_: a tool set for constructing parallel PDE solvers.  A large portion of
  PETSc is for linear algebra.  PETSc is developed at Argonne_ National
  Laboratory.
- Trilinos_: a collection of software packages for linear algebra, parallel
  processing, I/O, and thus solving PDEs.  Trilinos is developed at Sandia_
  National Laboratories.

.. _Hypre: http://acts.nersc.gov/hypre/
.. _PETSc: http://www.mcs.anl.gov/petsc/petsc-as/
.. _Argonne: http://www.anl.gov/
.. _Trilinos: http://trilinos.sandia.gov/
.. _Sandia: http://www.sandia.gov/
