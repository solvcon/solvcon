=============
Related Links
=============

This page collects information about software related to SOLVCON:

- Grid generator

  - CUBIT_: an advanced mesh generator developed at Sandia_ National
    Laboratories.  CUBIT has a very easy-to-use graphical interface and a
    comprehensive and productive command line interface.  CUBIT can generate
    very large meshes and supports parallel mesh generation.
  - FLUENT GAMBIT: a mesh generator widely used for CFD.  GAMBIT is supported
    by ANSYS_.
- Numerical library

  - LAPACK_ (Linear Algebra PACKage): the de facto tool set for linear algebra.
    LAPACK solves linear systems, eigen problems, and singular value problems.
  - METIS_: a software library for graph partitioning, developed by `George
    Karypis`_ at University of Minnesota, Twin Cities.
  - numpy_: the de facto software package for N-dimensional array in Python_.
    numpy is the core of scipy_, a comprehensive tool box for scientific
    computing in Python.
- GPU computing

  - CUDA_ (Compute Unified Device Architecture): CUDA is the most widely used
    programming environment for General-Purpose Graphic Processing Unit (GPGPU)
    computing.  CUDA is developed and provided by NVIDIA_.  It supports only
    the hardware made by NVIDIA, in general.
- I/O and Visualization

  - NetCDF_: a library for array data in scientific applications.  The file
    format has several versions, and the newer ones are based on HDF5_.  The
    Genesis/ExodusII mesh format is based on netCDF.
  - ParaView_: a powerful, open-source post-processing tool developed by
    Kitware_, Inc.  ParaView support parallel visualization, and provides a
    comprehensive set of functionalities.  ParaView is scriptable by using
    Python_.  It is build upon VTK_.
  - VTK_ (Visualization Toolkit): an open-source software library for computer
    graphics and visualization.  It is developed with C++ and provides a
    Python_ interface.  It is very easy to invoke VTK in Python.  For example,
    MayaVi_ is a Python package that uses VTK.
- Miscellaneous

  - Epydoc_: a tool for generating API documentation for Python
    packages/modules.  Epydoc supports multiple text markups, including
    epytext_, reStructuredText_, and Javadoc_.
  - MPI (Message-Passing Interface): the de facto standard for
    distributed-memory parallel computing.  Multiple implementations of the
    standard exist, including MPICH_, MVAPICH_, `Open MPI`_, etc.
  - nose_: a comprehensive test runner for Python code.  nose supports many
    unit test discovery modes and makes running unit tests easy.
  - SCons_: a software construction tool.  SCons is fundamentally a huge set of
    rule implemented in Python, and manipulated in Python.  The control files
    of SCons are nothing more than Python scripts that follow the special
    context management imposed by SCons.  SCons is a convenient replacement of
    make_.

.. _Python: http://python.org/

.. _CUBIT: http://cubit.sandia.gov/
.. _Sandia: http://www.sandia.gov/
.. _ANSYS: http://www.ansys.com/
.. _LAPACK: http://www.netlib.org/lapack/
.. _METIS: http://glaros.dtc.umn.edu/gkhome/views/metis
.. _George Karypis: http://glaros.dtc.umn.edu/gkhome/
.. _numpy: http://numpy.scipy.org/
.. _scipy: http://www.scipy.org/
.. _CUDA: http://www.nvidia.com/object/cuda_home_new.html
.. _NVIDIA: http://www.nvidia.com/
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf/index.html
.. _HDF5: http://www.hdfgroup.org/HDF5/
.. _ParaView: http://www.paraview.org/
.. _Kitware: http://www.kitware.com/
.. _VTK: http://www.vtk.org/
.. _MayaVi: http://code.enthought.com/projects/mayavi/
.. _SCons: http://www.scons.org/
.. _make: http://www.gnu.org/software/make/
.. _nose: http://somethingaboutorange.com/mrl/projects/nose/
.. _Epydoc: http://epydoc.sf.net/
.. _epytext: http://epydoc.sourceforge.net/epytextintro.html
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _Javadoc: http://java.sun.com/j2se/javadoc/
.. _MPICH: http://www.mcs.anl.gov/research/projects/mpich2/
.. _MVAPICH: http://mvapich.cse.ohio-state.edu/
.. _Open MPI: http://www.open-mpi.org/

You may also want to check :doc:`link_other`
