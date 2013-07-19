========
Tutorial
========

The goal of SOLVCON is to help code developers to focus on the numerical
algorithms.  These computing code can be written in C or other high-speed
languages and interfaced with SOLVCON.  SOLVCON has a general work flow that
support mesh loaders (`Gmsh <http://www.geuz.org/gmsh/>`__, FLUENT Gambit (R),
and `CUBIT <http://cubit.sandia.gov/>`__), MPI, and VTK.  These supportive
functionalities are ready to help developing problem solvers.

Set up the Environment
======================

Assume:

- SOLVCON is compiled without problems.  See :doc:`install` for more
  information.
- The compiled SOLVCON is located at ``$SCSRC``.
- You are using bash.

Usually we don't install SOLVCON into the OS, but use environment variable to
enable it, so that it's easier to modify the source code::

  export PYTHONPATH=$SCSRC:$PYTHONPATH

And then the following command::

  python -c "import solvcon; print solvcon"

should show you the correct location of SOLVCON.

There are various examples located at ``$SCSRC/examples``.  To follow the
examples, you need to:

- Install `ParaView <http://www.paraview.org/>`__.  On a Debian/Ubuntu, you can
  do it by executing::

    sudo apt-get install paraview

- Obtain example data.  You can do it by executing::

    scons --get-scdata

  in ``$SCSRC``.

More information of the verification examples can be found in
:doc:`verification`.

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

.. py:module:: solvcon.parcel.fake

Problem Solver
==============

To demonstrate how to develop a problem solver, SOLVCON provides a "fake" one
in :py:mod:`solvcon.parcel.fake`.  The package implements a trivial and
meaningless algorithm which is easy to validate.  The related files are all in
the directory ``$SCSRC/solvcon/parcel/fake``.  You can follow the source code
and test cases to learn about how to write a problem solver with SOLVCON.

There are two modules, :py:mod:`solver <solvcon.parcel.fake.solver>` and
:py:mod:`fake_algorithm <solvcon.parcel.fake.fake_algorithm>`, inside that
package.  They provides two classes: :py:class:`FakeSolver
<solvcon.parcel.fake.solver.FakeSolver>` and :py:class:`FakeAlgorithm
<solvcon.parcel.fake.fake_algorithm.FakeAlgorithm>`.  The former is the
higher-level API and purely written in Python.  The latter is implemented with
`Cython <http://cython.org>`__ to call low-level C code.  The real
number-crunching code is written in C:

.. c:function:: void fake_calc_soln(sc_mesh_t *msd, fake_algorithm_t *alg)

  :ref:`(Jump to the code listing). <fake_calc_soln_listing>`  Let
  :math:`u_j^n` be the solution at :math:`j`-th cell and :math:`n`-th time
  step, and :math:`v_j` be the volume at :math:`j`-th cell.  This function
  advances each value in the solution array (:c:data:`fake_algorithm_t.sol` and
  :c:data:`fake_algorithm_t.soln`) by using the following expression:

  .. math::

    u_j^n = u_j^{n-\frac{1}{2}} + \frac{\Delta t}{2}v_j

  Total number of values per cell is set in :c:data:`fake_algorithm_t.neq`.

.. c:function:: void fake_calc_dsoln(sc_mesh_t *msd, fake_algorithm_t *alg)

  :ref:`(Jump to the code listing). <fake_calc_dsoln_listing>`  Let
  :math:`(u_{x_{\mu}})_j^n` be the :math:`x_{\mu}` component of the gradient of
  :math:`u_j^n`, and :math:`(c_{\mu})_j` be the :math:`x_{\mu}` component of
  the centroid of the :math:`j`-th cell.  :math:`\mu = 1, 2` or :math:`\mu = 1,
  2, 3`.  This function advances each value in the solution gradient array
  (:c:data:`fake_algorithm_t.dsol` and :c:data:`fake_algorithm_t.dsoln`) by
  using the following expression:

  .. math::

    (u_{x_{\mu}})_j^n =
      (u_{x_{\mu}})j^{n-\frac{1}{2}} + \frac{\Delta t}{2}(c_{\mu})_j

  Total number of values per cell is set in :c:data:`fake_algorithm_t.neq`.

The Python/Cython/C hybrid style may seem complicated, but it is important for
performance.  The two C functions are wrapped with the Cython methods
:py:meth:`FakeAlgorithm.calc_soln
<solvcon.parcel.fake.fake_algorithm.FakeAlgorithm.calc_soln>` and
:py:meth:`FakeAlgorithm.calc_dsoln
<solvcon.parcel.fake.fake_algorithm.FakeAlgorithm.calc_dsoln>`, respectively.
Then, the higher level :py:class:`FakeSolver
<solvcon.parcel.fake.solver.FakeSolver>` will use the lower-level
:py:class:`FakeAlgorithm <solvcon.parcel.fake.fake_algorithm.FakeAlgorithm>` to
connect the underneath numerical algorithm to the supportive functionalities
prepared in SOLVCON.

.. py:module:: solvcon.parcel.fake.solver

:py:mod:`fake.solver <solvcon.parcel.fake.solver>`
++++++++++++++++++++++++++++++++++++++++++++++++++

This is the higher level module implemented in Python.

.. autoclass:: FakeSolver

  .. inheritance-diagram:: FakeSolver

  .. automethod:: __init__

  .. autoinstanceattribute:: neq

  .. automethod:: create_alg

  .. autoattribute:: _MMNAMES

  The following six methods build up the numerical algorithm.  They are
  registered into :py:attr:`_MMNAMES` with the present order.

  .. automethod:: update

  .. automethod:: calcsoln

  .. automethod:: ibcsoln

  .. automethod:: calccfl

  .. automethod:: calcdsoln

  .. automethod:: ibcdsoln

:py:class:`FakeSolver` defines the following solution arrays:

.. autoinstanceattribute:: FakeSolver.sol

.. autoinstanceattribute:: FakeSolver.soln

.. autoinstanceattribute:: FakeSolver.dsol

.. autoinstanceattribute:: FakeSolver.dsoln

.. py:module:: solvcon.parcel.fake.fake_algorithm

:py:mod:`fake.fake_algorithm <solvcon.parcel.fake.fake_algorithm>`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This is the lower level module implemented in Cython.  It is composed by two
files.  ``$SCSRC/solvcon/parcel/fake/fake_algorithm.pxd`` declares the data
structure for C.  ``$SCSRC/solvcon/parcel/fake/fake_algorithm.pyx`` defines the
wrapping code.

C API Declaration
-----------------

The Cython file ``fake_algorithm.pxd`` defines the following type for the
low-level C functions to access the data in a :py:class:`FakeAlgorithm` that
proxies :py:class:`FakeSolver <solvcon.parcel.fake.solver.FakeSolver>`:

.. c:type:: fake_algorithm_t

  .. c:member:: int neq

    This should be set to :py:attr:`FakeSolver.neq
    <solvcon.parcel.fake.solver.FakeSolver.neq>`.

  .. c:member:: double time

    This should be set to :py:attr:`MeshSolver.time
    <solvcon.solver.MeshSolver.time>`.

  .. c:member:: double time_increment

    This should be set to :py:attr:`MeshSolver.time_increment
    <solvcon.solver.MeshSolver.time_increment>`.

  .. c:member:: double* sol

    This should point to the 0-th cell of :py:attr:`FakeSolver.sol
    <solvcon.parcel.fake.solver.FakeSolver.sol>`.  Therefore the address of
    ghost cells is *smaller* than :c:data:`fake_algorithm_t.sol`.

  .. c:member:: double* soln

    This should point to the 0-th cell of :py:attr:`FakeSolver.soln
    <solvcon.parcel.fake.solver.FakeSolver.soln>`.  Therefore the address of
    ghost cells is *smaller* than :c:data:`fake_algorithm_t.soln`.

  .. c:member:: double* dsol

    This should point to the 0-th cell of :py:attr:`FakeSolver.dsol
    <solvcon.parcel.fake.solver.FakeSolver.dsol>`.  Therefore the address of
    ghost cells is *smaller* than :c:data:`fake_algorithm_t.dsol`.

  .. c:member:: double* dsoln

    This should point to the 0-th cell of :py:attr:`FakeSolver.dsoln
    <solvcon.parcel.fake.solver.FakeSolver.dsoln>`.  Therefore the address of
    ghost cells is *smaller* than :c:data:`fake_algorithm_t.dsoln`.

Wrapper Class
-------------

.. py:class:: FakeAlgorithm

  This class wraps around the C portion of the numerical method.

  .. py:method:: setup_algorithm(svr)

    A :py:class:`FakeAlgorithm` object shouldn't allocate memory.  Instead, a
    :py:class:`FakeSolver <solvcon.parcel.fake.solver.FakeSolver>` object
    should allocate the memory and pass the solver into the algorithm.

  .. py:method:: calc_soln

    Wraps the C functions :c:func:`fake_calc_soln` (where the algorithm is
    defined).

  .. py:method:: calc_dsoln

    Wraps the C functions :c:func:`fake_calc_dsoln` (where the algorithm is
    defined).

C Code Listing
++++++++++++++

.. _fake_calc_soln_listing:

.. rubric:: solvcon/parcel/fake/src/fake_calc_soln.c

.. literalinclude:: ../../solvcon/parcel/fake/src/fake_calc_soln.c
  :language: c
  :linenos:

.. _fake_calc_dsoln_listing:

.. rubric:: solvcon/parcel/fake/src/fake_calc_dsoln.c

.. literalinclude:: ../../solvcon/parcel/fake/src/fake_calc_dsoln.c
  :language: c
  :linenos:
