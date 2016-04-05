==============
Nested Looping
==============

.. py:currentmodule:: solvcon

The whole numerical simulation is controlled by an `Outer Loop`_ and many
`Inner Loops`_.  SOLVCON materializes them with :py:class:`MeshCase` and
:py:class:`MeshSolver`, respectively.

Outer Loop
==========

SOLVCON simulation is orchestrated by :py:class:`MeshCase`, which should be
subclassed to implement control logic for a specific application.  The
application can be a concrete model for a certain physical process, or an
abstraction of a group of related physical processes, which can be further
subclassed.

Because a case controls the whole process of a simulation run, for parallel
execution, there can be only one :py:class:`MeshCase` object residing in the
controller (head) node.

.. py:class:: MeshCase(**kw)

  :py:meth:`init` and :py:meth:`run` are the two primary methods responsible
  for the execution of the simulation case object.  Both methods accept a
  keyword parameter "level":

  - run level 0: fresh run (default),
  - run level 1: restart run,
  - run level 2: initialization only.

Initialize
++++++++++

.. automethod:: MeshCase.init

Time-March
++++++++++

.. automethod:: MeshCase.run

Arrangement
+++++++++++

.. py:attribute:: MeshCase.arrangements

  The class-level registry for arrangements.

.. automethod:: MeshCase.register_arrangement

Hooks on Cases
++++++++++++++

:py:class:`MeshHook` performs custom operations at certain pre-defined stages.

.. autoclass:: MeshHook

.. automethod:: MeshCase.defer

Inner Loops
===========

Numerical methods should be implemented by subclassing :py:class:`MeshSolver`.
The base class is defined as:

.. autoclass:: MeshSolver

  Two instance attributes are used to record the temporal information:
  
  .. autoinstanceattribute:: time
  
  .. autoinstanceattribute:: time_increment
  
  Four instance attributes are used to track the status of time-marching:
  
  .. autoinstanceattribute:: step_current
  
  .. autoinstanceattribute:: step_global
  
  .. autoinstanceattribute:: substep_run
  
  .. autoinstanceattribute:: substep_current

  Time-marchers:

  .. automethod:: register_marcher

  .. autoattribute:: mmnames

Useful entities are attached to the class :py:class:`MeshSolver`:

.. autoattribute:: MeshSolver.ALMOST_ZERO
  :annotation:

  A positive floating-point number close to zero.  The value is not
  ``DBL_MIN``, which can be accessed through :py:data:`sys.float_info`.

Time-Marching
+++++++++++++

.. automethod:: MeshSolver.march

.. py:attribute:: MeshSolver.marchret

  Values to be returned by this solver.  It will be set to a :py:class:`dict`
  in :py:meth:`march`.

.. py:attribute:: MeshSolver.runanchors

  This attribute is of type :py:class:`MeshAnchorList`, and the foundation of
  the anchor mechanism of SOLVCON.  An :py:class:`MeshAnchorList` object like
  this collects a set of :py:class:`MeshAnchor` objects, and is callable.  When
  being called, :py:attr:`runanchors` iterates the contained
  :py:class:`MeshAnchor` objects and invokes the corresponding methods of the
  individual :py:class:`MeshAnchor`.

.. py:attribute:: MeshSolver.der

  Derived data container as a :py:class:`dict`.

Parallel Computing
++++++++++++++++++

.. py:attribute:: MeshSolver.svrn

  This member indicates the serial number (0-based) of the
  :py:class:`MeshSolver` object.

.. py:attribute:: MeshSolver.nsvr

  The total number of collaborative solvers in the parallel run, and is
  initialized to ``None``.

Anchors on Solvers
++++++++++++++++++

.. autoclass:: MeshAnchor

  .. py:attribute:: svr

    The associated :py:class:`MeshSolver <solvcon.solver.MeshSolver>` instance.

.. autoclass:: MeshAnchorList

  .. py:attribute:: svr

    The associated :py:class:`MeshSolver <solvcon.solver.MeshSolver>` instance.

Boundary-Condition Treatments
=============================

.. autoclass:: BC

  .. autoinstanceattribute:: facn

  .. autoinstanceattribute:: value

  .. autoattribute:: nvalue

  .. automethod:: __len__

  .. automethod:: cloneTo

  .. automethod:: create_bcd

.. autoattribute:: BC.vnames

.. autoattribute:: BC.vdefaults

.. c:type:: sc_bound_t

  This ``struct`` contains essential information of a :py:class:`BC` object in
  C.

  .. c:member:: int nbound

    Number of boundary faces.  It's equivalent to what :py:meth:`BC.__len__`
    returns.

  .. c:member:: int nvalue

    Number of values per boundary face.

  .. c:member:: int* facn

    Pointer to the data storage of :py:attr:`BC.facn
    <solvcon.boundcond.BC.facn>`.

  .. c:member:: double* value

    Pointer to the data storage of :py:attr:`BC.value
    <solvcon.boundcond.BC.value>`.

.. py:class:: solvcon.mesh.Bound

  This class associates the C functions for mesh operations to the mesh data
  and exposes the functions to Python.

  .. py:attribute:: _bcd

    This attribute holds a C ``struct`` :c:type:`sc_bound_t` for internal use.

  .. py:method:: setup_bound(bc)

    :param bc: The :py:class:`~solvcon.boundcond.BC` object that supplies
      information.

.. vim: set spell ff=unix fenc=utf8 ft=rst:
