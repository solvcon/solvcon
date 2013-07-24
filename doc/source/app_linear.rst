============================================================
Second-Order Linear Solver (:py:mod:`solvcon.parcel.linear`)
============================================================

.. py:module:: solvcon.parcel.linear

.. py:module:: solvcon.parcel.linear.velstress

Velocity-Stress Equation Solver
===============================

See [1]_ for the basic formulation.

.. py:module:: solvcon.parcel.linear.velstress.logic

Solver Logic (:py:mod:`.velstress.logic <solvcon.parcel.linear.velstress.logic>`)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. autoclass:: VslinPWSolution

.. autoclass:: VslinSolver

  .. autoinstanceattribute:: mtrldict

  .. autoinstanceattribute:: mtrllist

.. autoclass:: VslinCase

.. py:module:: solvcon.parcel.linear.velstress.material

Material Definition (:py:mod:`.velstress.logic <solvcon.parcel.linear.velstress.material>`)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. autodata:: mltregy

.. autoclass:: Material

  .. autoinstanceattribute:: rho

  .. autoinstanceattribute:: al

  .. autoinstanceattribute:: be

  .. autoinstanceattribute:: ga

  .. autoinstanceattribute:: origstiff

  .. autoinstanceattribute:: stiff

.. py:module:: solvcon.parcel.linear._algorithm

Numerical Implementation (:py:mod:`._algorithm <solvcon.parcel.linear._algorithm>`)
===================================================================================

Let

- :math:`u_m` be the :math:`m`-th solution variable and :math:`m = 1, \ldots,
  M`.  :math:`M` is the total number of variables.
- :math:`u_{mx_{\mu}}` be the component of the gradient of :math:`u_m` along
  the :math:`x_{\mu}` axis in a Cartesian coordinate system.  :math:`\mu = 1,
  2` in two-dimensional space and :math:`\mu = 1, 2, 3` in three-dimensional
  space.

Common Data Structure
+++++++++++++++++++++

.. c:type:: linear_algorithm_t

  .. rubric:: Basic Information of the Solver

  .. c:member:: int neq
                double time
                double time_increment

    :c:data:`linear_algorithm_t.neq` is the number of equations or the number
    of variables in a mesh cell.  :c:data:`linear_algorithm_t.time` is the
    current time of the solver.  :c:data:`linear_algorithm_t.time_increment`
    is :math:`\Delta t`.

  .. rubric:: Parameters to the :math:`c\mbox{-}\tau` Scheme

  .. c:member:: int alpha
                double sigma0
                double taylor
                double cnbfac
                double sftfac
                double taumin
                double tauscale

  .. rubric:: Metric Arrays for the CESE Method

  .. c:member:: double *cecnd
                double *cevol
                double *sfmrc

  .. rubric:: Group Data

  .. c:member:: int ngroup
                int gdlen
                double *grpda

  .. rubric:: Scalar Parameters

  .. c:member:: int nsca
                double *amsca

  .. rubric:: Vector Parameters

  .. c:member:: int nvec
                double *amvec

  .. rubric:: Solution Arrays

  .. c:member:: double *sol
                double *soln
                double *solt
                double *dsol
                double *dsoln

  .. c:member:: double *stm
                double *cfl
                double *ocfl

Metric for CEs & SEs
++++++++++++++++++++

.. c:function:: void linear_prepare_ce_3d(sc_mesh_t *msd, linear_algorithm_t *alg)
                void linear_prepare_ce_2d(sc_mesh_t *msd, linear_algorithm_t *alg)

  Calculate the volume and centroid of conservation elements.

.. c:function:: void linear_prepare_sf_3d(sc_mesh_t *msd, linear_algorithm_t *alg)
                void linear_prepare_sf_2d(sc_mesh_t *msd, linear_algorithm_t *alg)

  Calculate the centroid and normal of hyperplanes of conservation elements
  and solution elements.

Time Marching
+++++++++++++

.. c:function:: void linear_calc_soln_3d(sc_mesh_t *msd, linear_algorithm_t *alg)
                void linear_calc_soln_2d(sc_mesh_t *msd, linear_algorithm_t *alg)

  Calculate the solutions of the next half time step (:math:`(u_m)_j^{n+1/2})`
  based on the informaiton at the current time step (:math:`n`).

.. c:function:: void linear_calc_solt_3d(sc_mesh_t *msd, linear_algorithm_t *alg)
                void linear_calc_solt_2d(sc_mesh_t *msd, linear_algorithm_t *alg)

  Calculate the changing rate of solutions (:math:`(u_mt)_j^n`).

.. c:function:: void linear_calc_jaco_3d(\
                  sc_mesh_t *msd, linear_algorithm_t *alg, \
                  int icl, double fcn[NEQ][NDIM], double jacos[NEQ][NEQ][NDIM])
                void linear_calc_jaco_2d(\
                  sc_mesh_t *msd, linear_algorithm_t *alg, \
                  int icl, double fcn[NEQ][NDIM], double jacos[NEQ][NEQ][NDIM])

  Calculate the Jacobian matrices.

.. c:function:: void linear_calc_dsoln_3d(sc_mesh_t *msd, linear_algorithm_t *alg)
                void linear_calc_dsoln_2d(sc_mesh_t *msd, linear_algorithm_t *alg)

  Calculate the gradient of solutions of the next half time step
  (:math:`(u_{mx_{\mu}})_j^{n+1/2}`) based on the information at the current
  time step (:math:`n`).

.. c:function:: void linear_calc_cfl_3d(sc_mesh_t *msd, linear_algorithm_t *alg)
                void linear_calc_cfl_2d(sc_mesh_t *msd, linear_algorithm_t *alg)

  Calculate the CFL number based on the eigenvalues of the linear Jacobian
  matrices.

Plane Wave Solution
+++++++++++++++++++

.. c:function:: void linear_calc_planewave_3d(\
                  sc_mesh_t *msd, linear_algorithm_t *alg, \
                  double *asol, double *adsol, double *amp, \
                  double *ctr, double *wvec, double afreq)
                void linear_calc_planewave_2d(\
                  sc_mesh_t *msd, linear_algorithm_t *alg, \
                  double *asol, double *adsol, double *amp, \
                  double *ctr, double *wvec, double afreq)

  Calculate the plane-wave solutions.

Wrapper for Numerical Code
++++++++++++++++++++++++++

.. py:class:: LinearAlgorithm

  This class wraps around the C functions for the second-order CESE method.

.. py:module:: solvcon.parcel.linear.solver

Numerical Controller (:py:mod:`.solver <solvcon.parcel.linear.solver>`)
=======================================================================

.. autoclass:: LinearSolver

  .. inheritance-diagram:: LinearSolver

.. autoclass:: LinearPeriodic

  .. inheritance-diagram:: LinearPeriodic

.. py:module:: solvcon.parcel.linear.case

Simulation Controller (:py:mod:`.case <solvcon.parcel.linear.case>`)
====================================================================

.. autoclass:: LinearCase

  .. inheritance-diagram:: LinearCase

.. py:module:: solvcon.parcel.linear.planewave

Helpers for Plane Wave (:py:mod:`.planewave <solvcon.parcel.linear.planewave>`)
===============================================================================

.. autoclass:: PlaneWaveSolution

.. autoclass:: PlaneWaveAnchor

  .. inheritance-diagram:: PlaneWaveAnchor

  .. autoinstanceattribute:: planewaves

.. autoclass:: PlaneWaveHook

  .. inheritance-diagram:: PlaneWaveHook

  .. autoinstanceattribute:: planewaves

  .. autoinstanceattribute:: norm

.. py:module:: solvcon.parcel.linear.inout

Helpers for I/O (:py:mod:`.inout <solvcon.parcel.linear.inout>`)
================================================================

.. autoclass:: MeshInfoHook

  .. inheritance-diagram:: MeshInfoHook

  .. autoinstanceattribute:: show_bclist

  .. autoinstanceattribute:: perffn

.. autoclass:: ProgressHook

  .. inheritance-diagram:: ProgressHook

  .. autoinstanceattribute:: linewidth

.. autoclass:: FillAnchor

  .. inheritance-diagram:: FillAnchor

  .. autoinstanceattribute:: mappers

.. autoclass:: CflAnchor

  .. inheritance-diagram:: CflAnchor

  .. autoinstanceattribute:: rsteps

.. autoclass:: CflHook

  .. inheritance-diagram:: CflHook

  .. autoinstanceattribute:: rsteps

  .. autoinstanceattribute:: name

  .. autoinstanceattribute:: cflmin

  .. autoinstanceattribute:: cflmax

  .. autoinstanceattribute:: fullstop

  .. autoinstanceattribute:: aCFL

  .. autoinstanceattribute:: mCFL

  .. autoinstanceattribute:: hnCFL

  .. autoinstanceattribute:: hxCFL

  .. autoinstanceattribute:: aadj

  .. autoinstanceattribute:: haadj

.. autoclass:: MarchSaveAnchor

  .. inheritance-diagram:: MarchSaveAnchor

  .. autoinstanceattribute:: anames

  .. autoinstanceattribute:: compressor

  .. autoinstanceattribute:: fpdtype

  .. autoinstanceattribute:: psteps

  .. autoinstanceattribute:: vtkfn_tmpl

.. autoclass:: PMarchSave

  .. inheritance-diagram:: PMarchSave

  .. autoinstanceattribute:: anames

  .. autoinstanceattribute:: compressor

  .. autoinstanceattribute:: fpdtype

  .. autoinstanceattribute:: altdir

  .. autoinstanceattribute:: altsym

  .. autoinstanceattribute:: vtkfn_tmpl

  .. autoinstanceattribute:: pextmpl

Reference
=========

.. [1] Yung-Yu Chen, Lixiang Yang, and Sheng-Tao John Yu, "Hyperbolicity of
  Velocity-Stress Equations for Waves in Anisotropic Elastic Solids", *Journal
  of Elasticity*, Volume 106, Number 2, Feb. 2012, Page 149-164.  `doi:
  10.1007/s10659-011-9315-8 <http://dx.doi.org/10.1007/s10659-011-9315-8>`.

.. vim: set spell ff=unix fenc=utf8 ft=rst:
