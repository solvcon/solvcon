# Copyright (c) 2013, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from solvcon.mesh cimport sc_mesh_t, sc_bound_t, Mesh, Bound
from ._algorithm cimport sc_bulk_algorithm_t
import numpy as np
cimport numpy as cnp

# initialize NumPy.
cnp.import_array()

cdef extern:
    # metrics.
    void sc_bulk_prepare_ce_3d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg)
    void sc_bulk_prepare_ce_2d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg)
    void sc_bulk_prepare_sf_3d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg)
    void sc_bulk_prepare_sf_2d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg)
    # utility calculators.
    void sc_bulk_process_physics_2d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg,
        double gasconst, double *vel, double *vor, double *vorm, double *rho,
        double *pre, double *sos, double *mac)
    void sc_bulk_process_physics_3d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg,
        double gasconst, double *vel, double *vor, double *vorm, double *rho,
        double *pre, double *sos, double *mac)
    # algorithm calculators.
    void sc_bulk_calc_cfl_2d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg)
    void sc_bulk_calc_cfl_3d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg)
    void sc_bulk_calc_solt_2d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg)
    void sc_bulk_calc_solt_3d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg)
    void sc_bulk_calc_soln_2d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg)
    void sc_bulk_calc_soln_3d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg)
    void sc_bulk_calc_dsoln_2d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg)
    void sc_bulk_calc_dsoln_3d(sc_mesh_t *msd, sc_bulk_algorithm_t *alg)
    # ghost information calculators.
    void sc_bulk_ghostgeom_mirror_2d(
        sc_mesh_t *msd, sc_bound_t *bcd, sc_bulk_algorithm_t *alg)
    void sc_bulk_ghostgeom_mirror_3d(
        sc_mesh_t *msd, sc_bound_t *bcd, sc_bulk_algorithm_t *alg)
    # boundary-condition treaters.
    void sc_bulk_bound_nonrefl_soln_2d(
        sc_mesh_t *msd, sc_bound_t *bcd, sc_bulk_algorithm_t *alg)
    void sc_bulk_bound_nonrefl_soln_3d(
        sc_mesh_t *msd, sc_bound_t *bcd, sc_bulk_algorithm_t *alg)
    void sc_bulk_bound_nonrefl_dsoln_2d(
        sc_mesh_t *msd, sc_bound_t *bcd, sc_bulk_algorithm_t *alg)
    void sc_bulk_bound_nonrefl_dsoln_3d(
        sc_mesh_t *msd, sc_bound_t *bcd, sc_bulk_algorithm_t *alg)

cdef extern from "stdlib.h":
    void* malloc(size_t size)
    void free(void* ptr)

cdef class BulkAlgorithm(Mesh):
    """
    An algorithm class that does trivial calculation.
    """
    def __cinit__(self):
        self._alg = <sc_bulk_algorithm_t *>malloc(sizeof(sc_bulk_algorithm_t))

    def __dealloc__(self):
        if NULL != self._alg:
            free(<void*>self._alg)
            self._alg = NULL

    def setup_algorithm(self, svr):
        # equations number.
        self._alg.neq = svr.neq
        # temporal information.
        self._alg.time = svr.time
        self._alg.time_increment = svr.time_increment
        # c-tau scheme parameters.
        self._alg.alpha = svr.alpha
        self._alg.sigma0 = svr.sigma0
        self._alg.taylor = svr.taylor
        self._alg.cnbfac = svr.cnbfac
        self._alg.sftfac = svr.sftfac
        self._alg.taumin = svr.taumin
        self._alg.tauscale = svr.tauscale
        # arrays.
        self._setup_cese_metrics(svr)
        self._setup_parameters(svr)
        self._setup_solutions(svr)

    def _setup_cese_metrics(self, svr):
        cdef cnp.ndarray[double, ndim=3, mode="c"] cecnd = svr.cecnd
        self._alg.cecnd = &cecnd[self._msd.ngstcell,0,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] cevol = svr.cevol
        self._alg.cevol = &cevol[self._msd.ngstcell,0]
        cdef cnp.ndarray[double, ndim=5, mode="c"] sfmrc = svr.sfmrc
        self._alg.sfmrc = &sfmrc[0,0,0,0,0]

    def _setup_parameters(self, svr):
        # group data.
        self._alg.ngroup = svr.ngroup
        self._alg.gdlen = svr.gdlen
        cdef cnp.ndarray[double, ndim=2, mode="c"] grpda = svr.grpda
        if 0 != svr.grpda.shape[1]:
            self._alg.grpda = &grpda[0,0]
        else:
            self._alg.grpda = NULL
        # scalar parameters.
        self._alg.nsca = svr.amsca.shape[1]
        cdef cnp.ndarray[double, ndim=2, mode="c"] amsca = svr.amsca
        if 0 != svr.amsca.shape[1]:
            self._alg.amsca = &amsca[self._msd.ngstcell,0]
        else:
            self._alg.amsca = NULL
        # vector parameters.
        self._alg.nvec = svr.amvec.shape[1]
        cdef cnp.ndarray[double, ndim=3, mode="c"] amvec = svr.amvec
        if 0 != svr.amvec.shape[1]:
            self._alg.amvec = &amvec[self._msd.ngstcell,0,0]
        else:
            self._alg.amvec = NULL

    def _setup_solutions(self, svr):
        cdef cnp.ndarray[double, ndim=2, mode="c"] sol = svr.sol
        self._alg.sol = &sol[self._msd.ngstcell,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] soln = svr.soln
        self._alg.soln = &soln[self._msd.ngstcell,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] solt = svr.solt
        self._alg.solt = &solt[self._msd.ngstcell,0]
        cdef cnp.ndarray[double, ndim=3, mode="c"] dsol = svr.dsol
        self._alg.dsol = &dsol[self._msd.ngstcell,0,0]
        cdef cnp.ndarray[double, ndim=3, mode="c"] dsoln = svr.dsoln
        self._alg.dsoln = &dsoln[self._msd.ngstcell,0,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] stm = svr.stm
        self._alg.stm = &stm[self._msd.ngstcell,0]
        cdef cnp.ndarray[double, ndim=1, mode="c"] cfl = svr.cfl
        self._alg.cfl = &cfl[self._msd.ngstcell]
        cdef cnp.ndarray[double, ndim=1, mode="c"] ocfl = svr.ocfl
        self._alg.ocfl = &ocfl[self._msd.ngstcell]

    def prepare_ce(self):
        if self._msd.ndim == 3:
            sc_bulk_prepare_ce_3d(self._msd, self._alg)
        else:
            sc_bulk_prepare_ce_2d(self._msd, self._alg)

    def prepare_sf(self):
        if self._msd.ndim == 3:
            sc_bulk_prepare_sf_3d(self._msd, self._alg)
        else:
            sc_bulk_prepare_sf_2d(self._msd, self._alg)

    def update(self, time, time_increment):
        self._alg.time = time
        self._alg.time_increment = time_increment

    def calc_cfl(self):
        if self._msd.ndim == 3:
            sc_bulk_calc_cfl_3d(self._msd, self._alg)
        else:
            sc_bulk_calc_cfl_2d(self._msd, self._alg)

    def calc_solt(self):
        if self._msd.ndim == 3:
            sc_bulk_calc_solt_3d(self._msd, self._alg)
        else:
            sc_bulk_calc_solt_2d(self._msd, self._alg)

    def calc_soln(self):
        if self._msd.ndim == 3:
            sc_bulk_calc_soln_3d(self._msd, self._alg)
        else:
            sc_bulk_calc_soln_2d(self._msd, self._alg)

    def calc_dsoln(self):
        if self._msd.ndim == 3:
            sc_bulk_calc_dsoln_3d(self._msd, self._alg)
        else:
            sc_bulk_calc_dsoln_2d(self._msd, self._alg)

    def ghostgeom_mirror(self, Bound bcd):
        if self._msd.ndim == 3:
            sc_bulk_ghostgeom_mirror_3d(self._msd, bcd._bcd, self._alg)
        else:
            sc_bulk_ghostgeom_mirror_2d(self._msd, bcd._bcd, self._alg)

    def bound_nonrefl_soln(self, Bound bcd):
        if self._msd.ndim == 3:
            sc_bulk_bound_nonrefl_soln_3d(self._msd, bcd._bcd, self._alg)
        else:
            sc_bulk_bound_nonrefl_soln_2d(self._msd, bcd._bcd, self._alg)

    def bound_nonrefl_dsoln(self, Bound bcd):
        if self._msd.ndim == 3:
            sc_bulk_bound_nonrefl_dsoln_3d(self._msd, bcd._bcd, self._alg)
        else:
            sc_bulk_bound_nonrefl_dsoln_2d(self._msd, bcd._bcd, self._alg)

# vim: set fenc=utf8 ft=pyrex ff=unix ai et sw=4 ts=4 tw=79:
