# Copyright (C) 2012 Yung-Yu Chen <yyc@solvcon.net>.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from mesh cimport sc_mesh_t, Mesh
from lincese_algorithm cimport sc_lincese_algorithm_t
import numpy as np
cimport numpy as cnp

# initialize NumPy.
cnp.import_array()

cdef extern:
    void sc_lincese_algorithm_calc_solt_2d(sc_mesh_t *msd, sc_lincese_algorithm_t *alg)
    void sc_lincese_algorithm_calc_solt_3d(sc_mesh_t *msd, sc_lincese_algorithm_t *alg)
    void sc_lincese_algorithm_calc_soln_2d(sc_mesh_t *msd, sc_lincese_algorithm_t *alg)
    void sc_lincese_algorithm_calc_soln_3d(sc_mesh_t *msd, sc_lincese_algorithm_t *alg)
    void sc_lincese_algorithm_calc_dsoln_2d(sc_mesh_t *msd, sc_lincese_algorithm_t *alg)
    void sc_lincese_algorithm_calc_dsoln_3d(sc_mesh_t *msd, sc_lincese_algorithm_t *alg)

cdef extern from "stdlib.h":
    void* malloc(size_t size)

cdef class LinceseAlgorithm(Mesh):
    """
    An algorithm class that does trivial calculation.
    """
    def __cinit__(self):
        self._alg = <sc_lincese_algorithm_t *>malloc(sizeof(sc_lincese_algorithm_t))

    def set_alg_double_array_2d(self,
            cnp.ndarray[double, ndim=2, mode="c"] nda, name, int shift):
        #self._alg[name] = &nda[shift,0]
        pass

    def setup_algorithm(self, svr):
        # meta data.
        self._alg.neq = svr.neq
        self._alg.time = svr.time
        self._alg.time_increment = svr.time_increment
        # arrays.
        cdef cnp.ndarray[double, ndim=2, mode="c"] sol = svr.sol
        self._alg.sol = &sol[self._msd.ngstcell,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] soln = svr.soln
        self._alg.soln = &soln[self._msd.ngstcell,0]
        cdef cnp.ndarray[double, ndim=3, mode="c"] dsol = svr.dsol
        self._alg.dsol = &dsol[self._msd.ngstcell,0,0]
        cdef cnp.ndarray[double, ndim=3, mode="c"] dsoln = svr.dsoln
        self._alg.dsoln = &dsoln[self._msd.ngstcell,0,0]
        cdef cnp.ndarray[double, ndim=3, mode="c"] cecnd = svr.cecnd
        self._alg.cecnd = &cecnd[self._msd.ngstcell,0,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] cevol = svr.cevol
        self._alg.cevol = &cevol[self._msd.ngstcell,0]

    def calc_solt(self):
        if self._msd.ndim == 3:
            sc_lincese_algorithm_calc_solt_3d(self._msd, self._alg)
        else:
            sc_lincese_algorithm_calc_solt_2d(self._msd, self._alg)

    def calc_soln(self):
        if self._msd.ndim == 3:
            sc_lincese_algorithm_calc_soln_3d(self._msd, self._alg)
        else:
            sc_lincese_algorithm_calc_soln_2d(self._msd, self._alg)

    def calc_dsoln(self):
        if self._msd.ndim == 3:
            sc_lincese_algorithm_calc_dsoln_3d(self._msd, self._alg)
        else:
            sc_lincese_algorithm_calc_dsoln_2d(self._msd, self._alg)

# vim: set fenc=utf8 ft=pyrex ff=unix ai et sw=4 ts=4 tw=79:
