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

from mesh cimport MeshData, CLMFC
from fake_algorithm cimport (sc_fake_algorithm_t,
    sc_fake_algorithm_calc_soln, sc_fake_algorithm_calc_dsoln)
import numpy as np
cimport numpy as cnp

# initialize NumPy.
cnp.import_array()

cdef extern from "stdlib.h":
    void* malloc(size_t size)

cdef class FakeAlgorithm(MeshData):
    def __cinit__(self):
        self._alg = <sc_fake_algorithm_t *>malloc(sizeof(sc_fake_algorithm_t))

    def set_alg_double_array_2d(self,
            cnp.ndarray[double, ndim=2, mode="c"] nda, name, int shift):
        #self._alg[name] = &nda[shift,0]
        pass

    def setup_algorithm(self, svr):
        # meta data.
        self._alg.ncore = svr.ncore
        self._alg.neq = svr.neq
        self._alg.time = svr.time
        self._alg.time_increment = svr.time_increment
        # arrays.
        cdef cnp.ndarray[double, ndim=2, mode="c"] sol = svr.sol
        self._alg.sol = &sol[self._mesh.ngstcell,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] soln = svr.soln
        self._alg.soln = &soln[self._mesh.ngstcell,0]
        cdef cnp.ndarray[double, ndim=3, mode="c"] dsol = svr.dsol
        self._alg.dsol = &dsol[self._mesh.ngstcell,0,0]
        cdef cnp.ndarray[double, ndim=3, mode="c"] dsoln = svr.dsoln
        self._alg.dsoln = &dsoln[self._mesh.ngstcell,0,0]
        cdef cnp.ndarray[double, ndim=3, mode="c"] cecnd = svr.cecnd
        self._alg.cecnd = &cecnd[self._mesh.ngstcell,0,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] cevol = svr.cevol
        self._alg.cevol = &cevol[self._mesh.ngstcell,0]

    def calc_soln(self):
        sc_fake_algorithm_calc_soln(self._mesh, self._alg)

    def calc_dsoln(self):
        sc_fake_algorithm_calc_dsoln(self._mesh, self._alg)

# vim: set fenc=utf8 ft=pyrex ff=unix ai et sw=4 ts=4 tw=79:
