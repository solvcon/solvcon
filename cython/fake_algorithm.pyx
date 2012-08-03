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

    property ncore:
        def __get__(self):
            return self._alg.ncore
        def __set__(self, val):
            self._alg.ncore = val
    property neq:
        def __get__(self):
            return self._alg.neq
        def __set__(self, val):
            self._alg.neq = val
    property time:
        def __get__(self):
            return self._alg.time
        def __set__(self, val):
            self._alg.time = val
    property time_increment:
        def __get__(self):
            return self._alg.time_increment
        def __set__(self, val):
            self._alg.time_increment = val

    property sol:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = self._mesh.ngstcell + self._mesh.ncell
            shape[1] = self._alg.neq
            cdef int shift = self._mesh.ngstcell * shape[1]
            return cnp.PyArray_SimpleNewFromData(
                2, shape, cnp.NPY_DOUBLE, self._alg.sol - shift)
        def __set__(self, cnp.ndarray[double, ndim=2, mode="c"] nda):
            assert nda.shape[0] == self._mesh.ngstcell + self._mesh.ncell
            assert nda.shape[1] == self._alg.neq
            if nda.shape[0] * nda.shape[1] == 0:
                self._alg.sol = NULL
            else:
                self._alg.sol = &nda[self._mesh.ngstcell,0]

    property soln:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = self._mesh.ngstcell + self._mesh.ncell
            shape[1] = self._alg.neq
            cdef int shift = self._mesh.ngstcell * shape[1]
            return cnp.PyArray_SimpleNewFromData(
                2, shape, cnp.NPY_DOUBLE, self._alg.soln - shift)
        def __set__(self, cnp.ndarray[double, ndim=2, mode="c"] nda):
            assert nda.shape[0] == self._mesh.ngstcell + self._mesh.ncell
            assert nda.shape[1] == self._alg.neq
            if nda.shape[0] * nda.shape[1] == 0:
                self._alg.soln = NULL
            else:
                self._alg.soln = &nda[self._mesh.ngstcell,0]

    property dsol:
        def __get__(self):
            cdef cnp.npy_intp shape[3]
            shape[0] = self._mesh.ngstcell + self._mesh.ncell
            shape[1] = self._alg.neq
            shape[2] = self._mesh.ndim
            cdef int shift = self._mesh.ngstcell * shape[1] * shape[2]
            return cnp.PyArray_SimpleNewFromData(
                3, shape, cnp.NPY_DOUBLE, self._alg.dsol - shift)
        def __set__(self, cnp.ndarray[double, ndim=3, mode="c"] nda):
            assert nda.shape[0] == self._mesh.ngstcell + self._mesh.ncell
            assert nda.shape[1] == self._alg.neq
            assert nda.shape[2] == self._mesh.ndim
            if nda.shape[0] * nda.shape[1] * nda.shape[2] == 0:
                self._alg.dsol = NULL
            else:
                self._alg.dsol = &nda[self._mesh.ngstcell,0,0]

    property dsoln:
        def __get__(self):
            cdef cnp.npy_intp shape[3]
            shape[0] = self._mesh.ngstcell + self._mesh.ncell
            shape[1] = self._alg.neq
            shape[2] = self._mesh.ndim
            cdef int shift = self._mesh.ngstcell * shape[1] * shape[2]
            return cnp.PyArray_SimpleNewFromData(
                3, shape, cnp.NPY_DOUBLE, self._alg.dsoln - shift)
        def __set__(self, cnp.ndarray[double, ndim=3, mode="c"] nda):
            assert nda.shape[0] == self._mesh.ngstcell + self._mesh.ncell
            assert nda.shape[1] == self._alg.neq
            assert nda.shape[2] == self._mesh.ndim
            if nda.shape[0] * nda.shape[1] * nda.shape[2] == 0:
                self._alg.dsoln = NULL
            else:
                self._alg.dsoln = &nda[self._mesh.ngstcell,0,0]

    property cecnd:
        def __get__(self):
            cdef cnp.npy_intp shape[3]
            shape[0] = self._mesh.ngstcell + self._mesh.ncell
            shape[1] = CLMFC+1
            shape[2] = self._mesh.ndim
            cdef int shift = self._mesh.ngstcell * shape[1] * shape[2]
            return cnp.PyArray_SimpleNewFromData(
                3, shape, cnp.NPY_DOUBLE, self._alg.cecnd - shift)
        def __set__(self, cnp.ndarray[double, ndim=3, mode="c"] nda):
            assert nda.shape[0] == self._mesh.ngstcell + self._mesh.ncell
            assert nda.shape[1] == CLMFC+1
            assert nda.shape[2] == self._mesh.ndim
            if nda.shape[0] * nda.shape[1] * nda.shape[2] == 0:
                self._alg.cecnd = NULL
            else:
                self._alg.cecnd = &nda[self._mesh.ngstcell,0,0]

    property cevol:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = self._mesh.ngstcell + self._mesh.ncell
            shape[1] = CLMFC+1
            cdef int shift = self._mesh.ngstcell * shape[1]
            return cnp.PyArray_SimpleNewFromData(
                2, shape, cnp.NPY_DOUBLE, self._alg.cevol - shift)
        def __set__(self, cnp.ndarray[double, ndim=2, mode="c"] nda):
            assert nda.shape[0] == self._mesh.ngstcell + self._mesh.ncell
            assert nda.shape[1] == CLMFC+1
            if nda.shape[0] * nda.shape[1] == 0:
                self._alg.cevol = NULL
            else:
                self._alg.cevol = &nda[self._mesh.ngstcell,0]

    def calc_soln(self):
        sc_fake_algorithm_calc_soln(self._mesh, self._alg)

    def calc_dsoln(self):
        sc_fake_algorithm_calc_dsoln(self._mesh, self._alg)

# vim: set fenc=utf8 ft=pyrex ff=unix ai et sw=4 ts=4 tw=79:
