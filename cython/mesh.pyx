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

from mesh cimport sc_mesh, FCMND, CLMND, CLMFC, FCREL, BFREL
from mesh cimport (sc_mesh_build_ghost, sc_mesh_calc_metric,
    sc_mesh_extract_faces_from_cells, sc_mesh_build_rcells, sc_mesh_build_csr)
import numpy as np
cimport numpy as cnp

# initialize NumPy.
cnp.import_array()

cdef extern from "stdlib.h":
    void* malloc(size_t size)

cdef class MeshData:
    def __cinit__(self):
        self._mesh = <sc_mesh *>malloc(sizeof(sc_mesh));

    property ndim:
        def __get__(self):
            return self._mesh.ndim
        def __set__(self, val):
            self._mesh.ndim = val
    property nnode:
        def __get__(self):
            return self._mesh.nnode
        def __set__(self, val):
            self._mesh.nnode = val
    property nface:
        def __get__(self):
            return self._mesh.nface
        def __set__(self, val):
            self._mesh.nface = val
    property ncell:
        def __get__(self):
            return self._mesh.ncell
        def __set__(self, val):
            self._mesh.ncell = val
    property nbound:
        def __get__(self):
            return self._mesh.nbound
        def __set__(self, val):
            self._mesh.nbound = val
    property ngstnode:
        def __get__(self):
            return self._mesh.ngstnode
        def __set__(self, val):
            self._mesh.ngstnode = val
    property ngstface:
        def __get__(self):
            return self._mesh.ngstface
        def __set__(self, val):
            self._mesh.ngstface = val
    property ngstcell:
        def __get__(self):
            return self._mesh.ngstcell
        def __set__(self, val):
            self._mesh.ngstcell = val

    property ndcrd:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = self._mesh.nnode
            shape[1] = self._mesh.ndim
            return cnp.PyArray_SimpleNewFromData(
                2, shape, cnp.NPY_DOUBLE, self._mesh.ndcrd)
        def __set__(self, cnp.ndarray[double, ndim=2, mode="c"] nda):
            assert nda.shape[0] == self._mesh.nnode
            assert nda.shape[1] == self._mesh.ndim
            if nda.shape[0] * nda.shape[1] == 0:
                self._mesh.ndcrd = NULL
            else:
                self._mesh.ndcrd = &nda[0,0]
    property fccnd:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = self._mesh.nface
            shape[1] = self._mesh.ndim
            return cnp.PyArray_SimpleNewFromData(
                2, shape, cnp.NPY_DOUBLE, self._mesh.fccnd)
        def __set__(self, cnp.ndarray[double, ndim=2, mode="c"] nda):
            assert nda.shape[0] == self._mesh.nface
            assert nda.shape[1] == self._mesh.ndim
            if nda.shape[0] * nda.shape[1] == 0:
                self._mesh.fccnd = NULL
            else:
                self._mesh.fccnd = &nda[0,0]
    property fcnml:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = self._mesh.nface
            shape[1] = self._mesh.ndim
            return cnp.PyArray_SimpleNewFromData(
                2, shape, cnp.NPY_DOUBLE, self._mesh.fcnml)
        def __set__(self, cnp.ndarray[double, ndim=2, mode="c"] nda):
            assert nda.shape[0] == self._mesh.nface
            assert nda.shape[1] == self._mesh.ndim
            if nda.shape[0] * nda.shape[1] == 0:
                self._mesh.fcnml = NULL
            else:
                self._mesh.fcnml = &nda[0,0]
    property fcara:
        def __get__(self):
            cdef cnp.npy_intp shape[1]
            shape[0] = self._mesh.nface
            return cnp.PyArray_SimpleNewFromData(
                1, shape, cnp.NPY_DOUBLE, self._mesh.fcara)
        def __set__(self, cnp.ndarray[double, ndim=1, mode="c"] nda):
            assert nda.shape[0] == self._mesh.nface
            if nda.shape[0] == 0:
                self._mesh.fcara = NULL
            else:
                self._mesh.fcara = &nda[0]
    property clcnd:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = self._mesh.ncell
            shape[1] = self._mesh.ndim
            return cnp.PyArray_SimpleNewFromData(
                2, shape, cnp.NPY_DOUBLE, self._mesh.clcnd)
        def __set__(self, cnp.ndarray[double, ndim=2, mode="c"] nda):
            assert nda.shape[0] == self._mesh.ncell
            assert nda.shape[1] == self._mesh.ndim
            if nda.shape[0] * nda.shape[1] == 0:
                self._mesh.clcnd = NULL
            else:
                self._mesh.clcnd = &nda[0,0]
    property clvol:
        def __get__(self):
            cdef cnp.npy_intp shape[1]
            shape[0] = self._mesh.ncell
            return cnp.PyArray_SimpleNewFromData(
                1, shape, cnp.NPY_DOUBLE, self._mesh.clvol)
        def __set__(self, cnp.ndarray[double, ndim=1, mode="c"] nda):
            assert nda.shape[0] == self._mesh.ncell
            if nda.shape[0] == 0:
                self._mesh.clvol = NULL
            else:
                self._mesh.clvol = &nda[0]

    property fctpn:
        def __get__(self):
            cdef cnp.npy_intp shape[1]
            shape[0] = self._mesh.nface
            return cnp.PyArray_SimpleNewFromData(
                1, shape, cnp.NPY_INT, self._mesh.fctpn)
        def __set__(self, cnp.ndarray[int, ndim=1, mode="c"] nda):
            assert nda.shape[0] == self._mesh.nface
            if nda.shape[0] == 0:
                self._mesh.fctpn = NULL
            else:
                self._mesh.fctpn = &nda[0]
    property cltpn:
        def __get__(self):
            cdef cnp.npy_intp shape[1]
            shape[0] = self._mesh.ncell
            return cnp.PyArray_SimpleNewFromData(
                1, shape, cnp.NPY_INT, self._mesh.cltpn)
        def __set__(self, cnp.ndarray[int, ndim=1, mode="c"] nda):
            assert nda.shape[0] == self._mesh.ncell
            if nda.shape[0] == 0:
                self._mesh.cltpn = NULL
            else:
                self._mesh.cltpn = &nda[0]
    property clgrp:
        def __get__(self):
            cdef cnp.npy_intp shape[1]
            shape[0] = self._mesh.ncell
            return cnp.PyArray_SimpleNewFromData(
                1, shape, cnp.NPY_INT, self._mesh.clgrp)
        def __set__(self, cnp.ndarray[int, ndim=1, mode="c"] nda):
            assert nda.shape[0] == self._mesh.ncell
            if nda.shape[0] == 0:
                self._mesh.clgrp = NULL
            else:
                self._mesh.clgrp = &nda[0]

    property fcnds:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = self._mesh.nface
            shape[1] = FCMND+1
            return cnp.PyArray_SimpleNewFromData(
                2, shape, cnp.NPY_INT, self._mesh.fcnds)
        def __set__(self, cnp.ndarray[int, ndim=2, mode="c"] nda):
            assert nda.shape[0] == self._mesh.nface
            assert nda.shape[1] == FCMND+1
            if nda.shape[0] * nda.shape[1] == 0:
                self._mesh.fcnds = NULL
            else:
                self._mesh.fcnds = &nda[0,0]
    property fccls:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = self._mesh.nface
            shape[1] = FCREL
            return cnp.PyArray_SimpleNewFromData(
                2, shape, cnp.NPY_INT, self._mesh.fccls)
        def __set__(self, cnp.ndarray[int, ndim=2, mode="c"] nda):
            assert nda.shape[0] == self._mesh.nface
            assert nda.shape[1] == FCREL
            if nda.shape[0] * nda.shape[1] == 0:
                self._mesh.fccls = NULL
            else:
                self._mesh.fccls = &nda[0,0]
    property clnds:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = self._mesh.ncell
            shape[1] = CLMND+1
            return cnp.PyArray_SimpleNewFromData(
                2, shape, cnp.NPY_INT, self._mesh.clnds)
        def __set__(self, cnp.ndarray[int, ndim=2, mode="c"] nda):
            assert nda.shape[0] == self._mesh.ncell
            assert nda.shape[1] == CLMND+1
            if nda.shape[0] * nda.shape[1] == 0:
                self._mesh.clnds = NULL
            else:
                self._mesh.clnds = &nda[0,0]
    property clfcs:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = self._mesh.ncell
            shape[1] = CLMFC+1
            return cnp.PyArray_SimpleNewFromData(
                2, shape, cnp.NPY_INT, self._mesh.clfcs)
        def __set__(self, cnp.ndarray[int, ndim=2, mode="c"] nda):
            assert nda.shape[0] == self._mesh.ncell
            assert nda.shape[1] == CLMFC+1
            if nda.shape[0] * nda.shape[1] == 0:
                self._mesh.clfcs = NULL
            else:
                self._mesh.clfcs = &nda[0,0]

    def build_ghost(self, cnp.ndarray[int, ndim=2, mode="c"] bndfcs):
        sc_mesh_build_ghost(self._mesh, &bndfcs[0,0])

    def calc_metric(self, use_incenter):
        cdef use_incenter_val
        use_incenter_val = 1 if use_incenter else 0
        sc_mesh_calc_metric(self._mesh, use_incenter_val)

    def extract_faces_from_cells(self, max_nfc):
        # declare data.
        cdef int nface
        cdef cnp.ndarray[int, ndim=2, mode="c"] clfcs = np.empty(
            (self.ncell, CLMFC+1), dtype='int32')
        cdef cnp.ndarray[int, ndim=1, mode="c"] fctpn = np.empty(
            max_nfc, dtype='int32')
        cdef cnp.ndarray[int, ndim=2, mode="c"] fcnds = np.empty(
            (max_nfc, FCMND+1), dtype='int32')
        cdef cnp.ndarray[int, ndim=2, mode="c"] fccls = np.empty(
            (max_nfc, FCREL), dtype='int32')
        # initialize data.
        for arr in clfcs, fcnds, fccls:
            arr.fill(-1)
        # call worker.
        sc_mesh_extract_faces_from_cells(self._mesh, <int> max_nfc,
                &nface, &clfcs[0,0], &fctpn[0], &fcnds[0,0], &fccls[0,0])
        # shuffle the result.
        clfcs = clfcs[:nface,:].copy()
        fctpn = fctpn[:nface].copy()
        fcnds = fcnds[:nface,:].copy()
        fccls = fccls[:nface,:].copy()
        # return.
        return clfcs, fctpn, fcnds, fccls

    def build_csr(self):
        # build the table of related cells.
        cdef cnp.ndarray[int, ndim=2, mode="c"] rcells = np.empty(
            (self.ncell, CLMFC), dtype='int32')
        cdef cnp.ndarray[int, ndim=1, mode="c"] rcellno = np.empty(
            self.ncell, dtype='int32')
        sc_mesh_build_rcells(self._mesh, &rcells[0,0], &rcellno[0])
        # build xadj: cell boundaries.
        xadj = np.empty(self.ncell+1, dtype='int32')
        xadj[0] = 0
        xadj[1:] = np.add.accumulate(rcellno)
        # build adjncy: edge/relations.
        cdef cnp.ndarray[int, ndim=1, mode="c"] adjncy = np.empty(
            xadj[-1], dtype='int32')
        sc_mesh_build_csr(self._mesh, &rcells[0,0], &adjncy[0])
        return xadj, adjncy

# vim: set fenc=utf8 ft=pyrex ff=unix ai et nu sw=4 ts=4 tw=79 
