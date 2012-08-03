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

    def setup_mesh(self, blk):
        # meta data.
        self._mesh.ndim = blk.ndim
        self._mesh.nnode = blk.nnode
        self._mesh.nface = blk.nface
        self._mesh.ncell = blk.ncell
        self._mesh.nbound = blk.nbound
        self._mesh.ngstnode = blk.ngstnode
        self._mesh.ngstface = blk.ngstface
        self._mesh.ngstcell = blk.ngstcell
        # geometry arrays.
        cdef cnp.ndarray[double, ndim=2, mode="c"] ndcrd = blk.ndcrd
        if ndcrd.shape[0] * ndcrd.shape[1] == 0:
            self._mesh.ndcrd = NULL
        else:
            self._mesh.ndcrd = &ndcrd[0,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] fccnd = blk.fccnd
        if fccnd.shape[0] * fccnd.shape[1] == 0:
            self._mesh.fccnd = NULL
        else:
            self._mesh.fccnd = &fccnd[0,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] fcnml = blk.fcnml
        if fcnml.shape[0] * fcnml.shape[1] == 0:
            self._mesh.fcnml = NULL
        else:
            self._mesh.fcnml = &fcnml[0,0]
        cdef cnp.ndarray[double, ndim=1, mode="c"] fcara = blk.fcara
        if fcara.shape[0] == 0:
            self._mesh.fcara = NULL
        else:
            self._mesh.fcara = &fcara[0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] clcnd = blk.clcnd
        if clcnd.shape[0] * clcnd.shape[1] == 0:
            self._mesh.clcnd = NULL
        else:
            self._mesh.clcnd = &clcnd[0,0]
        cdef cnp.ndarray[double, ndim=1, mode="c"] clvol = blk.clvol
        if clvol.shape[0] == 0:
            self._mesh.clvol = NULL
        else:
            self._mesh.clvol = &clvol[0]
        # meta arrays.
        cdef cnp.ndarray[int, ndim=1, mode="c"] fctpn = blk.fctpn
        if fctpn.shape[0] == 0:
            self._mesh.fctpn = NULL
        else:
            self._mesh.fctpn = &fctpn[0]
        cdef cnp.ndarray[int, ndim=1, mode="c"] cltpn = blk.cltpn
        if cltpn.shape[0] == 0:
            self._mesh.cltpn = NULL
        else:
            self._mesh.cltpn = &cltpn[0]
        cdef cnp.ndarray[int, ndim=1, mode="c"] clgrp = blk.clgrp
        if clgrp.shape[0] == 0:
            self._mesh.clgrp = NULL
        else:
            self._mesh.clgrp = &clgrp[0]
        # connectivity arrays.
        cdef cnp.ndarray[int, ndim=2, mode="c"] fcnds = blk.fcnds
        if fcnds.shape[0] * fcnds.shape[1] == 0:
            self._mesh.fcnds = NULL
        else:
            self._mesh.fcnds = &fcnds[0,0]
        cdef cnp.ndarray[int, ndim=2, mode="c"] fccls = blk.fccls
        if fccls.shape[0] * fccls.shape[1] == 0:
            self._mesh.fccls = NULL
        else:
            self._mesh.fccls = &fccls[0,0]
        cdef cnp.ndarray[int, ndim=2, mode="c"] clnds = blk.clnds
        if clnds.shape[0] * clnds.shape[1] == 0:
            self._mesh.clnds = NULL
        else:
            self._mesh.clnds = &clnds[0,0]
        cdef cnp.ndarray[int, ndim=2, mode="c"] clfcs = blk.clfcs
        if clfcs.shape[0] * clfcs.shape[1] == 0:
            self._mesh.clfcs = NULL
        else:
            self._mesh.clfcs = &clfcs[0,0]

    def build_ghost(self, cnp.ndarray[int, ndim=2, mode="c"] bndfcs):
        sc_mesh_build_ghost(self._mesh, &bndfcs[0,0])

    def calc_metric(self, use_incenter):
        cdef use_incenter_val = 1 if use_incenter else 0
        sc_mesh_calc_metric(self._mesh, use_incenter_val)

    def extract_faces_from_cells(self, max_nfc):
        # declare data.
        cdef int nface
        cdef cnp.ndarray[int, ndim=2, mode="c"] clfcs = np.empty(
            (self._mesh.ncell, CLMFC+1), dtype='int32')
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
            (self._mesh.ncell, CLMFC), dtype='int32')
        cdef cnp.ndarray[int, ndim=1, mode="c"] rcellno = np.empty(
            self._mesh.ncell, dtype='int32')
        sc_mesh_build_rcells(self._mesh, &rcells[0,0], &rcellno[0])
        # build xadj: cell boundaries.
        xadj = np.empty(self._mesh.ncell+1, dtype='int32')
        xadj[0] = 0
        xadj[1:] = np.add.accumulate(rcellno)
        # build adjncy: edge/relations.
        cdef cnp.ndarray[int, ndim=1, mode="c"] adjncy = np.empty(
            xadj[-1], dtype='int32')
        sc_mesh_build_csr(self._mesh, &rcells[0,0], &adjncy[0])
        return xadj, adjncy

# vim: set fenc=utf8 ft=pyrex ff=unix ai et nu sw=4 ts=4 tw=79:
