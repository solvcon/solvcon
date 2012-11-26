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

from mesh cimport sc_mesh_t, FCMND, CLMND, CLMFC, FCREL, BFREL
import numpy as np
cimport numpy as cnp

cdef extern:
    void sc_mesh_build_ghost(sc_mesh_t *msd, int *bndfcs)
    int sc_mesh_calc_metric(sc_mesh_t *msd, int use_incenter)
    int sc_mesh_extract_faces_from_cells(sc_mesh_t *msd, int mface,
            int *pnface, int *clfcs, int *fctpn, int *fcnds, int *fccls)
    int sc_mesh_build_rcells(sc_mesh_t *msd, int *rcells, int *rcellno)
    int sc_mesh_build_csr(sc_mesh_t *msd, int *rcells, int *adjncy)

cdef extern from "stdlib.h":
    void* malloc(size_t size)

# initialize NumPy.
cnp.import_array()

cdef class Mesh:
    """
    Data set of unstructured meshes of mixed elements.
    """
    def __cinit__(self):
        self._msd = <sc_mesh_t *>malloc(sizeof(sc_mesh_t));

    def setup_mesh(self, blk):
        """
        :param blk: External source of mesh data.
        :type blk: .block.Block

        Set up mesh data from external object.
        """
        # meta data.
        self._msd.ndim = blk.ndim
        self._msd.nnode = blk.nnode
        self._msd.nface = blk.nface
        self._msd.ncell = blk.ncell
        self._msd.nbound = blk.nbound
        self._msd.ngstnode = blk.ngstnode
        self._msd.ngstface = blk.ngstface
        self._msd.ngstcell = blk.ngstcell
        # geometry arrays.
        cdef cnp.ndarray[double, ndim=2, mode="c"] ndcrd = blk.ndcrd
        if ndcrd.shape[0] * ndcrd.shape[1] == 0:
            self._msd.ndcrd = NULL
        else:
            self._msd.ndcrd = &ndcrd[0,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] fccnd = blk.fccnd
        if fccnd.shape[0] * fccnd.shape[1] == 0:
            self._msd.fccnd = NULL
        else:
            self._msd.fccnd = &fccnd[0,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] fcnml = blk.fcnml
        if fcnml.shape[0] * fcnml.shape[1] == 0:
            self._msd.fcnml = NULL
        else:
            self._msd.fcnml = &fcnml[0,0]
        cdef cnp.ndarray[double, ndim=1, mode="c"] fcara = blk.fcara
        if fcara.shape[0] == 0:
            self._msd.fcara = NULL
        else:
            self._msd.fcara = &fcara[0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] clcnd = blk.clcnd
        if clcnd.shape[0] * clcnd.shape[1] == 0:
            self._msd.clcnd = NULL
        else:
            self._msd.clcnd = &clcnd[0,0]
        cdef cnp.ndarray[double, ndim=1, mode="c"] clvol = blk.clvol
        if clvol.shape[0] == 0:
            self._msd.clvol = NULL
        else:
            self._msd.clvol = &clvol[0]
        # meta arrays.
        cdef cnp.ndarray[int, ndim=1, mode="c"] fctpn = blk.fctpn
        if fctpn.shape[0] == 0:
            self._msd.fctpn = NULL
        else:
            self._msd.fctpn = &fctpn[0]
        cdef cnp.ndarray[int, ndim=1, mode="c"] cltpn = blk.cltpn
        if cltpn.shape[0] == 0:
            self._msd.cltpn = NULL
        else:
            self._msd.cltpn = &cltpn[0]
        cdef cnp.ndarray[int, ndim=1, mode="c"] clgrp = blk.clgrp
        if clgrp.shape[0] == 0:
            self._msd.clgrp = NULL
        else:
            self._msd.clgrp = &clgrp[0]
        # connectivity arrays.
        cdef cnp.ndarray[int, ndim=2, mode="c"] fcnds = blk.fcnds
        if fcnds.shape[0] * fcnds.shape[1] == 0:
            self._msd.fcnds = NULL
        else:
            self._msd.fcnds = &fcnds[0,0]
        cdef cnp.ndarray[int, ndim=2, mode="c"] fccls = blk.fccls
        if fccls.shape[0] * fccls.shape[1] == 0:
            self._msd.fccls = NULL
        else:
            self._msd.fccls = &fccls[0,0]
        cdef cnp.ndarray[int, ndim=2, mode="c"] clnds = blk.clnds
        if clnds.shape[0] * clnds.shape[1] == 0:
            self._msd.clnds = NULL
        else:
            self._msd.clnds = &clnds[0,0]
        cdef cnp.ndarray[int, ndim=2, mode="c"] clfcs = blk.clfcs
        if clfcs.shape[0] * clfcs.shape[1] == 0:
            self._msd.clfcs = NULL
        else:
            self._msd.clfcs = &clfcs[0,0]

    def build_ghost(self, cnp.ndarray[int, ndim=2, mode="c"] bndfcs):
        """
        :param bndfcs: Boundary faces.
        :type bndfcs: numpy.ndarray
        :return: Nothing.

        Build data for ghost cells and related information.
        """
        sc_mesh_build_ghost(self._msd, &bndfcs[0,0])

    def calc_metric(self, use_incenter):
        """
        :return: Nothing.

        Calculate metrics including normal vector and area of faces, and
        centroid coordinates and volume of cells.
        """
        cdef use_incenter_val = 1 if use_incenter else 0
        sc_mesh_calc_metric(self._msd, use_incenter_val)

    def extract_faces_from_cells(self, int max_nfc):
        """
        :param max_nfc: Maximum number of faces allowed.
        :type max_nfc: int
        :return: clfcs, fctpn, fcnds, fccls
        :rtype: tuple of numpy.ndarray

        Extract face definition from defined cell data.
        """
        # declare data.
        assert max_nfc > 0
        assert self._msd.ncell > 0
        cdef cnp.ndarray[int, ndim=2, mode="c"] clfcs = np.empty(
            (self._msd.ncell, CLMFC+1), dtype='int32')
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
        cdef int nface
        sc_mesh_extract_faces_from_cells(self._msd, max_nfc,
                &nface, &clfcs[0,0], &fctpn[0], &fcnds[0,0], &fccls[0,0])
        # shuffle the result.
        clfcs = clfcs[:nface,:].copy()
        fctpn = fctpn[:nface].copy()
        fcnds = fcnds[:nface,:].copy()
        fccls = fccls[:nface,:].copy()
        # return.
        return clfcs, fctpn, fcnds, fccls

    def create_csr(self):
        """
        :return: xadj, adjncy
        :rtype: tuple of numpy.ndarray

        Build the connectivity graph in the CSR (compressed storage format)
        required by METIS.
        """
        # build the table of related cells.
        cdef cnp.ndarray[int, ndim=2, mode="c"] rcells = np.empty(
            (self._msd.ncell, CLMFC), dtype='int32')
        cdef cnp.ndarray[int, ndim=1, mode="c"] rcellno = np.empty(
            self._msd.ncell, dtype='int32')
        sc_mesh_build_rcells(self._msd, &rcells[0,0], &rcellno[0])
        # build xadj: cell boundaries.
        xadj = np.empty(self._msd.ncell+1, dtype='int32')
        xadj[0] = 0
        xadj[1:] = np.add.accumulate(rcellno)
        # build adjncy: edge/relations.
        cdef cnp.ndarray[int, ndim=1, mode="c"] adjncy = np.empty(
            xadj[-1], dtype='int32')
        sc_mesh_build_csr(self._msd, &rcells[0,0], &adjncy[0])
        return xadj, adjncy

# vim: set fenc=utf8 ft=pyrex ff=unix ai et nu sw=4 ts=4 tw=79:
