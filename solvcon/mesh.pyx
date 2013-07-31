# Copyright (c) 2012, Yung-Yu Chen <yyc@solvcon.net>
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

    void METIS_PartGraphKway( int *n, int *xadj, int *adjncy, int *vwgt,
        int *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options,
        int *edgecut, int *part)

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
        :param max_nfc: Maximum value of possible number of faces to be extracted.
        :type max_nfc: int
        :return: Four interior :py:class:`numpy.ndarray` for
          :py:class:`solvcon.block.Block.clfcs`,
          :py:class:`solvcon.block.Block.fctpn`,
          :py:class:`solvcon.block.Block.fcnds`, and
          :py:class:`solvcon.block.Block.fccls`.

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

    def partition(self, int npart, vwgtarr=None):
        # obtain CSR.
        ret = self.create_csr()
        cdef cnp.ndarray[int, ndim=1, mode="c"] xadj = ret[0]
        cdef cnp.ndarray[int, ndim=1, mode="c"] adjncy = ret[1]
        # weighting.
        if vwgtarr is None:
            vwgtarr = np.empty(1, dtype='int32')
        cdef cnp.ndarray[int, ndim=1, mode="c"] vwgt = vwgtarr
        cdef int wgtflag
        if len(vwgtarr) == self._msd.ncell:
            wgtflag = 2
        else:
            vwgt.fill(0)
            wgtflag = 0
        # FIXME: not consistent when len(vwgt) == ncell.
        cdef cnp.ndarray[int, ndim=1, mode="c"] adjwgt = np.empty(
            1, dtype='int32')
        adjwgt.fill(0)
        # options.
        cdef cnp.ndarray[int, ndim=1, mode="c"] options = np.empty(
            5, dtype='int32')
        options.fill(0)
        # do the partition.
        cdef cnp.ndarray[int, ndim=1, mode="c"] part = np.empty(
            self._msd.ncell, dtype='int32')
        cdef int numflag = 0
        cdef int edgecut
        METIS_PartGraphKway(
            &self._msd.ncell,
            &xadj[0],
            &adjncy[0],
            &vwgt[0],
            &adjwgt[0],
            &wgtflag,
            &numflag,
            &npart,
            &options[0],
            # output.
            &edgecut,
            &part[0],
        )
        return edgecut, part

# vim: set fenc=utf8 ft=pyrex ff=unix ai et nu sw=4 ts=4 tw=79:
