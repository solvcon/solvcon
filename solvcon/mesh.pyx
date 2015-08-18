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

from __future__ import absolute_import, division, print_function

from .mesh cimport sc_mesh_t, FCMND, CLMND, CLMFC, FCREL, BFREL
from libc.stdint cimport intptr_t
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

# Initialize NumPy.
cnp.import_array()


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


cdef class Table:
    """
    Lookup table that allows ghost entity.
    """

    def __cinit__(self, *args, **kw):
        self.nghost = -1 # Sentinel
        self.nbody = -1 # Sentinel
        self._full = <char*>(NULL) # Sentinel
        self._body = <char*>(NULL) # Sentinel

    def __init__(self, nghost, nbody, *args, **kw):
        self.nghost = nghost
        self.nbody = nbody
        # Pop all custom keyword arguments.
        dtype = kw.pop("dtype", None)
        creator_name = kw.pop("creation", "empty")
        # Create the ndarray.
        create = getattr(np, creator_name)
        self._nda = create([nghost+nbody]+list(args), dtype=dtype)
        if not self._nda.flags.c_contiguous:
            raise ValueError("not C Contiguous")
        ndim = len(self._nda.shape)
        if ndim == 0:
            raise ValueError("zero dimension is not allowed")
        # Assign pointers.
        cdef cnp.ndarray cnda = self._nda
        self._full = self._body = <char*>(cnda.data)
        offset = self._calc_offset(self._nda.shape, self.nghost)
        self._body += self._nda.itemsize * offset

    @staticmethod
    def _calc_offset(shape, nghost):
        second = 1 if 1 == len(shape) else np.multiply.reduce(shape[1:])
        return nghost * second

    def __getattr__(self, name):
        return getattr(self._nda, name)

    @property
    def _ghostaddr(self):
        cdef cnp.ndarray cnda = self._nda
        return <intptr_t>(cnda.data)

    @property
    def _bodyaddr(self):
        return <intptr_t>(self._body)

    @property
    def offset(self):
        """
        Element offset from the head of the ndarray to where the body starts.
        """
        return self._calc_offset(self._nda.shape, self.nghost)

    property F:
        """Full array."""
        def __get__(self):
            return self._nda[...]
        def __set__(self, arr):
            self._nda[...] = arr

    property G:
        """Ghost-part array."""
        def __get__(self):
            if self.nghost:
                return self._nda[self.nghost-1::-1,...]
            else:
                return self._nda[0:0,...]
        def __set__(self, arr):
            if self.nghost:
                self._nda[self.nghost-1::-1,...] = arr
            else:
                raise IndexError('nghost is zero')

    @property
    def _ghostpart(self):
        """Ghost-part array without setter."""
        if self.nghost:
            return self._nda[self.nghost-1::-1,...]
        else:
            return self._nda[0:0,...]

    property B:
        """Body-part array."""
        def __get__(self):
            return self._nda[self.nghost:,...]
        def __set__(self, arr):
            self._nda[self.nghost:,...] = arr

    @property
    def _bodypart(self):
        """Body-part array without setter."""
        return self._nda[self.nghost:,...]


cdef class Mesh:
    """
    Data set of unstructured meshes of mixed elements.
    """
    def __cinit__(self):
        self._msd = <sc_mesh_t *>malloc(sizeof(sc_mesh_t))

    def __dealloc__(self):
        if NULL != self._msd:
            free(<void*>self._msd)
            self._msd = NULL

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
        # before set array pointers, check them.
        self._check_pointers(blk)
        # geometry arrays.
        self._msd.ndcrd = <double*>self._get_table_bodyaddr(blk.tbndcrd)
        self._msd.fccnd = <double*>self._get_table_bodyaddr(blk.tbfccnd)
        self._msd.fcnml = <double*>self._get_table_bodyaddr(blk.tbfcnml)
        self._msd.fcara = <double*>self._get_table_bodyaddr(blk.tbfcara)
        self._msd.clcnd = <double*>self._get_table_bodyaddr(blk.tbclcnd)
        self._msd.clvol = <double*>self._get_table_bodyaddr(blk.tbclvol)
        # meta arrays.
        self._msd.fctpn = <int*>self._get_table_bodyaddr(blk.tbfctpn)
        self._msd.cltpn = <int*>self._get_table_bodyaddr(blk.tbcltpn)
        self._msd.clgrp = <int*>self._get_table_bodyaddr(blk.tbclgrp)
        # connectivity arrays.
        self._msd.fcnds = <int*>self._get_table_bodyaddr(blk.tbfcnds)
        self._msd.fccls = <int*>self._get_table_bodyaddr(blk.tbfccls)
        self._msd.clnds = <int*>self._get_table_bodyaddr(blk.tbclnds)
        self._msd.clfcs = <int*>self._get_table_bodyaddr(blk.tbclfcs)

    @staticmethod
    def _check_pointers(blk):
        cdef Table _table
        cdef cnp.ndarray _cnda
        # not checking ghost arrays because they aren't contiguous and thus
        # their addresses can't match.
        cdef cnp.ndarray _sharr
        cdef cnp.ndarray _arr
        for name in blk.TABLE_NAMES:
            table = getattr(blk, 'tb'+name)
            sharr = getattr(blk, 'sh'+name)
            assert sharr.flags.c_contiguous
            arr = getattr(blk, name)
            _table = table
            _cnda = _table._nda
            _sharr = sharr
            _arr = arr
            msgs = []
            # shared.
            vals = <size_t>(_cnda.data), <size_t>(_sharr.data)
            if vals[0] != vals[1]:
                msgs.append('shared(%d,%d)' % vals)
            # body.
            vals = <size_t>(_table._body), <size_t>(_arr.data)
            if vals[0] != vals[1]:
                msgs.append('body(%d,%d)' % vals)
            if msgs:
                tmpl = '%s array mismatch: %s'
                raise AttributeError(tmpl % (name, ', '.join(msgs)))

    cdef void* _get_table_bodyaddr(self, table):
        cdef Table _table = table
        if 0 == table.size:
            return NULL
        else:
            return <void*>(_table._body)

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


cdef class Bound:
    """
    Data set of boundary-condition treatment.
    """
    def __cinit__(self):
        self._bcd = <sc_bound_t *>malloc(sizeof(sc_bound_t))

    def setup_bound(self, bc):
        """
        :param bc: External source of mesh data.
        :type bc: .boundcond.BC

        Set up mesh data from external object.
        """
        # meta data.
        self._bcd.nbound = len(bc)
        self._bcd.nvalue = bc.nvalue
        # geometry arrays.
        cdef cnp.ndarray[int, ndim=2, mode="c"] facn = bc.facn
        if facn.shape[0] * facn.shape[1] == 0:
            self._bcd.facn = NULL
        else:
            self._bcd.facn = &facn[0,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] value = bc.value
        if value.shape[0] * value.shape[1] == 0:
            self._bcd.value = NULL
        else:
            self._bcd.value = &value[0,0]

# vim: set fenc=utf8 ft=pyrex ff=unix nobomb ai et sw=4 ts=4 tw=79:
