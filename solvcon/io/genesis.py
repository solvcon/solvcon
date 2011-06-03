# -*- coding: UTF-8 -*-
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
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

"""
Adapter to Genesis/ExodusII format.
"""

from .core import FormatIO
from .netcdf import NetCDF

class Genesis(NetCDF):
    """
    Model of Genesis/ExodusII file, based on netCDF reader wrapper.  The
    Genesis/ExodusII file must meet the following criteria: (i) have at least
    one block, (ii) each block needs a name, and (iii) each sideset (BC) needs
    a name.

    @ivar ndim: dimension (2 or 3).
    @itype ndim: int
    @ivar nnode: number of nodes.
    @itype nnode: int
    @ivar ncell: number of cells/elements.
    @itype ncell: int
    @ivar blks: list of tuples of (name, type_name, clnds) for each Genesis
        block.
    @itype blks: list
    @ivar bcs: list of tuples of (name, elem, side) for each BC (Genesis
        sideset).
    @itype bcs: list
    @ivar ndcrd: coordiate array.
    @itype ndcrd: numpy.ndarray
    """
    def __init__(self, *arg, **kw):
        super(Genesis, self).__init__(*arg, **kw)
        # shape data.
        self.ndim = None
        self.nnode = None
        self.ncell = None
        # blocks and BCs.
        self.blks = None
        self.bcs = None
        # coordinate.
        self.ndcrd = None
        # mapper.
        self.emap = None

    def get_attr_text(self, name, varname):
        """
        Get the attribute text attached to an variable.

        @param name: name of the attribute.
        @type name: str
        @param varname: name of the variable.
        @type varname: str
        @return: the text.
        @rtype: str
        """
        from ctypes import POINTER, c_int, c_char, byref, create_string_buffer
        # get value ID.
        varid = c_int()
        retval = self.nc_inq_varid(self.ncid, varname, byref(varid))
        if retval != self.NC_NOERR:
            raise IOError(self.nc_strerror(retval))
        # get text.
        slen = self.get_dim('len_string')
        buf = create_string_buffer(slen)
        retval = self.nc_get_att_text(self.ncid, varid, name, buf)
        if retval != self.NC_NOERR:
            raise IOError(self.nc_strerror(retval))
        return buf.value

    def load(self):
        """
        Load mesh data.

        @return: nothing
        """
        from ctypes import c_int, byref
        from numpy import hstack
        # meta data.
        self.ndim = ndim = self.get_dim('num_dim')
        self.nnode = nnode = self.get_dim('num_nodes')
        self.ncell = self.get_dim('num_elem')
        # blocks.
        nblk = self.get_dim('num_el_blk')
        slen = self.get_dim('len_string')
        self.blks = self.get_lines('eb_names', (nblk, slen))
        for iblk in range(nblk):
            ncell = self.get_dim('num_el_in_blk%d' % (iblk+1))
            clnnd = self.get_dim('num_nod_per_el%d' % (iblk+1))
            clnds = self.get_array('connect%d' % (iblk+1),
                (ncell, clnnd), 'int32')
            type_name = self.get_attr_text('elem_type', 'connect%d' % (iblk+1))
            self.blks[iblk] = (self.blks[iblk], type_name, clnds)
        # BCs.
        nbc = self.get_dim('num_side_sets')
        self.bcs = self.get_lines('ss_names', (nbc, slen))
        for ibc in range(nbc):
            nface = self.get_dim('num_side_ss%d'%(ibc+1))
            elem = self.get_array('elem_ss%d'%(ibc+1), (nface,), 'int32')
            side = self.get_array('side_ss%d'%(ibc+1), (nface,), 'int32')
            self.bcs[ibc] = (self.bcs[ibc], elem, side)
        # coordinate.
        large = c_int()
        self.nc_get_att_int(self.ncid, self.NC_GLOBAL, 'file_size',
            byref(large))
        if large.value:
            vnames = ['coord%s' % ('xyz'[idm]) for idm in range(ndim)]
            crds = [self.get_array(vn, nnode, 'float64') for vn in vnames]
            self.ndcrd = hstack([crd.reshape((nnode,1)) for crd in crds])
        else:
            self.ndcrd = self.get_array('coord', (ndim, nnode),
                'float64').T.copy()
        # mapper.
        self.emap = self.get_array('elem_map', (self.ncell,), 'int32')

    def toblock(self, onlybcnames=None, bcname_mapper=None, fpdtype=None):
        """
        Convert Cubit/Genesis/ExodusII object to Block object.

        @keyword onlybcnames: positively list wanted names of BCs.
        @type onlybcnames: list
        @keyword bcname_mapper: map name to bc type number.
        @type bcname_mapper: dict
        @keyword fpdtype: floating-point dtype.
        @type fpdtype: str
        @return: Block object.
        @rtype: solvcon.block.Block
        """
        from ..block import Block
        blk = Block(ndim=self.ndim, nnode=self.nnode, ncell=self.ncell,
            fpdtype=fpdtype)
        self._convert_interior_to(blk)
        blk.build_interior()
        self._convert_bc_to(blk,
            onlynames=onlybcnames, name_mapper=bcname_mapper)
        blk.build_boundary()
        blk.build_ghost()
        return blk

    CLTPN_MAP = {
        'SHELL4': 2,    # CUBIT wants 2D quads in this way <shrug>.
        'TRI3': 3,
        'HEX8': 4,
        'TETRA': 5,
        'WEDGE': 6,
        'PYRAMID': 7,
    }
    def _convert_interior_to(self, blk):
        """
        Convert interior connectivities to Block object.

        @param blk: to-be-written Block object.
        @type blk: solvcon.block.Block
        @return: nothing
        """
        from ..block import elemtype
        # coordinate.
        blk.ndcrd[:] = self.ndcrd[:]
        # node definition.
        ien = 0
        for name, tname, clnds in self.blks:
            ist = ien
            ien += clnds.shape[0]
            # type.
            blk.cltpn[ist:ien] = self.CLTPN_MAP[tname]
            # nodes.
            nnd = elemtype[self.CLTPN_MAP[tname],2]
            sclnds = blk.clnds[ist:ien]
            sclnds[:,0] = nnd
            sclnds[:,1:nnd+1] = clnds
            sclnds[:,1:nnd+1] -= 1
            if tname == 'PRISM':
                arr = sclnds[:,2].copy()
                sclnds[:,2] = sclnds[:,3]
                sclnds[:,3] = arr
                arr[:] = sclnds[:,5]
                sclnds[:,5] = sclnds[:,6]
                sclnds[:,6] = arr
        # groups.
        blk.grpnames = [it[0] for it in self.blks]
        iblk = 0
        ien = 0
        for name, tname, clnds in self.blks:
            ist = ien
            ien += clnds.shape[0]
            blk.clgrp[ist:ien] = iblk
            iblk += 1

    def _convert_bc_to(self, blk, onlynames=None, name_mapper=None):
        """
        Convert boundary condition information into Block object.
        
        @param blk: to-be-written Block object.
        @type blk: solvcon.block.Block
        @keyword onlynames: positively list wanted names of BCs.
        @type onlynames: list
        @keyword name_mapper: map name to bc type and value dictionary; the two
            objects are organized in a tuple.
        @type name_mapper: dict
        @return: nothing.
        """
        # process all neutral bc objects.
        for ibc in range(len(self.bcs)):
            # extract boundary faces.
            bc = self._tobc(ibc, blk)
            # skip unwanted BCs.
            if onlynames:
                if bc.name not in onlynames:
                    continue
            # recreate BC according to name mapping.
            if name_mapper is not None:
                bct, vdict = name_mapper.get(bc.name, (None, None))
                if bct is not None:
                    bc = bct(bc=bc)
                    bc.feedValue(vdict)
            # save to block object.
            bc.sern = len(blk.bclist)
            bc.blk = blk
            blk.bclist.append(bc)

    # define map for clfcs.
    CLFCS_MAP = {}
    CLFCS_MAP[2] = [0,0,1,2,3,4]    # quadrilateral.
    CLFCS_MAP[3] = [1,2,3]  # triangle.
    CLFCS_MAP[4] = [5,2,6,4,1,3]    # hexahedron.
    CLFCS_MAP[5] = [2,4,3,1]    # tetrahedron.
    CLFCS_MAP[6] = [4,5,3,1,2]  # prism.
    CLFCS_MAP[7] = [2,3,4,1,5]  # pyramid.
    def _tobc(self, ibc, blk):
        """
        Extract boundary condition information from self to become a BC object.
        Only process element/cell type of boundary information.

        @param ibc: index of the BC to be extracted.
        @type ibc: int
        @param blk: Block object for reference, nothing will be altered.
        @type blk: solvcon.block.Block
        @return: generic BC object.
        @rtype: solvcon.boundcond.BC
        """
        from numpy import empty
        from ..boundcond import BC
        clfcs_map = self.CLFCS_MAP
        cltpn = blk.cltpn
        clfcs = blk.clfcs
        name, elem, side = self.bcs[ibc]
        nbnd = elem.shape[0]
        # extrace boundary face list.
        facn = empty((nbnd,3), dtype='int32')
        facn.fill(-1)
        ibnd = 0
        while ibnd < nbnd:
            icl = elem[ibnd] - 1
            tpn = cltpn[icl]
            ifl = clfcs_map[tpn][side[ibnd]-1]
            facn[ibnd,0] = clfcs[icl,ifl]
            ibnd += 1
        # craft BC object.
        bc = BC(fpdtype=blk.fpdtype)
        bc.name = name
        slct = facn[:,0].argsort()   # sort face list for bc object.
        bc.facn = facn[slct]
        # finish.
        return bc

class GenesisIO(FormatIO):
    """
    Proxy to Cubit/Genesis/ExodusII file format.
    """
    def load(self, stream, bcrej=None):
        """
        Load block from stream with BC mapper applied.

        @keyword stream: file name to be read.
        @type stream: str
        @keyword bcrej: names of the BC to reject.
        @type bcrej: list
        @return: the loaded block.
        @rtype: solvcon.block.Block
        """
        # load file into memory.
        assert isinstance(stream, basestring)
        gn = Genesis(stream)
        gn.load()
        gn.close_file()
        # convert loaded neutral object into block object.
        if bcrej:
            onlybcnames = list()
            for name, elem, side in gn.bcs:
                if name not in bcrej:
                    onlybcnames.append(name)
        else:
            onlybcnames = None
        blk = gn.toblock(onlybcnames=onlybcnames)
        return blk
