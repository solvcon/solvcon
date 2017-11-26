# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
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
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

"""Unstructured mesh definition."""


__all__ = [
    'UnstructuredBlock',
    'elemtype',
    'Block',
]


import numpy as np

from . import boundcond
from . import dependency
from . import block_legacy
dependency.import_module_may_fail('.march')


class UnstructuredBlock(object):
    """
    >>> # build a 2D block.
    >>> blk = UnstructuredBlock(ndim=2, nnode=4, nface=6, ncell=3, nbound=3)
    >>> blk.ndcrd[:,:] = (0,0), (-1,-1), (1,-1), (0,1)
    >>> blk.cltpn[:] = 3
    >>> blk.clnds[:,:4] = (3, 0,1,2), (3, 0,2,3), (3, 0,3,1)
    >>> blk.build_interior()
    >>> blk.build_boundary()
    >>> blk.build_ghost()
    >>> # also test for a short-hand method.
    >>> from .testing import create_trivial_2d_blk
    >>> tblk = create_trivial_2d_blk()
    >>> (tblk.shndcrd == blk.shndcrd).all()
    True
    >>> (tblk.shfccnd == blk.shfccnd).all()
    True
    >>> (tblk.shfccls == blk.shfccls).all()
    True
    """

    try:
        FCMND = march.UnstructuredBlock2D.FCMND
    except NameError:
        FCMND = 0
    try:
        CLMND = march.UnstructuredBlock2D.CLMND
    except NameError:
        CLMND = 0
    try:
        CLMFC = march.UnstructuredBlock2D.CLMFC
    except NameError:
        CLMFC = 0

    GEOMETRY_TABLE_NAMES = (
        'ndcrd', 'fccnd', 'fcnml', 'fcara', 'clcnd', 'clvol')
    META_TABLE_NAMES = (
        'fctpn', 'cltpn', 'clgrp')
    CONNECTIVITY_TABLE_NAMES = (
        'fcnds', 'fccls', 'clnds', 'clfcs')
    TABLE_NAMES = (
        GEOMETRY_TABLE_NAMES + META_TABLE_NAMES + CONNECTIVITY_TABLE_NAMES)

    def __init__(self, ndim=0, nnode=0, nface=0, ncell=0, nbound=0,
                 use_incenter=False, fpdtype=None, **kw
    ):
        # construct UnstructuredBlock.
        self.ndim = ndim
        if 3 == ndim:
            BlockClass = march.UnstructuredBlock3D
        else:
            BlockClass = march.UnstructuredBlock2D
        self._ustblk = BlockClass(nnode, nface, ncell, use_incenter)
        # serial number of the block.
        self.blkn = None
        # boundary conditions and boundary faces information.
        self.bclist = list()
        # group names.
        self.grpnames = list()
        # keep initialization sequence.
        super(UnstructuredBlock, self).__init__()

    def check_sanity(self):
        pass

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __getattr__(self, key):
        if "_ustblk" == key:
            raise RuntimeError("%s shouldn't be retrieved by this method" % key)
        return getattr(self._ustblk, key)

    @property
    def fpdtype(self):
        return np.dtype('float64')
    @property
    def fpdtypestr(self):
        return dependency.str_of(self.fpdtype)

    def __str__(self):
        return ', '.join([
            '[Block (%dD/%s): %d nodes'%(self.ndim,
                'incenter' if self.use_incenter else 'centroid',
                self.nnode),
            '%d faces (%d BC)'%(self.nface, self.nbound),
            '%d cells]'%self.ncell
        ])

    def check_simplex(self):
        """
        Check whether or not the block is composed purely by simplices, i.e.,
        triangles in two-dimensional space, or tetrahedra in three-dimensional
        space.

        @return: True if the block is composed purely by simplices; otherwise
            False.
        @rtype: bool
        """
        if self.ndim == 2:
            ret = (self.cltpn == 3).all()
        elif self.ndim == 3:
            ret = (self.cltpn == 5).all()
        else:
            raise ValueError('ndim %d != 2 or 3' % self.ndim)
        return ret

    def build_boundary(self, unspec_type=None, unspec_name="unspecified"):
        self._ustblk.set_bndvec([bc._data for bc in self.bclist])
        self._ustblk.build_boundary()
        if self._ustblk._bndvec_size != len(self.bclist):
            if unspec_type is None:
                unspec_type = boundcond.bctregy.unspecified
            bc = unspec_type()
            bc.name = unspec_name
            bc.sern = len(self.bclist)
            bc.blk = self
            bnddata = self._ustblk.get_bnddata(bc.sern)
            bc.facn = bnddata.facn
            bc.values = bnddata.values
            self.bclist.append(bc)


# compatibility

# FIXME: this should go into UnstructuredBlock C++ code.
elemtype = block_legacy.elemtype

USE_UNSTRUCTURED_BLOCK = True
if USE_UNSTRUCTURED_BLOCK:
    Block = UnstructuredBlock
else:
    Block = block_legacy.LegacyBlock
