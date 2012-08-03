# -*- coding: UTF-8 -*-
#
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

"""
A fake solver that uses the :py:mod:`solvcon.fake_algorithm`.
"""

from .solver import Solver

class FakeSolver(Solver):
    """
    >>> from .block import Block
    >>> blk = Block(ndim=2, nnode=4, nface=6, ncell=3, nbound=3)
    >>> blk.ndcrd[:,:] = (0,0), (-1,-1), (1,-1), (0,1)
    >>> blk.cltpn[:] = 3
    >>> blk.clnds[:,:4] = (3, 0,1,2), (3, 0,2,3), (3, 0,3,1)
    >>> blk.build_interior()
    >>> blk.build_boundary()
    >>> blk.build_ghost()
    >>> svr = FakeSolver(blk, neq=1)
    >>> # initialize.
    >>> svr.sol.fill(0)
    >>> svr.dsol.fill(0)
    >>> ret = svr.march(0.0, 0.01, 100)
    >>> # calculate and compare soln.
    >>> from numpy import empty_like
    >>> soln = svr.soln[svr.ngstcell:,:]
    >>> arr = empty_like(soln)
    >>> arr.fill(0)
    >>> for iistep in range(200):
    ...     arr[:,0] += svr.clvol[svr.ngstcell:]*svr.time_increment/2
    >>> (soln==arr).all()
    True
    """

    _interface_init_ = ['cecnd', 'cevol']

    def __init__(self, blk, *args, **kw):
        """
        >>> from .block import Block
        >>> blk = Block(ndim=2, nnode=4, nface=6, ncell=3, nbound=3)
        >>> blk.ndcrd[:,:] = (0,0), (-1,-1), (1,-1), (0,1)
        >>> blk.cltpn[:] = 3
        >>> blk.clnds[:,:4] = (3, 0,1,2), (3, 0,2,3), (3, 0,3,1)
        >>> blk.build_interior()
        >>> blk.build_boundary()
        >>> blk.build_ghost()
        >>> svr = FakeSolver(blk, neq=1)
        """
        from numpy import empty
        self.blk = blk
        super(FakeSolver, self).__init__(blk, *args, **kw)
        # arrays.
        ndim = self.ndim
        ncell = self.ncell
        ngstcell = self.ngstcell
        ## solutions.
        neq = self.neq
        self.sol = empty((ngstcell+ncell, neq), dtype=self.fpdtype)
        self.soln = empty((ngstcell+ncell, neq), dtype=self.fpdtype)
        self.dsol = empty((ngstcell+ncell, neq, ndim), dtype=self.fpdtype)
        self.dsoln = empty((ngstcell+ncell, neq, ndim), dtype=self.fpdtype)
        ## metrics.
        self.cecnd = empty(
            (ngstcell+ncell, self.CLMFC+1, ndim), dtype=self.fpdtype)
        self.cevol = empty((ngstcell+ncell, self.CLMFC+1), dtype=self.fpdtype)

    def create_alg(self):
        """
        >>> from .block import Block
        >>> blk = Block(ndim=2, nnode=4, nface=6, ncell=3, nbound=3)
        >>> blk.ndcrd[:,:] = (0,0), (-1,-1), (1,-1), (0,1)
        >>> blk.cltpn[:] = 3
        >>> blk.clnds[:,:4] = (3, 0,1,2), (3, 0,2,3), (3, 0,3,1)
        >>> blk.build_interior()
        >>> blk.build_boundary()
        >>> blk.build_ghost()
        >>> svr = FakeSolver(blk, neq=1)
        >>> alg = svr.create_alg()
        """
        from .fake_algorithm import FakeAlgorithm
        alg = FakeAlgorithm()
        # mesh.
        alg.ndim = self.blk.ndim
        alg.nnode = self.blk.nnode
        alg.nface = self.blk.nface
        alg.ncell = self.blk.ncell
        alg.ngstnode = self.blk.ngstnode
        alg.ngstface = self.blk.ngstface
        alg.ngstcell = self.blk.ngstcell
        alg.ndcrd = self.blk.ndcrd
        alg.fccnd = self.blk.fccnd
        alg.fcnml = self.blk.fcnml
        alg.fcara = self.blk.fcara
        alg.clcnd = self.blk.clcnd
        alg.clvol = self.blk.clvol
        alg.fctpn = self.blk.fctpn
        alg.cltpn = self.blk.cltpn
        alg.clgrp = self.blk.clgrp
        alg.fcnds = self.blk.fcnds
        alg.fccls = self.blk.fccls
        alg.clnds = self.blk.clnds
        alg.clfcs = self.blk.clfcs
        # solver.
        alg.setup_algorithm(self)
        return alg

    ##################################################
    # marching algorithm.
    ##################################################
    MMNAMES = list()
    MMNAMES.append('update')
    def update(self, worker=None):
        self.sol[:,:] = self.soln[:,:]
        self.dsol[:,:,:] = self.dsoln[:,:,:]

    MMNAMES.append('calcsoln')
    def calcsoln(self, worker=None):
        self.create_alg().calc_soln()

    MMNAMES.append('ibcsoln')
    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    MMNAMES.append('calccfl')
    def calccfl(self, worker=None):
        self.marchret = -2.0

    MMNAMES.append('calcdsoln')
    def calcdsoln(self, worker=None):
        self.create_alg().calc_dsoln()

    MMNAMES.append('ibcdsoln')
    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)
