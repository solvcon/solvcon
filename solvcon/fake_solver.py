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
A fake solver that uses :py:mod:`solvcon.fake_algorithm`.
"""

from .mesh_solver import MeshSolver

class FakeSolver(MeshSolver):
    """
    .. inheritance-diagram:: FakeSolver

    A fake solver that calculates trivial things to demonstrate the use of
    :py:class:`.mesh_solver.MeshSolver`.

    >>> # build a block before creating a solver.
    >>> from .block import Block
    >>> blk = Block(ndim=2, nnode=4, nface=6, ncell=3, nbound=3)
    >>> blk.ndcrd[:,:] = (0,0), (-1,-1), (1,-1), (0,1)
    >>> blk.cltpn[:] = 3
    >>> blk.clnds[:,:4] = (3, 0,1,2), (3, 0,2,3), (3, 0,3,1)
    >>> blk.build_interior()
    >>> blk.build_boundary()
    >>> blk.build_ghost()
    >>> # create a solver.
    >>> svr = FakeSolver(blk, neq=1)
    >>> # initialize the solver.
    >>> svr.sol.fill(0)
    >>> svr.soln.fill(0)
    >>> svr.dsol.fill(0)
    >>> svr.dsoln.fill(0)
    >>> # run the solver.
    >>> ret = svr.march(0.0, 0.01, 100)
    >>> # calculate and compare the results in soln.
    >>> from numpy import empty_like
    >>> soln = svr.soln[svr.ngstcell:,:]
    >>> clvol = empty_like(soln)
    >>> clvol.fill(0)
    >>> for iistep in range(200):
    ...     clvol[:,0] += svr.blk.clvol*svr.time_increment/2
    >>> (soln==clvol).all()
    True
    >>> # calculate and compare the results in dsoln.
    >>> dsoln = svr.dsoln[svr.ngstcell:,0,:]
    >>> clcnd = empty_like(dsoln)
    >>> clcnd.fill(0)
    >>> for iistep in range(200):
    ...     clcnd += svr.blk.clcnd*svr.time_increment/2
    >>> # compare.
    >>> (dsoln==clcnd).all()
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
        super(FakeSolver, self).__init__(blk, *args, **kw)
        # arrays.
        ndim = blk.ndim
        ncell = blk.ncell
        ngstcell = blk.ngstcell
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
        Create a :py:class:`.fake_algorithm.FakeAlgorithm` object.

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
        alg.setup_mesh(self.blk)
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
