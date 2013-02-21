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

from .solver import MeshSolver

class FakeSolver(MeshSolver):
    """
    .. inheritance-diagram:: FakeSolver

    A fake solver that calculates trivial things to demonstrate the use of
    :py:class:`.mesh_solver.MeshSolver`.

    >>> # build a block before creating a solver.
    >>> from .testing import create_trivial_2d_blk
    >>> blk = create_trivial_2d_blk()
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
    >>> soln = svr.soln[svr.blk.ngstcell:,:]
    >>> clvol = empty_like(soln)
    >>> clvol.fill(0)
    >>> for iistep in range(200):
    ...     clvol[:,0] += svr.blk.clvol*svr.time_increment/2
    >>> (soln==clvol).all()
    True
    >>> # calculate and compare the results in dsoln.
    >>> dsoln = svr.dsoln[svr.blk.ngstcell:,0,:]
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
        >>> from .testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> svr = FakeSolver(blk, neq=1)
        """
        from numpy import empty
        # meta data.
        self.neq = kw.pop('neq')
        super(FakeSolver, self).__init__(blk, *args, **kw)
        self.substep_run = 2
        # arrays.
        ndim = blk.ndim
        ncell = blk.ncell
        ngstcell = blk.ngstcell
        fpdtype = 'float64'
        ## solutions.
        neq = self.neq
        self.sol = empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.soln = empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.dsol = empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        self.dsoln = empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        ## metrics.
        self.cecnd = empty(
            (ngstcell+ncell, self.blk.CLMFC+1, ndim), dtype=fpdtype)
        self.cevol = empty((ngstcell+ncell, self.blk.CLMFC+1), dtype=fpdtype)

    def create_alg(self):
        """
        Create a :py:class:`.fake_algorithm.FakeAlgorithm` object.

        >>> from .testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> svr = FakeSolver(blk, neq=1)
        >>> alg = svr.create_alg()
        """
        from .fake_algorithm import FakeAlgorithm
        alg = FakeAlgorithm()
        alg.setup_mesh(self.blk)
        alg.setup_algorithm(self)
        return alg

    ###########################################################################
    # marching algorithm.
    ###########################################################################
    _MMNAMES = MeshSolver.new_method_list()

    @_MMNAMES.register
    def update(self, worker=None):
        """
        Update solution arrays.

        >>> from .testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> svr = FakeSolver(blk, neq=1)
        >>> # initialize with different solution arrays.
        >>> svr.sol.fill(0)
        >>> svr.soln.fill(2)
        >>> svr.dsol.fill(0)
        >>> svr.dsoln.fill(2)
        >>> (svr.sol != svr.soln).all()
        True
        >>> (svr.dsol != svr.dsoln).all()
        True
        >>> # update and then solution arrays become the same.
        >>> svr.update()
        >>> (svr.sol == svr.soln).all()
        True
        >>> (svr.dsol == svr.dsoln).all()
        True
        """
        self.sol[:,:] = self.soln[:,:]
        self.dsol[:,:,:] = self.dsoln[:,:,:]

    @_MMNAMES.register
    def calcsoln(self, worker=None):
        self.create_alg().calc_soln()

    @_MMNAMES.register
    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    @_MMNAMES.register
    def calccfl(self, worker=None):
        self.marchret = -2.0

    @_MMNAMES.register
    def calcdsoln(self, worker=None):
        self.create_alg().calc_dsoln()

    @_MMNAMES.register
    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
