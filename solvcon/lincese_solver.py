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
A two-/three-dimensional, second order CESE solver for generic linear PDEs. It
uses :py:mod:`solvcon.lincese_algorithm`.
"""

from .mesh_solver import MeshSolver

class LinceseSolver(MeshSolver):
    """
    .. inheritance-diagram:: LinceseSolver
    """

    _interface_init_ = ['cecnd', 'cevol']

    def __init__(self, blk, *args, **kw):
        from numpy import empty
        # meta data.
        self.neq = kw.pop('neq')
        super(LinceseSolver, self).__init__(blk, *args, **kw)
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
        Create a :py:class:`.lincese_algorithm.LinceseAlgorithm` object.

        >>> from .testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> svr = LinceseSolver(blk, neq=1)
        >>> alg = svr.create_alg()
        """
        from .lincese_algorithm import LinceseAlgorithm
        alg = LinceseAlgorithm()
        alg.setup_mesh(self.blk)
        alg.setup_algorithm(self)
        return alg

    ###########################################################################
    # marching algorithm.
    ###########################################################################
    MMNAMES = MeshSolver.new_method_list()

    @MMNAMES.register
    def update(self, worker=None):
        self.sol[:,:] = self.soln[:,:]
        self.dsol[:,:,:] = self.dsoln[:,:,:]

    @MMNAMES.register
    def calcsolt(self, worker=None):
        self.create_alg().calc_solt()

    @MMNAMES.register
    def calcsoln(self, worker=None):
        self.create_alg().calc_soln()

    @MMNAMES.register
    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    @MMNAMES.register
    def calccfl(self, worker=None):
        self.marchret = -2.0

    @MMNAMES.register
    def calcdsoln(self, worker=None):
        self.create_alg().calc_dsoln()

    @MMNAMES.register
    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
