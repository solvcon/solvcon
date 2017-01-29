# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
C++-based Gas-dynamics solver.
"""


from __future__ import absolute_import, division, print_function


import numpy as np

import solvcon as sc
from solvcon import march


class GasPlusSolver(sc.MeshSolver):
    """
    Spatial loops for the gas-dynamics solver.
    """

    _interface_init_ = ('cecnd', 'cevol', 'sfmrc')
    _solution_array_ = ('solt', 'sol', 'soln', 'dsol', 'dsoln')

    def __init__(self, blk, **kw):
        self.neq = blk.ndim + 2
        super(GasPlusSolver, self).__init__(blk, **kw)
        self.substep_run = 2
        ndim = blk.ndim
        ncell = blk.ncell
        ngstcell = blk.ngstcell
        fpdtype = 'float64'
        # algorithm object.
        solver_type = getattr(march.gas, "Solver%dD"%blk.ndim)
        alg = solver_type(blk._ustblk)
        self.alg = alg
        # parameters.
        self.grpda = np.empty((self.ngroup, 1), dtype=fpdtype)
        # set arrays.
        for name in (('amsca',)
                   + self._solution_array_ + ('stm', 'cfl', 'ocfl')):
            setattr(self, name, getattr(self, 'tb'+name).F)

    @property
    def gdlen(self):
        return self.grpda.shape[1]

    @property
    def sigma0(self):
        return self.alg.sigma0
    @sigma0.setter
    def sigma0(self, value):
        self.alg.sigma0 = value

    @property
    def taumin(self):
        return self.alg.taumin
    @taumin.setter
    def taumin(self, value):
        self.alg.taumin = value

    @property
    def tauscale(self):
        return self.alg.tauscale
    @tauscale.setter
    def tauscale(self, value):
        self.alg.tauscale = value

    @property
    def tbamsca(self):
        return self.alg.amsca

    @property
    def tbsol(self):
        return self.alg.sol

    @property
    def tbsoln(self):
        return self.alg.soln

    @property
    def tbsolt(self):
        return self.alg.solt

    @property
    def tbdsol(self):
        return self.alg.dsol

    @property
    def tbdsoln(self):
        return self.alg.dsoln

    @property
    def tbstm(self):
        return self.alg.stm

    @property
    def tbcfl(self):
        return self.alg.cfl

    @property
    def tbocfl(self):
        return self.alg.ocfl

    def init(self, **kw):
        # super method.
        super(GasPlusSolver, self).init(**kw)
        self._debug_check_array('soln', 'dsoln')

    def provide(self):
        # super method.
        super(GasPlusSolver, self).provide()
        self._debug_check_array('soln', 'dsoln')
        # density should not be zero.
        self._debug_check_array(np.abs(self.soln[:,0])<=self.ALMOST_ZERO)
        # fill group data array.
        self.grpda.fill(0)

    def apply_bc(self):
        super(GasPlusSolver, self).apply_bc()
        self.call_non_interface_bc('soln')
        self.call_non_interface_bc('dsoln')

    ###########################################################################
    # Begin marching algorithm.
    @sc.MeshSolver.register_marcher
    def update(self, worker=None):
        self._debug_check_array('soln', 'dsoln')
        self.alg.update(self.time, self.time_increment)
        self._debug_check_array('sol', 'dsol')

    @sc.MeshSolver.register_marcher
    def calcsolt(self, worker=None):
        self._debug_check_array('sol', 'dsol')
        self.alg.calc_solt()
        self._debug_check_array('solt')

    @sc.MeshSolver.register_marcher
    def calcsoln(self, worker=None):
        self._debug_check_array('sol', 'dsol')
        self.alg.calc_soln()
        if self.debug:
            self._debug_check_array('soln', 'dsoln')
            self._debug_check_array(self.soln[self.ngstcell:,0]<=0)

    @sc.MeshSolver.register_marcher
    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    @sc.MeshSolver.register_marcher
    def bcsoln(self, worker=None):
        self._debug_check_array('sol', 'dsol')
        self.call_non_interface_bc('soln')
        if self.debug:
            self._debug_check_array('soln', 'dsoln')
            self._debug_check_array(self.soln[self.ngstcell:,0]<=0)

    @sc.MeshSolver.register_marcher
    def calccfl(self, worker=None):
        self._debug_check_array('sol', 'dsol')
        self.alg.calc_cfl()
        self._debug_check_array('cfl', 'ocfl', 'soln', 'dsoln')

    @sc.MeshSolver.register_marcher
    def calcdsoln(self, worker=None):
        self._debug_check_array('sol', 'dsol')
        self.alg.calc_dsoln()
        self._debug_check_array('soln', 'dsoln')

    @sc.MeshSolver.register_marcher
    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)

    @sc.MeshSolver.register_marcher
    def bcdsoln(self, worker=None):
        self._debug_check_array('sol', 'dsol')
        self.call_non_interface_bc('dsoln')
        if self.debug:
            self._debug_check_array('soln', 'dsoln')
            self._debug_check_array(self.soln[self.ngstcell:,0]<=0)
    # End marching algorithm.
    ###########################################################################

# vim: set ff=unix fenc=utf8 ft=python nobomb et sw=4 ts=4 tw=79:
