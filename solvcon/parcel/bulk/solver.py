# -*- coding: UTF-8 -*-
#
# Copyright (c) 2013, Yung-Yu Chen <yyc@solvcon.net>, Po-Hsien Lin
# <lin.880@buckeyemail.osu.edu>
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


import os
import warnings

import numpy as np

from solvcon import solver
from solvcon import boundcond

try: # for readthedocs to work.
    from . import _algorithm
except ImportError:
    warnings.warn("solvcon.parcel.bulk._algorithm isn't built",
                  RuntimeWarning)


class BulkSolver(solver.MeshSolver):
    """This class controls the underneath algorithm :py:class:`BulkAlgorithm
    <._algorithm.BulkAlgorithm>`.
    """

    _interface_init_ = ['cecnd', 'cevol', 'sfmrc']
    _solution_array_ = ['solt', 'sol', 'soln', 'dsol', 'dsoln']

    def __init__(self, blk, **kw):
        """
        >>> # create a valid solver as the test fixture.
        >>> from solvcon import testing
        >>> from . import material
        >>> blk = testing.create_trivial_2d_blk()
        >>> blk.shclgrp.fill(0)
        >>> blk.grpnames.append('blank')
        >>> svrkw = dict(p0=1.0, rho0=1.0, fluids=[material.fluids.air])
        >>> svr = BulkSolver(blk, **svrkw)
        >>> svr.neq
        3
        """
        # meta data.
        self.neq = neq = 3 if blk.ndim == 2 else 4
        super(BulkSolver, self).__init__(blk, **kw)
        self.substep_run = 2
        ndim = blk.ndim
        ncell = blk.ncell
        ngstcell = blk.ngstcell
        fpdtype = 'float64'
        # scheme parameters.
        self.alpha = int(kw.pop('alpha', 1))
        self.sigma0 = int(kw.pop('sigma0', 3.0))
        self.taylor = float(kw.pop('taylor', 1))  # dirty hack.
        self.cnbfac = float(kw.pop('cnbfac', 1.0))  # dirty hack.
        self.sftfac = float(kw.pop('sftfac', 1.0))  # dirty hack.
        self.taumin = float(kw.pop('taumin', 0.0))
        self.tauscale = float(kw.pop('tauscale', 1.0))
        # physical parameters.
        self.p0 = float(kw.pop('p0'))
        self.rho0 = float(kw.pop('rho0'))
        self.fluids = kw.pop('fluids')
        assert len(self.fluids) == self.ngroup
        self.velocities = kw.pop('velocities', None)
        if None is self.velocities:
            self.velocities = [(0.0, 0.0, 0.0)] * self.ngroup
        assert len(self.velocities) == self.ngroup
        # dual mesh.
        self.cecnd = np.empty(
            (ngstcell+ncell, blk.CLMFC+1, ndim), dtype=fpdtype)
        self.cevol = np.empty(
            (ngstcell+ncell, blk.CLMFC+1), dtype=fpdtype)
        self.sfmrc = np.empty(
            (ncell, blk.CLMFC, blk.FCMND, 2, ndim), dtype=fpdtype)
        # parameters.
        self.grpda = np.empty((self.ngroup, 0), dtype=fpdtype)
        self.bulk = np.empty(ngstcell+ncell, dtype=fpdtype)
        self.dvisco = np.empty(ngstcell+ncell, dtype=fpdtype)
        # FIXME: remove amvec
        self.amvec = np.empty((ngstcell+ncell, 0, ndim), dtype=fpdtype)
        # solutions.
        self.sol = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.soln = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.solt = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.dsol = np.empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        self.dsoln = np.empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        self.stm = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.cfl = np.empty(ngstcell+ncell, dtype=fpdtype)
        self.ocfl = np.empty(ngstcell+ncell, dtype=fpdtype)
        # algorithm object.
        alg = _algorithm.BulkAlgorithm()
        alg.setup_mesh(blk)
        alg.setup_algorithm(self)
        self.alg = alg

    @property
    def gdlen(self):
        return self.grpda.shape[1]

    def init(self, **kw):
        # prepare ce metric data.
        self.cevol.fill(0.0)
        self.cecnd.fill(0.0)
        self.alg.prepare_ce()
        self._debug_check_array('cevol', 'cecnd')
        # super method.
        super(BulkSolver, self).init(**kw)
        self._debug_check_array('soln', 'dsoln')
        # prepare sub-face metric data.
        self.sfmrc.fill(0.0)
        self.alg.prepare_sf()
        self._debug_check_array('sfmrc')

    def provide(self):
        """
        >>> # create a valid solver as the test fixture.
        >>> from solvcon import testing
        >>> from . import material
        >>> blk = testing.create_trivial_2d_blk()
        >>> blk.shclgrp.fill(0)
        >>> blk.grpnames.append('blank')
        >>> svrkw = dict(p0=1.0, rho0=1.0, fluids=[material.fluids.air])
        >>> svr = BulkSolver(blk, **svrkw)
        >>> # initialize and provide the solver.
        >>> svr.init()
        >>> svr.provide()
        >>> (svr.bulk[:] == material.fluids.air.bulk).all()
        True
        >>> (svr.dvisco[:] == material.fluids.air.dvisco).all()
        True
        >>> (svr.soln[:,0] == material.fluids.air.rho).all()
        True
        """
        # super method.
        super(BulkSolver, self).provide()
        self._debug_check_array('soln', 'dsoln')
        # initialize array.
        self.soln.fill(self.ALMOST_ZERO)
        for key in ('dsoln', 'cfl', 'ocfl'):
            getattr(self, key).fill(0.0)
        for it, fluid in enumerate(self.fluids):
            self.bulk[self.blk.shclgrp==it] = fluid.bulk
            self.dvisco[self.blk.shclgrp==it] = fluid.dvisco
            self.soln[self.blk.shclgrp==it,0] = fluid.rho
            vel = self.velocities[it]
            for idim in range(self.ndim):
                val = fluid.rho * vel[idim]
                self.soln[self.blk.shclgrp==it,idim+1] = val
        self._debug_check_array('bulk', 'dvisco')
        # fill group data array.
        self.grpda.fill(0)

    def apply_bc(self):
        super(BulkSolver, self).apply_bc()
        self._debug_check_array('soln', 'dsoln')
        self.call_non_interface_bc('soln')
        self.call_non_interface_bc('dsoln')
        self._debug_check_array('soln', 'dsoln')

    ###########################################################################
    # Begin marching algorithm.
    _MMNAMES = solver.MeshSolver.new_method_list()

    @_MMNAMES.register
    def update(self, worker=None):
        self._debug_check_array('soln', 'dsoln')
        self.alg.update(self.time, self.time_increment)
        self.sol[:,:] = self.soln[:,:]
        self.dsol[:,:,:] = self.dsoln[:,:,:]
        self._debug_check_array('sol', 'dsol')

    @_MMNAMES.register
    def calcsolt(self, worker=None):
        self._debug_check_array('sol', 'dsol')
        self.alg.calc_solt()
        self._debug_check_array('solt')

    @_MMNAMES.register
    def calcsoln(self, worker=None):
        self._debug_check_array('sol', 'dsol')
        self.alg.calc_soln()
        if self.debug:
            self._debug_check_array('soln', 'dsoln')
            self._debug_check_array(self.soln[self.ngstcell:,0]<=0)

    @_MMNAMES.register
    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    @_MMNAMES.register
    def bcsoln(self, worker=None):
        self._debug_check_array('sol', 'dsol')
        self.call_non_interface_bc('soln')
        if self.debug:
            self._debug_check_array('soln', 'dsoln')
            self._debug_check_array(self.soln[self.ngstcell:,0]<=0)

    @_MMNAMES.register
    def calccfl(self, worker=None):
        self._debug_check_array('sol', 'dsol')
        self.alg.calc_cfl()
        self._debug_check_array('cfl', 'ocfl', 'soln', 'dsoln')

    @_MMNAMES.register
    def calcdsoln(self, worker=None):
        self._debug_check_array('sol', 'dsol')
        self.alg.calc_dsoln()
        self._debug_check_array('soln', 'dsoln')

    @_MMNAMES.register
    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)

    @_MMNAMES.register
    def bcdsoln(self, worker=None):
        self._debug_check_array('sol', 'dsol')
        self.call_non_interface_bc('dsoln')
        if self.debug:
            self._debug_check_array('soln', 'dsoln')
            self._debug_check_array(self.soln[self.ngstcell:,0]<=0)
    # End marching algorithm.
    ###########################################################################

    @staticmethod
    def get_reynolds(rhovel, dia, dvisco, out=None):
        """
        Calculate the Reynold number.
        
        It can be used as a simple utility function:

        >>> rho = 1.2250 # kg/m^3.
        >>> vel = 20.0 # m/s.
        >>> dia = 0.1 # meter.
        >>> dvisco = 1.983e-5 # kg/(m s).
        >>> BulkSolver.get_reynolds(rho*vel, dia, dvisco)
        123550.17650025214

        Or it can take solution arrays form the solver:

        >>> from solvcon import testing
        >>> from . import material
        >>> def get_solver(blk, rho, vel):
        ...     from solvcon import testing
        ...     blk.shclgrp.fill(0)
        ...     blk.grpnames.append('blank')
        ...     svrkw = dict(p0=1.0, rho0=1.0, fluids=[material.fluids.air])
        ...     svr = BulkSolver(blk, **svrkw)
        ...     svr.soln[1,:].fill(rho*vel)
        ...     return svr
        >>> svr = get_solver(testing.create_trivial_2d_blk(), rho, vel)
        >>> # the return is a brand new array.
        >>> BulkSolver.get_reynolds(svr.soln[1], dia, dvisco)
        array([ 123550.17650025,  123550.17650025,  123550.17650025])

        For memory efficiency, output array can be assigned:

        >>> svr = get_solver(testing.create_trivial_2d_blk(), rho, vel)
        >>> result = np.empty_like(svr.soln[1])
        >>> ret = BulkSolver.get_reynolds(svr.soln[1], dia, dvisco, out=result)
        >>> ret
        array([ 123550.17650025,  123550.17650025,  123550.17650025])
        >>> # the return is the preallocated array.
        >>> ret is result
        True
        """
        if None is out:
            if isinstance(rhovel, np.ndarray):
                out = rhovel.copy()
            else:
                out = rhovel
        else:
            out[:] = rhovel[:]
        out *= dia/dvisco
        return out


class BulkBC(boundcond.BC):
    """
    Base class for all boundary conditions of the bulk solver.
    """

    #: Ghost geometry calculator type.
    _ghostgeom_ = None

    def __init__(self, **kw):
        super(BulkBC, self).__init__(**kw)
        self.bcd = None

    @property
    def alg(self):
        return self.svr.alg

    def init(self, **kw):
        self.bcd = self.create_bcd()
        getattr(self.alg, 'ghostgeom_'+self._ghostgeom_)(self.bcd)


class BulkNonrefl(BulkBC):
    """
    General periodic boundary condition for sequential runs.
    """
    _ghostgeom_ = 'mirror'
    def soln(self):
        self.alg.bound_nonrefl_soln(self.bcd)
    def dsoln(self):
        self.alg.bound_nonrefl_dsoln(self.bcd)

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
