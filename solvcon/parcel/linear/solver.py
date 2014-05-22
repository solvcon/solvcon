# -*- coding: UTF-8 -*-
#
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

"""
A two-/three-dimensional, second order CESE solver for generic linear PDEs. It
uses :py:mod:`solvcon._algorithm`.
"""


import os
import warnings

import numpy as np

from solvcon import solver
from solvcon import boundcond

try: # for readthedocs to work.
    from . import _algorithm
except ImportError as e:
    warnings.warn(
        "solvcon.parcel.linear._algorithm isn't built; %s" % str(e.args),
        RuntimeWarning)


class LinearSolver(solver.MeshSolver):
    """This class controls the underneath algorithm :py:class:`LinearAlgorithm
    <._algorithm.LinearAlgorithm>`.
    """

    _interface_init_ = ['cecnd', 'cevol', 'sfmrc']
    _solution_array_ = ['solt', 'sol', 'soln', 'dsol', 'dsoln']

    def __init__(self, blk, **kw):
        """
        A linear solver needs a :py:class:`Block <solvcon.block.Block>` having
        at least one group:

        >>> from solvcon.testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> blk.clgrp.fill(0)
        >>> blk.grpnames.append('blank')

        A linear solver can't be instantiated directly:

        >>> svr = LinearSolver(blk, neq=1) # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: ...

        To instantiate the linear solver, at least :py:attr:`gdlen` needs to be
        implemented:

        >>> class SubSolver(LinearSolver):
        ...     @property
        ...     def gdlen(self):
        ...         return 1
        >>> svr = SubSolver(blk, neq=1)
        """
        # meta data.
        self.neq = kw.pop('neq')
        super(LinearSolver, self).__init__(blk, **kw)
        self.substep_run = 2
        ndim = blk.ndim
        ncell = blk.ncell
        ngstcell = blk.ngstcell
        fpdtype = 'float64'
        # scheme parameters.
        self.alpha = int(kw.pop('alpha', 0))
        self.sigma0 = int(kw.pop('sigma0', 3.0))
        self.taylor = float(kw.pop('taylor', 1.0))  # dirty hack.
        self.cnbfac = float(kw.pop('cnbfac', 1.0))  # dirty hack.
        self.sftfac = float(kw.pop('sftfac', 1.0))  # dirty hack.
        self.taumin = float(kw.pop('taumin', 0.0))
        self.tauscale = float(kw.pop('tauscale', 1.0))
        # dual mesh.
        self.cecnd = np.empty(
            (ngstcell+ncell, blk.CLMFC+1, ndim), dtype=fpdtype)
        self.cevol = np.empty(
            (ngstcell+ncell, blk.CLMFC+1), dtype=fpdtype)
        self.sfmrc = np.empty((ncell, blk.CLMFC, blk.FCMND, 2, ndim),
            dtype=fpdtype)
        # parameters.
        self.grpda = np.empty((self.ngroup, self.gdlen), dtype=fpdtype)
        nsca = kw.pop('nsca', 0)
        nvec = kw.pop('nvec', 0)
        self.amsca = np.empty((ngstcell+ncell, nsca), dtype=fpdtype)
        self.amvec = np.empty((ngstcell+ncell, nvec, ndim), dtype=fpdtype)
        # solutions.
        neq = self.neq
        self.sol = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.soln = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.solt = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.dsol = np.empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        self.dsoln = np.empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        self.stm = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.cfl = np.empty(ngstcell+ncell, dtype=fpdtype)
        self.ocfl = np.empty(ngstcell+ncell, dtype=fpdtype)

    @property
    def gdlen(self):
        return None

    def create_alg(self):
        """
        Create a :py:class:`._algorithm.LinearAlgorithm` object.

        >>> # create a valid solver as the test fixture.
        >>> from solvcon.testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> blk.clgrp.fill(0)
        >>> blk.grpnames.append('blank')
        >>> class SubSolver(LinearSolver):
        ...     @property
        ...     def gdlen(self):
        ...         return 1
        >>> svr = SubSolver(blk, neq=1)

        Create an associated algorithm object is straight-forward:

        >>> alg = svr.create_alg()
        """
        alg = _algorithm.LinearAlgorithm()
        alg.setup_mesh(self.blk)
        alg.setup_algorithm(self)
        return alg

    def init(self, **kw):
        self.create_alg().prepare_ce()
        super(LinearSolver, self).init(**kw)
        self.create_alg().prepare_sf()

    def provide(self):
        # fill group data array.
        self._make_grpda()
        # pre-calculate CFL.
        self.create_alg().calc_cfl()
        self.ocfl[:] = self.cfl[:]
        # super method.
        super(LinearSolver, self).provide()

    def apply_bc(self):
        super(LinearSolver, self).apply_bc()
        self.call_non_interface_bc('soln')
        self.call_non_interface_bc('dsoln')

    def _make_grpda(self):
        raise NotImplementedError

    ###########################################################################
    # Begin marching algorithm.
    _MMNAMES = solver.MeshSolver.new_method_list()

    @_MMNAMES.register
    def update(self, worker=None):
        self.sol[:,:] = self.soln[:,:]
        self.dsol[:,:,:] = self.dsoln[:,:,:]

    @_MMNAMES.register
    def calcsolt(self, worker=None):
        self.create_alg().calc_solt()

    @_MMNAMES.register
    def calcsoln(self, worker=None):
        self.create_alg().calc_soln()

    @_MMNAMES.register
    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    @_MMNAMES.register
    def bcsoln(self, worker=None):
        self.call_non_interface_bc('soln')

    @_MMNAMES.register
    def calcdsoln(self, worker=None):
        self.create_alg().calc_dsoln()

    @_MMNAMES.register
    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)

    @_MMNAMES.register
    def bcdsoln(self, worker=None):
        self.call_non_interface_bc('dsoln')
    # End marching algorithm.
    ###########################################################################


class LinearPeriodic(boundcond.periodic):
    """
    General periodic boundary condition for sequential runs.
    """
    def init(self, **kw):
        svr = self.svr
        blk = svr.blk
        ngstcell = blk.ngstcell
        ngstface = blk.ngstface
        facn = self.facn
        slctm = self.rclp[:,0] + ngstcell
        slctr = self.rclp[:,1] + ngstcell
        # move coordinates.
        shf = svr.cecnd[slctr,0,:] - blk.shfccnd[facn[:,2]+ngstface,:]
        svr.cecnd[slctm,0,:] = blk.shfccnd[facn[:,0]+ngstface,:] + shf

    def soln(self):
        svr = self.svr
        blk = svr.blk
        slctm = self.rclp[:,0] + blk.ngstcell
        slctr = self.rclp[:,1] + blk.ngstcell
        svr.soln[slctm,:] = svr.soln[slctr,:]

    def dsoln(self):
        svr = self.svr
        blk = svr.blk
        slctm = self.rclp[:,0] + blk.ngstcell
        slctr = self.rclp[:,1] + blk.ngstcell
        svr.dsoln[slctm,:,:] = svr.dsoln[slctr,:,:]


# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
