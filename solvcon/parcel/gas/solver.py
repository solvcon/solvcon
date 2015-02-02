# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014, Yung-Yu Chen <yyc@solvcon.net>
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
Gas-dynamics solver.
"""


import numpy as np

import solvcon as sc

try: # for readthedocs to work.
    from . import _algorithm
except ImportError as e:
    import warnings
    warnings.warn(
        "solvcon.parcel.gas._algorithm isn't built; %s" % str(e.args),
        RuntimeWarning)


class GasSolver(sc.MeshSolver):
    """
    Spatial loops for the gas-dynamics solver.
    """

    _interface_init_ = ['cecnd', 'cevol', 'sfmrc']
    _solution_array_ = ['solt', 'sol', 'soln', 'dsol', 'dsoln']

    def __init__(self, blk, **kw):
        """
        Create a :py:class:`._algorithm.GasAlgorithm` object.

        >>> # create a valid solver as the test fixture.
        >>> from solvcon.testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> blk.clgrp.fill(0)
        >>> blk.grpnames.append('blank')
        >>> class SubSolver(GasSolver):
        ...     pass
        >>> svr = SubSolver(blk)
        >>> # number of equations.
        >>> svr.neq
        4
        >>> # valid GasAlgorithm.
        >>> svr.alg is not None
        True
        """
        self.neq = blk.ndim + 2
        super(GasSolver, self).__init__(blk, **kw)
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
        self.grpda = np.empty((self.ngroup, 1), dtype=fpdtype)
        nsca = kw.pop('nsca', 1)
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
        # algorithm object.
        alg = _algorithm.GasAlgorithm()
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
        super(GasSolver, self).init(**kw)
        self._debug_check_array('soln', 'dsoln')
        # prepare sub-face metric data.
        self.sfmrc.fill(0.0)
        self.alg.prepare_sf()
        self._debug_check_array('sfmrc')

    def provide(self):
        # super method.
        super(GasSolver, self).provide()
        self._debug_check_array('soln', 'dsoln')
        # density should not be zero.
        self._debug_check_array(np.abs(self.soln[:,0])<=self.ALMOST_ZERO)
        # fill group data array.
        self.grpda.fill(0)

    def apply_bc(self):
        super(GasSolver, self).apply_bc()
        self.call_non_interface_bc('soln')
        self.call_non_interface_bc('dsoln')

    ###########################################################################
    # Begin marching algorithm.
    _MMNAMES = sc.MeshSolver.new_method_list()

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

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
