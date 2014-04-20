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
uses :py:mod:`solvcon./`.
"""


__all__ = [
    'VewaveSolver', 'VewavePeriodic', 'VewaveBC', 'VewaveNonRefl',
    'VewaveLongSineX',
]


import os
import warnings

import numpy as np

from solvcon import solver
from solvcon import boundcond

try: # for readthedocs to work.
    from . import _algorithm
except ImportError:
    warnings.warn("solvcon.parcel.vewave._algorithm isn't built",
                  RuntimeWarning)


class VewaveSolver(solver.MeshSolver):
    """This class controls the underneath algorithm :py:class:`VewaveAlgorithm
    <._algorithm.VewaveAlgorithm>`.
    """

    _interface_init_ = ['cecnd', 'cevol']
    _solution_array_ = ['solt', 'sol', 'soln', 'dsol', 'dsoln']

    def __init__(self, blk, mtrldict, **kw):
        """
        A linear solver needs a :py:class:`Block <solvcon.block.Block>` and a
        dictionary for mapping names to :py:class:`~.material.Material`:

        >>> from solvcon import testing
        >>> blk = testing.create_trivial_2d_blk()
        >>> blk.clgrp.fill(0)
        >>> blk.grpnames.append('blank')
        >>> svr = VewaveSolver(blk, {}) # doctest: +ELLIPSIS
        """
        super(VewaveSolver, self).__init__(blk, **kw)
        # meta data.
        ndim = blk.ndim
        ncell = blk.ncell
        ngstcell = blk.ngstcell
        self.clcnd = blk.clcnd
        fpdtype = 'float64'
        self.neq = self.determine_neq(ndim)
        #: A :py:class:`dict` that maps names to :py:class:`Material
        #: <.material.Material>` object.
        self.mtrldict = mtrldict if mtrldict else {}
        #: A :py:class:`list` of all :py:class:`Material <.material.Material>`
        #: objects.
        self.mtrllist = None
        # scheme parameters.
        self.substep_run = 2
        self.alpha = int(kw.pop('alpha', 1))
        self.sigma0 = int(kw.pop('sigma0', 3.0))
        self.taylor = float(kw.pop('taylor', 1))  # dirty hack.
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
        nsca = kw.pop('nsca', 4)
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
        alg = _algorithm.VewaveAlgorithm()
        alg.setup_mesh(blk)
        alg.setup_algorithm(self)
        self.alg = alg

    @staticmethod
    def determine_neq(ndim):
        return 45

    @property
    def gdlen(self):
        return self.determine_neq(self.ndim)**2 * self.ndim

    def init(self, **kw):
        self.cevol.fill(0.0)
        self.cecnd.fill(0.0)
        self.alg.prepare_ce()
        super(VewaveSolver, self).init(**kw)
        self.sfmrc.fill(0.0)
        self.alg.prepare_sf()
        self._debug_check_array('sfmrc')

    def provide(self):
        super(VewaveSolver, self).provide()
        self.grpda.fill(0)

    def preloop(self):
        # fill group data array.
        self.mtrllist = self._build_mtrllist(self.grpnames, self.mtrldict)
        for igrp in range(len(self.grpnames)):
            mtrl = self.mtrllist[igrp]
            jaco = self.grpda[igrp].reshape(self.neq, self.neq, self.ndim)
            mjacos = mtrl.get_jacos(self.ndim)
            for idm in range(self.ndim):
                jaco[:,:,idm] = mjacos[idm,:,:]
        # pre-calculate CFL.
        self.alg.calc_cfl()
        self.ocfl[:] = self.cfl[:]
        # super method.
        super(VewaveSolver, self).preloop()

    @staticmethod
    def _build_mtrllist(grpnames, mtrldict):
        """
        Build the material list out of the mapping dict.

        @type grpnames: list
        @param mtrldict: the map from names to material objects.
        @type mtrldict: dict
        @return: the list of material object.
        @rtype: Material
        """
        mtrllist = list()
        default_mtuple = mtrldict.get(None, None)
        for grpname in grpnames:
            try:
                mtrl = mtrldict.get(grpname, default_mtuple)
            except KeyError, e:
                args = e.args[:]
                args.append('no material named %s in mtrldict'%grpname)
                e.args = args
                raise
            mtrllist.append(mtrl)
        return mtrllist

    def apply_bc(self):
        super(VewaveSolver, self).apply_bc()
        self.call_non_interface_bc('soln')
        self.call_non_interface_bc('dsoln')

    ###########################################################################
    # Begin marching algorithm.
    _MMNAMES = solver.MeshSolver.new_method_list()

    @_MMNAMES.register
    def update(self, worker=None):
        self.alg.update(self.time, self.time_increment)
        self.sol[:,:] = self.soln[:,:]
        self.dsol[:,:,:] = self.dsoln[:,:,:]

    @_MMNAMES.register
    def calcsolt(self, worker=None):
        #self.create_alg().calc_solt()
        self.alg.calc_solt()

    @_MMNAMES.register
    def calcsoln(self, worker=None):
        #self.create_alg().calc_soln()
        self.alg.calc_soln()

    @_MMNAMES.register
    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    @_MMNAMES.register
    def bcsoln(self, worker=None):
        self.call_non_interface_bc('soln')

    @_MMNAMES.register
    def calcdsoln(self, worker=None):
        self.alg.calc_dsoln()

    @_MMNAMES.register
    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)

    @_MMNAMES.register
    def bcdsoln(self, worker=None):
        self.call_non_interface_bc('dsoln')
    # End marching algorithm.
    ###########################################################################


class VewavePeriodic(boundcond.periodic):
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


class VewaveBC(boundcond.BC):
    #: Ghost geometry calculator type.
    _ghostgeom_ = None

    @property
    def alg(self):
        return self.svr.alg

    def init(self, **kw):
        getattr(self.alg, 'ghostgeom_'+self._ghostgeom_)(self.facn)


class VewaveNonRefl(VewaveBC):
    _ghostgeom_ = 'mirror'
    def soln(self):
        self.alg.bound_nonrefl_soln(self.facn)
    def dsoln(self):
        self.alg.bound_nonrefl_dsoln(self.facn)
    

class VewaveLongSineX(VewaveBC):
    """
    Provide longitudinal wave in x-direction.
    Wave is a sinusoidal wave.
    """
    _ghostgeom_ = 'mirror'
    def soln(self):
        self.alg.bound_longsinex_soln(self.facn)
    def dsoln(self):
        self.alg.bound_longsinex_dsoln(self.facn)

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
