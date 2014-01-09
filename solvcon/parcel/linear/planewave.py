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
Plane wave solution and initializer.
"""


import os
import math
import cPickle as pickle

import numpy as np

from solvcon import anchor
from solvcon import hook


class PlaneWaveSolution(object):
    # FIXME: THIS GUY NEEDS UNIT TEST.  Debugging in these classes is like pain
    # in the ass.
    def __init__(self, **kw):
        wvec = kw['wvec']
        ctr = kw['ctr']
        amp = kw['amp']
        assert len(wvec) == len(ctr)
        # calculate eigenvalues and eigenvectors.
        evl, evc = self._calc_eigen(**kw)
        # store data to self.
        self.amp = evc * (amp / np.sqrt((evc**2).sum()))
        self.ctr = ctr
        self.wvec = wvec
        self.afreq = evl * np.sqrt((wvec**2).sum())
        self.wsp = evl

    def _calc_eigen(self, *args, **kw):
        """Calculate and return a :py:class:`tuple` for eigenvalues and
        eigenvectors.  This method needs to be subclassed.
        """
        raise NotImplementedError

    def __call__(self, svr, asol, adsol):
        svr.create_alg().calc_planewave(
            asol, adsol, self.amp, self.ctr, self.wvec, self.afreq)


class PlaneWaveAnchor(anchor.MeshAnchor):
    """Use :py:class:`PlaneWaveSolution` to calculate plane-wave solutions for
    :py:class:`LinearSolver <.solver.LinearSolver>`.
    """

    # FIXME: THIS GUY NEEDS UNIT TEST.  The coupling with Hook isn't really
    # easy to debug.
    def __init__(self, svr, planewaves=None, **kw):
        assert None is not planewaves
        #: Sequence of :py:class:`PlaneWaveSolution` objects.
        self.planewaves = planewaves
        super(PlaneWaveAnchor, self).__init__(svr, **kw)

    def _calculate(self, asol):
        for pw in self.planewaves:
            pw(self.svr, asol, self.adsol)

    def provide(self):
        ngstcell = self.svr.blk.ngstcell
        nacell = self.svr.blk.ncell + ngstcell
        # plane wave solution.
        asol = self.svr.der['analytical'] = np.empty(
            (nacell, self.svr.neq), dtype='float64')
        adsol = self.adsol = np.empty(
            (nacell, self.svr.neq, self.svr.blk.ndim),
            dtype='float64')
        asol.fill(0.0)
        self._calculate(asol)
        self.svr.soln[ngstcell:,:] = asol[ngstcell:,:]
        self.svr.dsoln[ngstcell:,:,:] = adsol[ngstcell:,:,:]
        # difference.
        diff = self.svr.der['difference'] = np.empty(
            (nacell, self.svr.neq), dtype='float64')
        diff[ngstcell:,:] = self.svr.soln[ngstcell:,:] - asol[ngstcell:,:]

    def postfull(self):
        ngstcell = self.svr.ngstcell
        # plane wave solution.
        asol = self.svr.der['analytical']
        self._calculate(asol)
        # difference.
        diff = self.svr.der['difference']
        diff[ngstcell:,:] = self.svr.soln[ngstcell:,:] - asol[ngstcell:,:]


class PlaneWaveHook(hook.MeshHook):
    # FIXME: THIS GUY NEEDS UNIT TEST.  The coupling with Anchor isn't really
    # easy to debug.
    def __init__(self, svr, planewaves=None, **kw):
        assert None is not planewaves
        #: Sequence of :py:class:`PlaneWaveSolution` objects.
        self.planewaves = planewaves
        #: A :py:class:`dict` holding the calculated norm.
        self.norm = dict()
        super(PlaneWaveHook, self).__init__(svr, **kw)

    def drop_anchor(self, svr):
        svr.runanchors.append(
            PlaneWaveAnchor(svr, planewaves=self.planewaves)
        )

    def _calculate(self):
        neq = self.cse.execution.neq
        var = self.cse.execution.var
        asol = self._collect_interior(
            'analytical', inder=True, consider_ghost=True)
        diff = self._collect_interior(
            'difference', inder=True, consider_ghost=True)
        norm_Linf = np.empty(neq, dtype='float64')
        norm_L2 = np.empty(neq, dtype='float64')
        clvol = self.blk.clvol
        for it in range(neq):
            norm_Linf[it] = np.abs(diff[:,it]).max()
            norm_L2[it] = np.sqrt((diff[:,it]**2*clvol).sum())
        self.norm['Linf'] = norm_Linf
        self.norm['L2'] = norm_L2

    def preloop(self):
        self.postmarch()
        for ipw in range(len(self.planewaves)):
            pw = self.planewaves[ipw]
            self.info("planewave[%d]:\n" % ipw)
            self.info("  c = %g, omega = %g, T = %.15e\n" % (
                pw.wsp, pw.afreq, 2*np.pi/pw.afreq))

    def postmarch(self):
        psteps = self.psteps
        istep = self.cse.execution.step_current
        if istep%psteps == 0:
            self._calculate()

    def postloop(self):
        fname = '%s_norm.pickle' % self.cse.io.basefn
        fname = os.path.join(self.cse.io.basedir, fname)
        pickle.dump(self.norm, open(fname, 'wb'), -1)
        self.info('Linf norm in velocity:\n')
        self.info('  %e, %e, %e\n' % tuple(self.norm['Linf'][:3]))
        self.info('L2 norm in velocity:\n')
        self.info('  %e, %e, %e\n' % tuple(self.norm['L2'][:3]))

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
