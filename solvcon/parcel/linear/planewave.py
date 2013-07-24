# -*- coding: UTF-8 -*-
#
# Copyright (C) 2012-2013 Yung-Yu Chen <yyc@solvcon.net>.
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
Plane wave solution and initializer.
"""


import os
import math
import cPickle as pickle

import numpy as np

from solvcon import anchor
from solvcon import hook


class PlaneWaveSolution(object):
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
        asol.fill(0.0)
        self._calculate(asol)
        # difference.
        diff = self.svr.der['difference']
        diff[ngstcell:,:] = self.svr.soln[ngstcell:,:] - asol[ngstcell:,:]


class PlaneWaveHook(hook.MeshHook):
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
