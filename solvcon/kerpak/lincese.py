# -*- coding: UTF-8 -*-
#
# Copyright (C) 2010-2011 Yung-Yu Chen <yyc@solvcon.net>.
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
CESE solver specialized for general linear equations.
"""

from solvcon.gendata import SingleAssignDict, AttributeDict
from solvcon.anchor import Anchor
from solvcon.hook import BlockHook
from solvcon.kerpak.cese import CeseSolver, CeseCase

###############################################################################
# Solver.
###############################################################################

class LinceseSolver(CeseSolver):
    """
    Basic linear CESE solver.

    @ivar cfldt: the time_increment for CFL calculation at boundcond.
    @itype cfldt: float
    @ivar cflmax: the maximum CFL number.
    @itype cflmax: float
    """
    from solvcon.dependency import getcdll
    __clib_lincese = {
        2: getcdll('lincese2d', raise_on_fail=False),
        3: getcdll('lincese3d', raise_on_fail=False),
    }
    del getcdll
    @property
    def _clib_lincese(self):
        return self.__clib_lincese[self.ndim]
    @property
    def _jacofunc_(self):
        return self._clib_lincese.calc_jaco
    def __init__(self, *args, **kw):
        self.cfldt = kw.pop('cfldt', None)
        self.cflmax = 0.0
        super(LinceseSolver, self).__init__(*args, **kw)
    def make_grpda(self):
        raise NotImplementedError
    def provide(self):
        from ctypes import byref, c_int
        # fill group data array.
        self.make_grpda()
        # pre-calculate CFL.
        self._set_time(self.time, self.cfldt)
        self._clib_lincese.calc_cfl(
            byref(self.exd), c_int(0), c_int(self.ncell))
        self.cflmax = self.cfl.max()
        # super method.
        super(LinceseSolver, self).provide()
    def calccfl(self, worker=None):
        self.marchret.setdefault('cfl', [0.0, 0.0, 0, 0])
        self.marchret['cfl'][0] = self.cflmax
        self.marchret['cfl'][1] = self.cflmax
        self.marchret['cfl'][2] = 0
        self.marchret['cfl'][3] = 0
        return self.marchret

###############################################################################
# Case.
###############################################################################

class LinceseCase(CeseCase):
    """
    Basic case with linear CESE method.
    """
    from solvcon.domain import Domain
    defdict = {
        'solver.solvertype': LinceseSolver,
        'solver.domaintype': Domain,
        'solver.alpha': 0,
        'solver.cfldt': None,
    }
    del Domain
    def make_solver_keywords(self):
        kw = super(LinceseCase, self).make_solver_keywords()
        # setup delta t for CFL calculation.
        cfldt = self.solver.cfldt
        cfldt = self.execution.time_increment if cfldt is None else cfldt
        kw['cfldt'] = cfldt
        return kw

###############################################################################
# Plane wave solution and initializer.
###############################################################################

class PlaneWaveSolution(object):
    def __init__(self, **kw):
        from numpy import sqrt
        wvec = kw['wvec']
        ctr = kw['ctr']
        amp = kw['amp']
        assert len(wvec) == len(ctr)
        # calculate eigenvalues and eigenvectors.
        evl, evc = self._calc_eigen(**kw)
        # store data to self.
        self.amp = evc * (amp / sqrt((evc**2).sum()))
        self.ctr = ctr
        self.wvec = wvec
        self.afreq = evl * sqrt((wvec**2).sum())
        self.wsp = evl
    def _calc_eigen(self, *args, **kw):
        """
        Calculate eigenvalues and eigenvectors.

        @return: eigenvalues and eigenvectors.
        @rtype: tuple
        """
        raise NotImplementedError
    def __call__(self, svr, asol, adsol):
        from ctypes import byref, c_double
        svr._clib_lincese.calc_planewave(
            byref(svr.exd),
            asol.ctypes._as_parameter_,
            adsol.ctypes._as_parameter_,
            self.amp.ctypes._as_parameter_,
            self.ctr.ctypes._as_parameter_,
            self.wvec.ctypes._as_parameter_,
            c_double(self.afreq),
        )

class PlaneWaveAnchor(Anchor):
    def __init__(self, svr, **kw):
        self.planewaves = kw.pop('planewaves')
        super(PlaneWaveAnchor, self).__init__(svr, **kw)
    def _calculate(self, asol):
        for pw in self.planewaves:
            pw(self.svr, asol, self.adsol)
    def provide(self):
        from numpy import empty
        ngstcell = self.svr.ngstcell
        nacell = self.svr.ncell + ngstcell
        # plane wave solution.
        asol = self.svr.der['analytical'] = empty(
            (nacell, self.svr.neq), dtype=self.svr.fpdtype)
        adsol = self.adsol = empty(
            (nacell, self.svr.neq, self.svr.ndim),
            dtype=self.svr.fpdtype)
        asol.fill(0.0)
        self._calculate(asol)
        self.svr.soln[ngstcell:,:] = asol[ngstcell:,:]
        self.svr.dsoln[ngstcell:,:,:] = adsol[ngstcell:,:,:]
        # difference.
        diff = self.svr.der['difference'] = empty(
            (nacell, self.svr.neq), dtype=self.svr.fpdtype)
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

class PlaneWaveHook(BlockHook):
    def __init__(self, svr, **kw):
        self.planewaves = kw.pop('planewaves')
        self.norm = dict()
        super(PlaneWaveHook, self).__init__(svr, **kw)
    def drop_anchor(self, svr):
        svr.runanchors.append(
            PlaneWaveAnchor(svr, planewaves=self.planewaves)
        )
    def _calculate(self):
        from numpy import empty, sqrt, abs
        neq = self.cse.execution.neq
        var = self.cse.execution.var
        asol = self._collect_interior(
            'analytical', inder=True, consider_ghost=True)
        diff = self._collect_interior(
            'difference', inder=True, consider_ghost=True)
        norm_Linf = empty(neq, dtype='float64')
        norm_L2 = empty(neq, dtype='float64')
        clvol = self.blk.clvol
        for it in range(neq):
            norm_Linf[it] = abs(diff[:,it]).max()
            norm_L2[it] = sqrt((diff[:,it]**2*clvol).sum())
        self.norm['Linf'] = norm_Linf
        self.norm['L2'] = norm_L2
    def preloop(self):
        from numpy import pi
        self.postmarch()
        for ipw in range(len(self.planewaves)):
            pw = self.planewaves[ipw]
            self.info("planewave[%d]:\n" % ipw)
            self.info("  c = %g, omega = %g, T = %.15e\n" % (
                pw.wsp, pw.afreq, 2*pi/pw.afreq))
    def postmarch(self):
        psteps = self.psteps
        istep = self.cse.execution.step_current
        if istep%psteps == 0:
            self._calculate()
    def postloop(self):
        import os
        from cPickle import dump
        fname = '%s_norm.pickle' % self.cse.io.basefn
        fname = os.path.join(self.cse.io.basedir, fname)
        dump(self.norm, open(fname, 'wb'), -1)
        self.info('Linf norm in velocity:\n')
        self.info('  %e, %e, %e\n' % tuple(self.norm['Linf'][:3]))
        self.info('L2 norm in velocity:\n')
        self.info('  %e, %e, %e\n' % tuple(self.norm['L2'][:3]))
