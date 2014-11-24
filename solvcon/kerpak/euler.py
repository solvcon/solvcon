# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2011 Yung-Yu Chen <yyc@solvcon.net>.
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
Euler equations solver using the CESE method.
"""

from .cese import CeseSolver
from .cese import CeseCase
from .cese import CeseBC
from solvcon.anchor import Anchor
from solvcon.hook import BlockHook

###############################################################################
# Solver.
###############################################################################

class EulerSolver(CeseSolver):
    """
    Inviscid aerodynamic solver for the Euler equations.
    """
    def __init__(self, blk, *args, **kw):
        self.cflname = kw.pop('cflname', 'adj')
        kw['nsca'] = 1
        super(EulerSolver, self).__init__(blk, *args, **kw)
        self.cflc = self.cfl.copy() # FIXME: obselete?
    from solvcon.dependency import getcdll
    __clib_euler = {
        2: getcdll('euler2d', raise_on_fail=False),
        3: getcdll('euler3d', raise_on_fail=False),
    }
    del getcdll
    @property
    def _clib_euler(self):
        return self.__clib_euler[self.ndim]
    _gdlen_ = 0
    @property
    def _jacofunc_(self):
        return self._clib_euler.calc_jaco
    def calccfl(self, worker=None):
        func = getattr(self._clib_euler, 'calc_cfl_'+self.cflname)
        self._tcall(func, 0, self.ncell)
        mincfl = self.ocfl.min()
        maxcfl = self.ocfl.max()
        nadj = (self.cfl==1).sum()
        self.marchret.setdefault('cfl', [0.0, 0.0, 0, 0])
        self.marchret['cfl'][0] = mincfl
        self.marchret['cfl'][1] = maxcfl
        self.marchret['cfl'][2] = nadj
        self.marchret['cfl'][3] += nadj
        return self.marchret

###############################################################################
# Case.
###############################################################################

class EulerCase(CeseCase):
    """
    Inviscid aerodynamic case for the Euler equations.
    """
    from solvcon.domain import Domain
    defdict = {
        'solver.solvertype': EulerSolver,
        'solver.domaintype': Domain,
        'solver.cflname': 'adj',
    }
    del Domain
    def make_solver_keywords(self):
        kw = super(EulerCase, self).make_solver_keywords()
        kw['cflname'] = self.solver.cflname
        return kw
    def load_block(self):
        loaded = super(EulerCase, self).load_block()
        if hasattr(loaded, 'ndim'):
            ndim = loaded.ndim
        else:
            ndim = loaded.blk.ndim
        self.execution.neq = ndim+2
        return loaded

###############################################################################
# Boundary conditions.
###############################################################################

class EulerBC(CeseBC):
    """
    Basic BC class for the Euler equations.
    """
    from solvcon.dependency import getcdll
    __clib_eulerb = {
        2: getcdll('eulerb2d', raise_on_fail=False),
        3: getcdll('eulerb3d', raise_on_fail=False),
    }
    del getcdll
    @property
    def _clib_eulerb(self):
        return self.__clib_eulerb[self.svr.ndim]

class EulerWall(EulerBC):
    _ghostgeom_ = 'mirror'
    def soln(self):
        from ctypes import byref, c_int
        svr = self.svr
        self._clib_eulerb.bound_wall_soln(
            byref(svr.exd),
            c_int(self.facn.shape[0]),
            self.facn.ctypes._as_parameter_,
        )
    def dsoln(self):
        from ctypes import byref, c_int
        svr = self.svr
        self._clib_eulerb.bound_wall_dsoln(
            byref(svr.exd),
            c_int(self.facn.shape[0]),
            self.facn.ctypes._as_parameter_,
        )
class EulerNonslipWall(EulerBC):
    _ghostgeom_ = 'mirror'
    def soln(self):
        from ctypes import byref, c_int
        svr = self.svr
        self._clib_eulerb.bound_nonslipwall_soln(
            byref(svr.exd),
            c_int(self.facn.shape[0]),
            self.facn.ctypes._as_parameter_,
        )
    def dsoln(self):
        from ctypes import byref, c_int
        svr = self.svr
        self._clib_eulerb.bound_nonslipwall_dsoln(
            byref(svr.exd),
            c_int(self.facn.shape[0]),
            self.facn.ctypes._as_parameter_,
        )

class EulerInlet(EulerBC):
    vnames = ['rho', 'v1', 'v2', 'v3', 'p', 'gamma']
    vdefaults = {
        'rho': 1.0, 'p': 1.0, 'gamma': 1.4, 'v1': 0.0, 'v2': 0.0, 'v3': 0.0,
    }
    _ghostgeom_ = 'mirror'
    def soln(self):
        from ctypes import byref, c_int
        svr = self.svr
        self._clib_eulerb.bound_inlet_soln(
            byref(svr.exd),
            c_int(self.facn.shape[0]),
            self.facn.ctypes._as_parameter_,
            c_int(self.value.shape[1]),
            self.value.ctypes._as_parameter_,
        )
    def dsoln(self):
        from ctypes import byref, c_int
        svr = self.svr
        self._clib_eulerb.bound_inlet_dsoln(
            byref(svr.exd),
            c_int(self.facn.shape[0]),
            self.facn.ctypes._as_parameter_,
        )

class EulerOutlet(EulerBC):
    _ghostgeom_ = 'translate'
    def soln(self):
        from ctypes import byref, c_int
        svr = self.svr
        self._clib_eulerb.bound_outlet_soln(
            byref(svr.exd),
            c_int(self.facn.shape[0]),
            self.facn.ctypes._as_parameter_,
        )
    def dsoln(self):
        from ctypes import byref, c_int
        svr = self.svr
        self._clib_eulerb.bound_outlet_dsoln(
            byref(svr.exd),
            c_int(self.facn.shape[0]),
            self.facn.ctypes._as_parameter_,
        )

###############################################################################
# Anchors.
###############################################################################

class EulerIAnchor(Anchor):
    """
    Basic initializing anchor class for all Euler problems.
    """
    def __init__(self, svr, **kw):
        assert isinstance(svr, EulerSolver)
        self.gamma = float(kw.pop('gamma'))
        super(EulerIAnchor, self).__init__(svr, **kw)
    def provide(self):
        from solvcon.solver_legacy import ALMOST_ZERO
        svr = self.svr
        svr.amsca.fill(self.gamma)
        svr.sol.fill(ALMOST_ZERO)
        svr.soln.fill(ALMOST_ZERO)
        svr.dsol.fill(ALMOST_ZERO)
        svr.dsoln.fill(ALMOST_ZERO)

class UniformIAnchor(EulerIAnchor):
    def __init__(self, svr, **kw):
        self.rho = float(kw.pop('rho'))
        self.v1 = float(kw.pop('v1'))
        self.v2 = float(kw.pop('v2'))
        self.v3 = float(kw.pop('v3'))
        self.p = float(kw.pop('p'))
        super(UniformIAnchor, self).__init__(svr, **kw)
    def provide(self):
        super(UniformIAnchor, self).provide()
        gamma = self.gamma
        svr = self.svr
        svr.soln[:,0].fill(self.rho)
        svr.soln[:,1].fill(self.rho*self.v1)
        svr.soln[:,2].fill(self.rho*self.v2)
        vs = self.v1**2 + self.v2**2
        if svr.ndim == 3:
            vs += self.v3**2
            svr.soln[:,3].fill(self.rho*self.v3)
        svr.soln[:,svr.ndim+1].fill(self.rho*vs/2 + self.p/(gamma-1))
        svr.sol[:] = svr.soln[:]

class EulerOAnchor(Anchor):
    _varlist_ = ['v', 'rho', 'p', 'T', 'ke', 'a', 'M', 'sch']
    def __init__(self, svr, **kw):
        self.schk = kw.pop('schk', 1.0)
        self.schk0 = kw.pop('schk0', 0.0)
        self.schk1 = kw.pop('schk1', 1.0)
        super(EulerOAnchor, self).__init__(svr, **kw)
    def provide(self):
        from numpy import empty
        svr = self.svr
        der = svr.der
        nelm = svr.ngstcell + svr.ncell
        der['v'] = empty((nelm, svr.ndim), dtype=svr.fpdtype)
        der['rho'] = empty(nelm, dtype=svr.fpdtype)
        der['p'] = empty(nelm, dtype=svr.fpdtype)
        der['T'] = empty(nelm, dtype=svr.fpdtype)
        der['ke'] = empty(nelm, dtype=svr.fpdtype)
        der['a'] = empty(nelm, dtype=svr.fpdtype)
        der['M'] = empty(nelm, dtype=svr.fpdtype)
        der['sch'] = empty(nelm, dtype=svr.fpdtype)
        self._calculate_physics()
        self._calculate_schlieren()
    def _calculate_physics(self):
        svr = self.svr
        der = svr.der
        svr._tcall(svr._clib_euler.calc_physics, -svr.ngstcell, svr.ncell,
            der['v'].ctypes._as_parameter_,
            der['rho'].ctypes._as_parameter_,
            der['p'].ctypes._as_parameter_,
            der['T'].ctypes._as_parameter_,
            der['ke'].ctypes._as_parameter_,
            der['a'].ctypes._as_parameter_,
            der['M'].ctypes._as_parameter_,
        )
    def _calculate_schlieren(self):
        from ctypes import c_double
        svr = self.svr
        sch = svr.der['sch']
        svr._tcall(svr._clib_euler.calc_schlieren_rhog,
            -svr.ngstcell, svr.ncell, sch.ctypes._as_parameter_)
        rhogmax = sch[svr.ngstcell:].max()
        svr._tcall(svr._clib_euler.calc_schlieren_sch,
            -svr.ngstcell, svr.ncell,
            c_double(self.schk), c_double(self.schk0), c_double(self.schk1),
            c_double(rhogmax), sch.ctypes._as_parameter_,
        )
    def postfull(self):
        self._calculate_physics()
        self._calculate_schlieren()

###############################################################################
# Hooks.
###############################################################################

class ConservationHook(BlockHook):
    def __init__(self, cse, **kw):
        self.records = list()
        self.recordpath = None
        super(ConservationHook, self).__init__(cse, **kw)
    def preloop(self):
        import os
        recordfn = '%s_conservation.npy' % self.cse.io.basefn
        self.recordpath = os.path.join(self.cse.io.basedir, recordfn)
    def _calculate(self):
        clvol = self.blk.clvol
        rho = self.cse.execution.var['rho']
        etot = self.cse.execution.var['soln'][:,-1]
        utot = etot - self.cse.execution.var['ke']
        crho = (clvol*rho).sum()
        cetot = (clvol*etot).sum()
        cutot = (clvol*utot).sum()
        self.records.append([self.cse.execution.time,
            crho, cetot, cutot])
    def postmarch(self):
        from numpy import array, save
        self._calculate()
        istep = self.cse.execution.step_current
        if istep%self.psteps == 0:
            info = self.info
            d0 = array(self.records[0][1:], dtype='float64')
            dn = array(self.records[-1][1:], dtype='float64')
            diff = (dn-d0)/d0 * 100
            info('density conservation difference: %7.3f%%\n' % diff[0])
            info('energy  conservation difference: %7.3f%%\n' % diff[1])
            info('thermal conservation difference: %7.3f%%\n' % diff[2])
            save(self.recordpath, self.records)
    def postloop(self):
        from numpy import array
        info = self.info
        self.records = array(self.records, dtype='float64')
        info('mass conservation: %g %g\n' % (
            self.records[0,1], self.records[-1,1]))
        info('energy conservation: %g %g\n' % (
            self.records[0,2], self.records[-1,2]))
        info('thermal conservation: %g %g\n' % (
            self.records[0,3], self.records[-1,3]))
        info('Conservation record saved to %s .\n' % self.recordpath)
