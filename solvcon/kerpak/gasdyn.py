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
Gas dynamics solver of the Euler equations.
"""

from solvcon.kerpak.cuse import CuseSolver
from solvcon.kerpak.cuse import CuseCase
from solvcon.kerpak.cuse import CuseBC
from solvcon.anchor import Anchor
from solvcon.hook import BlockHook

################################################################################
# Utility.
################################################################################

class MovingShock(object):
    """
    Define relations across a moving shock wave.  Subscript 1 denotes
    quantities before shock (have not touched by shock), subscript 2 denotes
    quantities after shock (passed by shock).

    @ivar ga: ratio of specific heat.
    @itype ga: float
    @ivar Ms: Mach number of shock wave.
    @itype Ms: float
    @ivar gasconst: gas constant.
    @itype gasconst: float
    """
    def __init__(self, ga, Ms, **kw):
        self.ga = ga
        self.Ms = Ms
        self.gasconst = kw.pop('gasconst', 1.0)

    @property
    def ratio_p(self):
        """
        ratio of upstream/downstream pressure.
        """
        ga = self.ga
        Ms = self.Ms
        return (2*ga*Ms**2 - (ga-1))/(ga+1)
    @property
    def ratio_rho(self):
        """
        ratio of upstream/downstream density.
        """
        ga = self.ga
        Ms = self.Ms
        return (ga+1)*Ms**2/(2+(ga-1)*Ms**2)
    @property
    def ratio_T(self):
        """
        ratio of upstream/downstream temperature.
        """
        ga = self.ga
        Ms = self.Ms
        return self.ratio_p/self.ratio_rho

    @property
    def M2(self):
        """
        Mach number behind standing normal shock wave.
        """
        from math import sqrt
        ga = self.ga
        Ms = self.Ms
        return sqrt(((ga-1)*Ms**2+2)/(2*ga*Ms**2-(ga-1)))
    @property
    def M2p(self):
        """
        Mach number behind moving normal shock wave.
        """
        from math import sqrt
        M1 = self.Ms
        M2 = self.M2
        ratio_a = sqrt(self.ratio_T)
        return M1/ratio_a - M2

    def calc_temperature(self, p, rho):
        """
        Calculate temperature according to given pressure and density.
        
        @param p: pressure.
        @type p: float
        @param rho: density.
        @type rho: float
        @return: temperature
        @rtype: float
        """
        return p/(rho*self.gasconst)
    def calc_speedofsound(self, p, rho):
        """
        Calculate speed of sound according to given pressure and density.
        
        @param p: pressure.
        @type p: float
        @param rho: density.
        @type rho: float
        @return: speed of sound
        @rtype: float
        """
        from math import sqrt
        ga = self.ga
        return sqrt(ga*p/rho)
    def calc_speeds(self, p, rho):
        """
        Calculate shock wave speed and upstream speed for static downstream.

        @param p: downstream pressure.
        @type p: float
        @param rho: downstream density.
        @type rho: float
        @return: a 2-tuple for shock wave and upstream speeds.
        @rtype: (float, float)
        """
        M1 = self.Ms
        M2 = self.M2
        a1 = self.calc_speedofsound(p, rho)
        a2 = self.calc_speedofsound(p*self.ratio_p, rho*self.ratio_rho)
        return M1*a1, M1*a1 - M2*a2

###############################################################################
# Solver.
###############################################################################

class GasdynSolver(CuseSolver):
    """
    Gas dynamics solver of the Euler equations.
    """
    def __init__(self, blk, *args, **kw):
        kw['nsca'] = 1
        super(GasdynSolver, self).__init__(blk, *args, **kw)
    from solvcon.dependency import getcdll
    __clib_gasdyn_c = {
        2: getcdll('gasdyn2d_c', raise_on_fail=False),
        3: getcdll('gasdyn3d_c', raise_on_fail=False),
    }
    __clib_gasdyn_cu = {
        2: getcdll('gasdyn2d_cu', raise_on_fail=False),
        3: getcdll('gasdyn3d_cu', raise_on_fail=False),
    }
    del getcdll
    @property
    def _clib_gasdyn_c(self):
        return self.__clib_gasdyn_c[self.ndim]
    @property
    def _clib_gasdyn_cu(self):
        return self.__clib_gasdyn_cu[self.ndim]
    @property
    def _clib_mcu(self):
        return self.__clib_gasdyn_cu[self.ndim]
    _gdlen_ = 0
    @property
    def _jacofunc_(self):
        return self._clib_gasdyn_c.calc_jaco
    def calccfl(self, worker=None):
        from ctypes import byref
        if self.scu:
            self._clib_gasdyn_cu.calc_cfl(self.ncuth,
                byref(self.cumgr.exd), self.cumgr.gexd.gptr)
        else:
            self._clib_gasdyn_c.calc_cfl(byref(self.exd))

###############################################################################
# Case.
###############################################################################

class GasdynCase(CuseCase):
    """
    Gas dynamics case.
    """
    from solvcon.domain import Domain
    defdict = {
        'solver.solvertype': GasdynSolver,
        'solver.domaintype': Domain,
    }
    del Domain
    def load_block(self):
        loaded = super(GasdynCase, self).load_block()
        if hasattr(loaded, 'ndim'):
            ndim = loaded.ndim
        else:
            ndim = loaded.blk.ndim
        self.execution.neq = ndim+2
        return loaded

###############################################################################
# Boundary conditions.
###############################################################################

class GasdynBC(CuseBC):
    """
    Basic BC class for gas dynamics.
    """
    from solvcon.dependency import getcdll
    __clib_gasdynb_c = {
        2: getcdll('gasdynb2d_c', raise_on_fail=False),
        3: getcdll('gasdynb3d_c', raise_on_fail=False),
    }
    __clib_gasdynb_cu = {
        2: getcdll('gasdynb2d_cu', raise_on_fail=False),
        3: getcdll('gasdynb3d_cu', raise_on_fail=False),
    }
    del getcdll
    @property
    def _clib_gasdynb_c(self):
        return self.__clib_gasdynb_c[self.svr.ndim]
    @property
    def _clib_gasdynb_cu(self):
        return self.__clib_gasdynb_cu[self.svr.ndim]

class GasdynWall(GasdynBC):
    _ghostgeom_ = 'mirror'
    def soln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_gasdynb_cu.bound_wall_soln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_gasdynb_c.bound_wall_soln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)
    def dsoln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_gasdynb_cu.bound_wall_dsoln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_gasdynb_c.bound_wall_dsoln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)

class GasdynNswall(GasdynWall):
    def soln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_gasdynb_cu.bound_nswall_soln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_gasdynb_c.bound_nswall_soln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)
    def dsoln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_gasdynb_cu.bound_nswall_dsoln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_gasdynb_c.bound_nswall_dsoln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)

class GasdynInlet(GasdynBC):
    vnames = ['rho', 'v1', 'v2', 'v3', 'p', 'gamma']
    vdefaults = {
        'rho': 1.0, 'p': 1.0, 'gamma': 1.4, 'v1': 0.0, 'v2': 0.0, 'v3': 0.0,
    }
    _ghostgeom_ = 'mirror'
    def soln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_gasdynb_cu.bound_inlet_soln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr,
                self.value.shape[1], self.cuvalue.gptr)
        else:
            self._clib_gasdynb_c.bound_inlet_soln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_,
                self.value.shape[1], self.value.ctypes._as_parameter_)
    def dsoln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_gasdynb_cu.bound_inlet_dsoln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_gasdynb_c.bound_inlet_dsoln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)

###############################################################################
# Anchors.
###############################################################################

class GasdynIAnchor(Anchor):
    """
    Basic initializing anchor class of GasdynSolver.
    """
    def __init__(self, svr, **kw):
        assert isinstance(svr, GasdynSolver)
        self.gamma = float(kw.pop('gamma'))
        super(GasdynIAnchor, self).__init__(svr, **kw)
    def provide(self):
        from solvcon.solver_legacy import ALMOST_ZERO
        svr = self.svr
        svr.amsca.fill(self.gamma)
        svr.sol.fill(ALMOST_ZERO)
        svr.soln.fill(ALMOST_ZERO)
        svr.dsol.fill(ALMOST_ZERO)
        svr.dsoln.fill(ALMOST_ZERO)

class UniformIAnchor(GasdynIAnchor):
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

class GasdynOAnchor(Anchor):
    """
    Calculates physical quantities for output.  Implements (i) provide() and
    (ii) postfull() methods.

    @ivar gasconst: gas constant.
    @itype gasconst: float.
    """
    _varlist_ = ['v', 'rho', 'p', 'T', 'ke', 'a', 'M', 'sch']
    def __init__(self, svr, **kw):
        self.rsteps = kw.pop('rsteps', 1)
        self.gasconst = kw.pop('gasconst', 1.0)
        self.schk = kw.pop('schk', 1.0)
        self.schk0 = kw.pop('schk0', 0.0)
        self.schk1 = kw.pop('schk1', 1.0)
        super(GasdynOAnchor, self).__init__(svr, **kw)
    def _calculate_physics(self):
        from ctypes import byref, c_double
        svr = self.svr
        der = svr.der
        svr._clib_gasdyn_c.process_physics(byref(svr.exd),
            c_double(self.gasconst),
            der['v'].ctypes._as_parameter_,
            der['w'].ctypes._as_parameter_,
            der['wm'].ctypes._as_parameter_,
            der['rho'].ctypes._as_parameter_,
            der['p'].ctypes._as_parameter_,
            der['T'].ctypes._as_parameter_,
            der['ke'].ctypes._as_parameter_,
            der['a'].ctypes._as_parameter_,
            der['M'].ctypes._as_parameter_,
        )
    def _calculate_schlieren(self):
        from ctypes import byref, c_double
        svr = self.svr
        sch = svr.der['sch']
        svr._clib_gasdyn_c.process_schlieren_rhog(byref(svr.exd),
            sch.ctypes._as_parameter_)
        rhogmax = sch[svr.ngstcell:].max()
        svr._clib_gasdyn_c.process_schlieren_sch(byref(svr.exd),
            c_double(self.schk), c_double(self.schk0), c_double(self.schk1),
            c_double(rhogmax), sch.ctypes._as_parameter_,
        )
    def provide(self):
        from numpy import empty
        svr = self.svr
        der = svr.der
        nelm = svr.ngstcell + svr.ncell
        der['v'] = empty((nelm, svr.ndim), dtype=svr.fpdtype)
        der['w'] = empty((nelm, svr.ndim), dtype=svr.fpdtype)
        der['wm'] = empty(nelm, dtype=svr.fpdtype)
        der['rho'] = empty(nelm, dtype=svr.fpdtype)
        der['p'] = empty(nelm, dtype=svr.fpdtype)
        der['T'] = empty(nelm, dtype=svr.fpdtype)
        der['ke'] = empty(nelm, dtype=svr.fpdtype)
        der['a'] = empty(nelm, dtype=svr.fpdtype)
        der['M'] = empty(nelm, dtype=svr.fpdtype)
        der['sch'] = empty(nelm, dtype=svr.fpdtype)
        self._calculate_physics()
        self._calculate_schlieren()
    def postfull(self):
        svr = self.svr
        istep = self.svr.step_global
        rsteps = self.rsteps
        if istep > 0 and istep%rsteps == 0:
            if svr.scu:
                svr.cumgr.arr_from_gpu('amsca', 'soln', 'dsoln')
            self._calculate_physics()
            self._calculate_schlieren()
