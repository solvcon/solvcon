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


###############################################################################
# Solver.
###############################################################################

class NSBulkSolver(CuseSolver):
    """
    Navier-Stokes solver based on the Bulk modulus.
    """
    def __init__(self, blk, *args, **kw):
        kw['nsca'] = 12
        super(NSBulkSolver, self).__init__(blk, *args, **kw)
    from solvcon.dependency import getcdll
    __clib_nsbulk_c = {
        2: getcdll('nsbulk2d'),
        3: getcdll('nsbulk3d'),
    }
    __clib_nsbulk_cu = {
        2: getcdll('nsbulk2d_cu', raise_on_fail=False),
        3: getcdll('nsbulk3d_cu', raise_on_fail=False),
    }
    del getcdll
    @property
    def _clib_nsbulk_c(self):
        return self.__clib_nsbulk_c[self.ndim]
    @property
    def _clib_nsbulk_cu(self):
        return self.__clib_nsbulk_cu[self.ndim]
    @property
    def _clib_mcu(self):
        return self.__clib_nsbulk_cu[self.ndim]
    _gdlen_ = 0
    @property
    def _jacofunc_(self):
        return self._clib_nsbulk_c.calc_jaco
    @property
    def _viscfunc_(self):
        return self._clib_nsbulk_c.calc_viscous
    def calccfl(self, worker=None):
        from ctypes import byref
        if self.scu:
            self._clib_nsbulk_cu.calc_cfl(self.ncuth,
                byref(self.cumgr.exd), self.cumgr.gexd.gptr)
        else:
            self._clib_nsbulk_c.calc_cfl(byref(self.exd))

###############################################################################
# Case.
###############################################################################

class NSBulkCase(CuseCase):
    """
    Navier-Stokes Bulk case.
    """
    from solvcon.domain import Domain
    defdict = {
        'solver.solvertype': NSBulkSolver,
        'solver.domaintype': Domain,
    }
    del Domain
    def load_block(self):
        loaded = super(NSBulkCase, self).load_block()
        if hasattr(loaded, 'ndim'):
            ndim = loaded.ndim
        else:
            ndim = loaded.blk.ndim
        self.execution.neq = ndim+1
        return loaded

###############################################################################
# Boundary conditions.
###############################################################################

class NSBulkBC(CuseBC):
    """
    Basic BC class for NS.
    """
    from solvcon.dependency import getcdll
    __clib_nsbulkb_c = {
        2: getcdll('nsbulkb2d'),
        3: getcdll('nsbulkb3d'),
    }
    __clib_nsbulkb_cu = {
        2: getcdll('nsbulkb2d_cu', raise_on_fail=False),
        3: getcdll('nsbulkb3d_cu', raise_on_fail=False),
    }
    del getcdll
    @property
    def _clib_nsbulkb_c(self):
        return self.__clib_nsbulkb_c[self.svr.ndim]
    @property
    def _clib_nsbulkb_cu(self):
        return self.__clib_nsbulkb_cu[self.svr.ndim]

class NSBNswall(NSBulkBC):
    _ghostgeom_= 'mirror'
    def soln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_nsbulkb_cu.bound_nswall_soln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_nsbulkb_c.bound_nswall_soln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)
    def dsoln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_nsbulkb_cu.bound_nswall_dsoln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_nsbulkb_c.bound_nswall_dsoln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)

class NSBWall(NSBulkBC):
    _ghostgeom_= 'mirror'
    def soln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_nsbulkb_cu.bound_wall_soln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_nsbulkb_c.bound_wall_soln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)
    def dsoln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_nsbulkb_cu.bound_wall_dsoln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_nsbulkb_c.bound_wall_dsoln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)

class NSBNonrefl(NSBulkBC):
    _ghostgeom_= 'mirror'
    def soln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_nsbulkb_cu.bound_nonrefl_soln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_nsbulkb_c.bound_nonrefl_soln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)
    def dsoln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_nsbulkb_cu.bound_nonrefl_dsoln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_nsbulkb_c.bound_nonrefl_dsoln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)

class NSBInlet(NSBulkBC):
    vnames = ['rho', 'v1', 'v2', 'v3']
    vdefaults = {
        'rho': 1.0, 'v1': 0.0, 'v2': 0.0, 'v3': 0.0,
    }
    _ghostgeom_ = 'mirror'
    def soln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_nsbulkb_cu.bound_inlet_soln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr,
                self.value.shape[1], self.cuvalue.gptr)
        else:
            self._clib_nsbulkb_c.bound_inlet_soln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_,
                self.value.shape[1], self.value.ctypes._as_parameter_)
    def dsoln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_nsbulkb_cu.bound_inlet_dsoln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_nsbulkb_c.bound_inlet_dsoln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)

class NSBNonreflInlet(NSBulkBC):
    vnames = ['rho', 'v1', 'v2', 'v3']
    vdefaults = {
        'rho': 1.0, 'v1': 0.0, 'v2': 0.0, 'v3': 0.0,
    }
    _ghostgeom_ = 'mirror'
    def soln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_nsbulkb_cu.bound_nonreflinlet_soln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr,
                self.value.shape[1], self.cuvalue.gptr)
        else:
            self._clib_nsbulkb_c.bound_nonreflinlet_soln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_,
                self.value.shape[1], self.value.ctypes._as_parameter_)
    def dsoln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_nsbulkb_cu.bound_nonreflinlet_dsoln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_nsbulkb_c.bound_nonreflinlet_dsoln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)

class NSBOutlet(NSBulkBC):
    vnames = ['rho', 'v1', 'v2', 'v3']
    vdefaults = {
        'rho': 1.0, 'v1': 0.0, 'v2': 0.0, 'v3': 0.0,
    }
    _ghostgeom_ = 'mirror'
    def soln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_nsbulkb_cu.bound_outlet_soln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr,
                self.value.shape[1], self.cuvalue.gptr)
        else:
            self._clib_nsbulkb_c.bound_outlet_soln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_,
                self.value.shape[1], self.value.ctypes._as_parameter_)
    def dsoln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_nsbulkb_cu.bound_outlet_dsoln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_nsbulkb_c.bound_outlet_dsoln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)

###############################################################################
# Anchors.
###############################################################################

class NSBulkIAnchor(Anchor):
    """
    Basic initializing anchor class of GasdynSolver.
    """
    def __init__(self, svr, **kw):
        assert isinstance(svr, NSBulkSolver)
        self.bulk = float(kw.pop('bulk'))
        self.p0 = float(kw.pop('p0'))
        self.rho0 = float(kw.pop('rho0'))
        self.eta = float(kw.pop('eta'))
        self.pref = float(kw.pop('pref'))
        self.pini = float(kw.pop('pini'))
        self.mu = float(kw.pop('mu'))
        self.d = float(kw.pop('d'))
        self.xmax = float(kw.pop('xmax'))
        self.xmin = float(kw.pop('xmin'))
        self.ymax = float(kw.pop('ymax'))
        self.ymin = float(kw.pop('ymin'))
        super(NSBulkIAnchor, self).__init__(svr, **kw)
    def provide(self):
        from solvcon.solver import ALMOST_ZERO
        svr = self.svr
        svr.amsca[:,0].fill(self.bulk)
        svr.amsca[:,1].fill(self.p0)
        svr.amsca[:,2].fill(self.rho0)
        svr.amsca[:,3].fill(self.eta)
        svr.amsca[:,4].fill(self.pref)
        svr.amsca[:,5].fill(self.pini)
        svr.amsca[:,6].fill(self.mu)
        svr.amsca[:,7].fill(self.d)
        svr.amsca[:,8].fill(self.xmax)
        svr.amsca[:,9].fill(self.xmin)
        svr.amsca[:,10].fill(self.ymax)
        svr.amsca[:,11].fill(self.ymin)
        svr.sol.fill(ALMOST_ZERO)
        svr.soln.fill(ALMOST_ZERO)
        svr.dsol.fill(ALMOST_ZERO)
        svr.dsoln.fill(ALMOST_ZERO)

class UniformIAnchor(NSBulkIAnchor):
    def __init__(self, svr, **kw):
        self.rho = float(kw.pop('rho'))
        self.v1 = float(kw.pop('v1'))
        self.v2 = float(kw.pop('v2'))
        self.v3 = float(kw.pop('v3'))
        super(UniformIAnchor, self).__init__(svr, **kw)
    def provide(self):
        super(UniformIAnchor, self).provide()
        svr = self.svr
        svr.soln[:,0].fill(self.rho)
        svr.soln[:,1].fill(self.rho*self.v1)
        svr.soln[:,2].fill(self.rho*self.v2)
        if svr.ndim == 3:
            svr.soln[:,3].fill(self.rho*self.v3)
        svr.sol[:] = svr.soln[:]

class TwoRegionIAnchor(NSBulkIAnchor):
    def __init__(self, svr, **kw):
        self.crd = int(kw.pop('crd'))
        self.rho = float(kw.pop('rho'))
        self.v1 = float(kw.pop('v1'))
        #self.ptotal = float(kw.pop('ptotal'))
        super(TwoRegionIAnchor, self).__init__(svr, **kw)
    def provide(self):
        super(TwoRegionIAnchor, self).provide()
        from numpy import sqrt
        svr = self.svr
        ncell = svr.ncell
        ngstcell = svr.ngstcell
        svr.soln[:,1].fill(self.rho*self.v1)
        svr.soln[:,2].fill(0.0)
        svr.soln[:,0].fill(self.rho)
        icl = ngstcell
        while icl < ncell+ngstcell:
            if svr.cecnd[icl,0,self.crd] > 0.0127:
                svr.soln[icl,0] = self.rho
                svr.soln[icl,1] = self.rho*self.v1*0.1
            icl += 1
        svr.sol[:] = svr.soln[:]

class NSBulkOAnchor(Anchor):
    """
    Calculates physical quantities for output.  Implements (i) provide() and
    (ii) postfull() methods.

    @ivar gasconst: gas constant.
    @itype gasconst: float.
    """
    _varlist_ = ['v', 'rho', 'p', 'T', 'ke', 'a', 'M', 'sch',
                 'predif', 'dB', ]
    def __init__(self, svr, **kw):
        self.rsteps = kw.pop('rsteps', 1)
        self.gasconst = kw.pop('gasconst', 1.0)
        self.schk = kw.pop('schk', 1.0)
        self.schk0 = kw.pop('schk0', 0.0)
        self.schk1 = kw.pop('schk1', 1.0)
        super(NSBulkOAnchor, self).__init__(svr, **kw)
    def _calculate_physics(self):
        from ctypes import byref, c_double
        svr = self.svr
        der = svr.der
        svr._clib_nsbulk_c.process_physics(byref(svr.exd),
            c_double(self.gasconst),
            der['v'].ctypes._as_parameter_,
            der['w'].ctypes._as_parameter_,
            der['wm'].ctypes._as_parameter_,
            der['rho'].ctypes._as_parameter_,
            der['p'].ctypes._as_parameter_,
            der['a'].ctypes._as_parameter_,
            der['M'].ctypes._as_parameter_,
        ) 
    def _calculate_schlieren(self):
        from ctypes import byref, c_double
        svr = self.svr
        sch = svr.der['sch']
        svr._clib_nsbulk_c.process_schlieren_rhog(byref(svr.exd),
            sch.ctypes._as_parameter_)
        rhogmax = sch[svr.ngstcell:].max()
        svr._clib_nsbulk_c.process_schlieren_sch(byref(svr.exd),
            c_double(self.schk), c_double(self.schk0), c_double(self.schk1),
            c_double(rhogmax), sch.ctypes._as_parameter_,
        )
    def _calculate_dB(self):
        from ctypes import byref, c_double
        svr = self.svr
        der = svr.der
        svr._clib_nsbulk_c.process_dB(byref(svr.exd),
            der['predif'].ctypes._as_parameter_,
            der['dB'].ctypes._as_parameter_,
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
        der['predif'] = empty(nelm, dtype=svr.fpdtype)
        der['dB'] = empty(nelm, dtype=svr.fpdtype)
        self._calculate_physics()
        self._calculate_schlieren()
        self._calculate_dB()
    def postfull(self):
        svr = self.svr
        istep = self.svr.step_global
        rsteps = self.rsteps
        self._calculate_dB()
        if istep > 0 and istep%rsteps == 0:
            if svr.scu:
                svr.cumgr.arr_from_gpu('amsca', 'soln', 'dsoln')
            self._calculate_physics()
            self._calculate_schlieren()
            #self._calculate_dB()
