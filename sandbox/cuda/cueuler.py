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
Euler equations solver using the CESE method with CUDA.
"""

from solvcon.kerpak.euler import EulerSolver
from solvcon.kerpak.cese import CeseCase
from solvcon.kerpak.cese import CeseBC
from solvcon.anchor import Anchor
from solvcon.hook import BlockHook

def getcdll(libname):
    """
    Load shared objects at the default location.

    @param libname: main basename of library without sc_ prefix.
    @type libname: str
    @return: ctypes library.
    @rtype: ctypes.CDLL
    """
    from solvcon.dependency import loadcdll
    return loadcdll('.', 'sc_'+libname)

###############################################################################
# Solver.
###############################################################################

class CueulerSolver(EulerSolver):
    """
    Inviscid aerodynamic solver for the Euler equations.
    """
    _pointers_ = ['exc']
    def __init__(self, blk, *args, **kw):
        from solvcon.conf import env
        self.scu = scu = env.scu
        kw['nsca'] = 1
        self.ncuthread = kw.pop('ncuthread',  32)
        super(CueulerSolver, self).__init__(blk, *args, **kw)
        # data structure for CUDA/C.
        self.exc = None
        self.gexc = None
        for key in (
            # connectivity.
            'clfcs', 'fcnds', 'fccls',
            # geometry.
            'ndcrd', 'fccnd', 'clcnd',
            # meta array.
            'cltpn', 'grpda',
            # dual mesh.
            'cecnd', 'cevol',
            # solutions.
            'solt', 'sol', 'soln', 'dsol', 'dsoln',
            'cfl', 'ocfl', 'amsca', 'amvec',
        ):
            nbytes = getattr(self, key).nbytes
            setattr(self, 'cu'+key, scu.alloc(nbytes))
    #from solvcon.dependency import getcdll
    __clib_cueuler = {
        2: getcdll('cueuler2d'),
        3: getcdll('cueuler3d'),
    }
    #del getcdll
    @property
    def _clib_cueuler(self):
        return self.__clib_cueuler[self.ndim]
    _gdlen_ = 0
    @property
    def _jacofunc_(self):
        return self._clib_euler.calc_jaco

    @staticmethod
    def __set_cuda_pointer(exc, svr, aname, shf):
        from ctypes import c_void_p
        arr = getattr(svr, aname)
        gptr = getattr(svr, 'cu'+aname).gptr
        nt = 1
        if len(arr.shape) > 1:
            for width in arr.shape[1:]:
                nt *= width
        shf *= nt * arr.itemsize
        addr = gptr.value + shf if gptr.value is not None else None
        setattr(exc, aname, c_void_p(addr))
    def set_execuda(self, exd, exc):
        from ctypes import c_void_p
        if exd is None:
            exd = self._exedatatype_(svr=self)
        # copy everything.
        for key, ctp in exd._fields_:
            setattr(self.exc, key, getattr(exd, key))
        # set shifted pointers.
        for key in ('clfcs',):
            self.__set_cuda_pointer(exc, self, key, self.ngstcell)
        for key in ('fcnds', 'fccls'):
            self.__set_cuda_pointer(exc, self, key, self.ngstface)
        for key in ('ndcrd',):
            self.__set_cuda_pointer(exc, self, key, self.ngstnode)
        for key in ('fccnd',):
            self.__set_cuda_pointer(exc, self, key, self.ngstface)
        for key in ('clcnd',):
            self.__set_cuda_pointer(exc, self, key, self.ngstcell)
        for key in ('cltpn',):
            self.__set_cuda_pointer(exc, self, key, self.ngstcell)
        for key in ('grpda',):
            self.__set_cuda_pointer(exc, self, key, 0)
        for key in ('cecnd', 'cevol'):
            self.__set_cuda_pointer(exc, self, key, self.ngstcell)
        for key in ('solt', 'sol', 'soln', 'dsol', 'dsoln',):
            self.__set_cuda_pointer(exc, self, key, self.ngstcell)
        for key in ('cfl', 'ocfl',):
            self.__set_cuda_pointer(exc, self, key, 0)
        for key in ('amsca', 'amvec',):
            self.__set_cuda_pointer(exc, self, key, self.ngstcell)

    def bind(self):
        from ctypes import sizeof
        if self.scu:
            self.exc = self._exedatatype_(svr=self)
            self.set_execuda(None, self.exc)
            self.gexc = self.scu.alloc(sizeof(self.exc))
        super(CueulerSolver, self).bind()

    def init(self, **kw):
        super(CueulerSolver, self).init(**kw)
        if self.scu:
            for key in ('cltpn',
                'clfcs', 'fcnds', 'fccls', 'ndcrd', 'fccnd', 'clcnd',
                'cecnd', 'cevol',
            ):
                self.scu.memcpy(getattr(self, 'cu'+key), getattr(self, key))

    def update(self, worker=None):
        if self.debug: self.mesg('update')
        from ctypes import byref, sizeof
        # exchange solution and gradient.
        self.sol, self.soln = self.soln, self.sol
        self.dsol, self.dsoln = self.dsoln, self.dsol
        # reset pointers in execution data.
        ngstcell = self.ngstcell
        self.exd.sol = self.sol[ngstcell:].ctypes._as_parameter_
        self.exd.soln = self.soln[ngstcell:].ctypes._as_parameter_
        self.exd.dsol = self.dsol[ngstcell:].ctypes._as_parameter_
        self.exd.dsoln = self.dsoln[ngstcell:].ctypes._as_parameter_
        # reset GPU execution data.
        if self.scu:
            self.set_execuda(self.exd, self.exc)
            self.scu.cudaMemcpy(self.gexc.gptr, byref(self.exc),
                sizeof(self.exc), self.scu.cudaMemcpyHostToDevice)
            for key in ('sol', 'dsol',):
                self.scu.memcpy(getattr(self, 'cu'+key), getattr(self, key))
        if self.debug: self.mesg(' done.\n')

    def ibcam(self, worker=None):
        if self.debug: self.mesg('ibcam')
        if worker:
            if self.nsca: self.exchangeibc('amsca', worker=worker)
            if self.nvec: self.exchangeibc('amvec', worker=worker)
        if self.nsca: self.scu.memcpy(self.cuamsca, self.amsca)
        if self.nvec: self.scu.memcpy(self.cuamvec, self.amvec)
        if self.debug: self.mesg(' done.\n')

    def calcsolt(self, worker=None):
        if self.debug: self.mesg('calcsolt')
        from ctypes import byref
        self._clib_cueuler.calc_solt(self.ncuthread,
            byref(self.exc), self.gexc.gptr)
        for key in ('solt',):
            self.scu.memcpy(getattr(self, key), getattr(self, 'cu'+key))
        if self.debug: self.mesg(' done.\n')

    def calcsoln(self, worker=None):
        if self.debug: self.mesg('calcsoln')
        from ctypes import byref
        self._clib_cueuler.calc_soln(self.ncuthread,
            byref(self.exc), self.gexc.gptr)
        for key in ('soln',):
            self.scu.memcpy(getattr(self, key), getattr(self, 'cu'+key))
        if self.debug: self.mesg(' done.\n')

    def vcalccfl(self, worker=None):
        from ctypes import byref
        self._clib_cueuler.calc_cfl(byref(self.exc))
        mincfl = self.ocfl.min()
        maxcfl = self.ocfl.max()
        nadj = (self.cfl==1).sum()
        if self.marchret is None:
            self.marchret = [0.0, 0.0, 0, 0]
        self.marchret[0] = mincfl
        self.marchret[1] = maxcfl
        self.marchret[2] = nadj
        self.marchret[3] += nadj
        return self.marchret

    def calcdsoln(self, worker=None):
        if self.debug: self.mesg('calcdsoln')
        from ctypes import byref
        for key in ('cfl',):
            self.scu.memcpy(getattr(self, 'cu'+key), getattr(self, key))
        self._clib_cueuler.calc_dsoln(self.ncuthread,
            byref(self.exc), self.gexc.gptr)
        for key in ('dsoln',):
            self.scu.memcpy(getattr(self, key), getattr(self, 'cu'+key))
        if self.debug: self.mesg(' done.\n')

###############################################################################
# Case.
###############################################################################

class CueulerCase(CeseCase):
    """
    Inviscid aerodynamic case for the Euler equations.
    """
    from solvcon.domain import Domain
    defdict = {
        'solver.solvertype': CueulerSolver,
        'solver.domaintype': Domain,
        'solver.cflname': 'adj',
    }
    del Domain
    def make_solver_keywords(self):
        kw = super(CueulerCase, self).make_solver_keywords()
        kw['cflname'] = self.solver.cflname
        return kw
    def load_block(self):
        loaded = super(CueulerCase, self).load_block()
        if hasattr(loaded, 'ndim'):
            ndim = loaded.ndim
        else:
            ndim = loaded.blk.ndim
        self.execution.neq = ndim+2
        return loaded

###############################################################################
# Boundary conditions.
###############################################################################

class CueulerBC(CeseBC):
    """
    Basic BC class for the Euler equations.
    """
    #from solvcon.dependency import getcdll
    __clib_cueulerb = {
        2: getcdll('cueulerb2d'),
        3: getcdll('cueulerb3d'),
    }
    #del getcdll
    @property
    def _clib_eulerb(self):
        return self.__clib_cueulerb[self.svr.ndim]

class CueulerWall(CueulerBC):
    _ghostgeom_ = 'mirror'
    def soln(self):
        from solvcon.dependency import intptr
        from ctypes import byref, c_int
        svr = self.svr
        self._clib_cueulerb.bound_wall_soln(
            byref(svr.exd),
            c_int(self.facn.shape[0]),
            self.facn.ctypes._as_parameter_,
        )
    def dsoln(self):
        from solvcon.dependency import intptr
        from ctypes import byref, c_int
        svr = self.svr
        self._clib_cueulerb.bound_wall_dsoln(
            byref(svr.exd),
            c_int(self.facn.shape[0]),
            self.facn.ctypes._as_parameter_,
        )

class CueulerInlet(CueulerBC):
    vnames = ['rho', 'v1', 'v2', 'v3', 'p', 'gamma']
    vdefaults = {
        'rho': 1.0, 'p': 1.0, 'gamma': 1.4, 'v1': 0.0, 'v2': 0.0, 'v3': 0.0,
    }
    _ghostgeom_ = 'mirror'
    def soln(self):
        from solvcon.dependency import intptr
        from ctypes import byref, c_int
        svr = self.svr
        self._clib_cueulerb.bound_inlet_soln(
            byref(svr.exd),
            c_int(self.facn.shape[0]),
            self.facn.ctypes._as_parameter_,
            c_int(self.value.shape[1]),
            self.value.ctypes._as_parameter_,
        )
    def dsoln(self):
        from solvcon.dependency import intptr
        from ctypes import byref, c_int
        svr = self.svr
        self._clib_cueulerb.bound_inlet_dsoln(
            byref(svr.exd),
            c_int(self.facn.shape[0]),
            self.facn.ctypes._as_parameter_,
        )

###############################################################################
# Anchors.
###############################################################################

class CueulerIAnchor(Anchor):
    """
    Basic initializing anchor class for all Euler problems.
    """
    def __init__(self, svr, **kw):
        assert isinstance(svr, CueulerSolver)
        self.gamma = float(kw.pop('gamma'))
        super(CueulerIAnchor, self).__init__(svr, **kw)
    def provide(self):
        from solvcon.solver import ALMOST_ZERO
        svr = self.svr
        svr.amsca.fill(self.gamma)
        svr.sol.fill(ALMOST_ZERO)
        svr.soln.fill(ALMOST_ZERO)
        svr.dsol.fill(ALMOST_ZERO)
        svr.dsoln.fill(ALMOST_ZERO)

class UniformIAnchor(CueulerIAnchor):
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
