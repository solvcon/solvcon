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

from solvcon.gendata import SingleAssignDict, AttributeDict
from solvcon.anchor import Anchor
from solvcon.hook import BlockHook
#from .cese import CeseSolver, CeseCase, CeseBC
from solvcon.kerpak.cese import CeseSolver, CeseCase, CeseBC

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

class LinceseSolver(CeseSolver):
    """
    Basic linear CESE solver.

    @ivar cfldt: the time_increment for CFL calculation at boundcond.
    @itype cfldt: float
    @ivar cflmax: the maximum CFL number.
    @itype cflmax: float
    """
    #from solvcon.dependency import getcdll
    __clib_lincese = {
        2: getcdll('lincese2d'),
        3: getcdll('lincese3d'),
    }
    #del getcdll
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
        if self.marchret is None:
            self.marchret = [0.0, 0.0, 0, 0]
        self.marchret[0] = self.cflmax
        self.marchret[1] = self.cflmax
        self.marchret[2] = 0
        self.marchret[3] = 0
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
        'solver.mtrldict': dict,
    }
    del Domain
    def make_solver_keywords(self):
        kw = super(LinceseCase, self).make_solver_keywords()
        # setup delta t for CFL calculation.
        cfldt = self.solver.cfldt
        cfldt = self.execution.time_increment if cfldt is None else cfldt
        kw['cfldt'] = cfldt
        # setup material mapper.
        kw['mtrldict'] = self.solver.mtrldict
        return kw

###############################################################################
# Boundary conditions.
###############################################################################

class LinceseBC(CeseBC):
    """
    Basic BC class for linear CESE solver.
    """

################################################################################
# Anchor.
################################################################################
