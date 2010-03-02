# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
One-dimensional solvers.
"""

from ..dependency import FortranType
from .core import BaseSolver

class OnedimSolverExeinfo(FortranType):
    """
    Execution information for BlockSolver.
    """

    _fortran_name_ = 'execution'
    from ctypes import c_int, c_double
    _fields_ = [
        ('dnx', c_int),
        ('neq', c_int), ('nmtrl', c_int),
        ('dnstep', c_int),
        ('time', c_double), ('time_increment', c_double),
        ('maxcfl', c_double),
    ]
    del c_int, c_double

class OnedimSolver(BaseSolver):
    """
    Generic class for one-dimensional sequential solvers.

    @ivar _march:
    @ivar _march_args:
    @ivar info:
    @ivar xgrid:
    @ivar xmtrl:
    @ivar xmid:
    @ivar sol:
    @ivar dsol:
    @ivar cfl:
    """

    _pointers_ = ['_march', '_march_args']  # for binder.
    _clib_solve = None  # subclass should override.

    def __init__(self, xgrid, xmtrl, **kw):
        """
        @keyword neq: number of equations (variables).
        @type neq: int
        """
        assert len(xgrid.shape) == 1
        assert xgrid.shape[0] == xmtrl.shape[0]
        dnx = xgrid.shape[0]
        assert dnx%2 == 1
        from numpy import empty
        neq = kw.pop('neq')
        super(OnedimSolver, self).__init__(**kw)
        self.exn = OnedimSolverExeinfo(
            dnx=dnx,
            neq=neq, nmtrl=xmtrl.max()+1,
            dnstep=2,
            time=0.0, time_increment=0.0,   # just placeholder for marchers.
        )
        # placeholders.
        self._march = None
        self._march_args = None
        # arrays.
        self.info = None    # placeholder.
        self.xgrid = xgrid.copy()
        self.xmtrl = xmtrl.copy()
        ## middle points.
        xmid = xgrid.copy()
        xmid[2:-2:2] = (xgrid[1:-2:2] + xgrid[3::2])/2
        xmid[1:-1:2] = (xgrid[0:-1:2] + xgrid[2::2])/2
        self.xmid = xmid
        ## solutions.
        self.sol = empty((dnx, neq), dtype=self.fpdtype)
        self.dsol = empty((dnx, neq), dtype=self.fpdtype)
        self.cfl = empty(dnx, dtype=self.fpdtype)

    def march(self, time, time_increment):
        """
        March the solution U vector in the solver and BCs.

        @return: maximum CFL number.
        """
        self.exn.time = time
        self.exn.time_increment = time_increment
        self.exn.dnstep = 2
        # march U.
        self._march(*self._march_args)
        return self.exn.maxcfl
