# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
C++-based Gas-dynamics solver.
"""


import solvcon as sc


class GasPlusSolver:
    """
    SOLVCON compatibility layer for :py:class:`solvcon.march.gas.Solver2D` or
    :py:class:`solvcon.march.gas.Solver3D`.
    """

    ALMOST_ZERO = sc.march.gas.Solver2D.ALMOST_ZERO

    _interface_init_ = sc.march.gas.Solver2D._interface_init_
    _solution_array_ = sc.march.gas.Solver2D._solution_array_

    def __init__(self, blk, sigma0, time, time_increment, **kw):
        #: 0-based serial number of this solver in a parallel execution.
        self.svrn = blk.blkn
        #: Total number of parallel solvers.
        self.nsvr = None
        # algorithm object.
        solver_type = getattr(sc.march.gas, "Solver%dD"%blk.ndim)
        self.alg = solver_type(blk._ustblk)
        # trims.
        trims = []
        for bc in blk.bclist:
            name = bc.__class__.__name__.lstrip("GasPlus")
            trim_type = getattr(sc.march.gas, "Trim%s%dD" % (name, blk.ndim))
            trim = trim_type(self.alg, bc._data)
            trims.append(trim)
        self.trims = trims
        # parameters and state.
        self.param.sigma0 = sigma0
        self.state.time = time
        self.state.time_increment = time_increment

    @property
    def solver(self):
        return self.alg

    @property
    def runanchors(self):
        return self.alg.anchors

    @property
    def neq(self):
        return self.alg.neq

    @property
    def block(self):
        return self.alg.block

    @property
    def trims(self):
        return self.alg.trims
    @trims.setter
    def trims(self, val):
        self.alg.trims = val

    @property
    def param(self):
        return self.alg.param

    @property
    def state(self):
        return self.alg.state

    @property
    def sol(self):
        return self.alg.sol

    @property
    def qty(self):
        return self.alg.qty

    def provide(self):
        self.runanchors.provide()

    def preloop(self):
        self.runanchors.preloop()

    def postloop(self):
        self.runanchors.postloop()

    def exhaust(self):
        self.runanchors.exhaust()

    def march(self, time_current, time_increment, steps_run, worker=None):
        self.alg.march(time_current, time_increment, steps_run)
        state = self.state
        marchret = {'cfl': [state.cfl_min, state.cfl_max, state.cfl_nadjusted,
                            state.cfl_nadjusted_accumulated]}
        if worker:
            worker.conn.send(marchret)
        return marchret

    def init(self, **kw):
        for arrname in ("so0c", "so0n", "so0t", "so1c", "so1n"):
            arr = getattr(self.sol, arrname)
            arr.fill(self.ALMOST_ZERO) # prevent initializer forgets to set!

    def final(self, **kw):
        pass

    def apply_bc(self):
        self.alg.trim_do0()
        self.alg.trim_do1()

# vim: set ff=unix fenc=utf8 ft=python nobomb et sw=4 ts=4 tw=79:
