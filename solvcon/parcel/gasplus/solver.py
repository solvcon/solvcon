# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
C++-based Gas-dynamics solver.
"""


import time

import numpy as np

import solvcon as sc
from solvcon import march
from solvcon import boundcond


class GasPlusSolver:
    """
    Spatial loops for the gas-dynamics solver.
    """

    ALMOST_ZERO = march.gas.Solver2D.ALMOST_ZERO

    _interface_init_ = march.gas.Solver2D._interface_init_
    _solution_array_ = march.gas.Solver2D._solution_array_

    def __init__(self, blk, **kw):
        # algorithm object.
        solver_type = getattr(march.gas, "Solver%dD"%blk.ndim)
        self.alg = solver_type(blk._ustblk)
        self.blk = blk
        for bc in self.blk.bclist:
            bc.svr = self
        #: 0-based serial number of this solver in a parallel execution.
        self.svrn = self.blk.blkn
        #: Total number of parallel solvers.
        self.nsvr = None
        # marching facilities.
        self.runanchors = sc.MeshAnchorList(self)
        self.marchret = None
        self.der = dict()

    @property
    def bclist(self):
        return self.blk.bclist

    @property
    def neq(self):
        return self.alg.neq

    @property
    def ndim(self):
        return self.alg.block.ndim

    @property
    def nnode(self):
        return self.alg.block.nnode

    @property
    def nface(self):
        return self.alg.block.nface

    @property
    def ncell(self):
        return self.alg.block.ncell

    @property
    def ngstnode(self):
        return self.alg.block.ngstnode

    @property
    def ngstface(self):
        return self.alg.block.ngstface

    @property
    def ngstcell(self):
        return self.alg.block.ngstcell

    @property
    def param(self):
        return self.alg.param

    @property
    def sigma0(self):
        return self.alg.param.sigma0
    @sigma0.setter
    def sigma0(self, value):
        self.alg.param.sigma0 = value

    @property
    def taumin(self):
        return self.alg.param.taumin
    @taumin.setter
    def taumin(self, value):
        self.alg.param.taumin = value

    @property
    def tauscale(self):
        return self.alg.param.tauscale
    @tauscale.setter
    def tauscale(self, value):
        self.alg.param.tauscale = value

    @property
    def state(self):
        return self.alg.state

    @property
    def time(self):
        return self.alg.state.time
    @time.setter
    def time(self, value):
        self.alg.state.time = value

    @property
    def time_increment(self):
        return self.alg.state.time_increment
    @time_increment.setter
    def time_increment(self, value):
        self.alg.state.time_increment = value

    @property
    def step_current(self):
        return self.alg.state.step_current
    @step_current.setter
    def step_current(self, value):
        self.alg.state.step_current = value

    @property
    def step_global(self):
        return self.alg.state.step_global
    @step_global.setter
    def step_global(self, value):
        self.alg.state.step_global = value

    @property
    def substep_run(self):
        return self.alg.state.substep_run
    @substep_run.setter
    def substep_run(self, value):
        self.alg.state.substep_run = value

    @property
    def step_current(self):
        return self.alg.state.step_current
    @step_current.setter
    def step_current(self, value):
        self.alg.state.step_current = value

    @property
    def solution(self):
        return self.alg.sol

    @property
    def sol(self):
        return self.alg.sol.so0c.F

    @property
    def soln(self):
        return self.alg.sol.so0n.F

    @property
    def solt(self):
        return self.alg.sol.so0t.F

    @property
    def dsol(self):
        return self.alg.sol.so1c.F

    @property
    def dsoln(self):
        return self.alg.sol.so1n.F

    @property
    def stm(self):
        return self.alg.sol.stm.F

    @property
    def cfl(self):
        return self.alg.sol.cflc.F

    @property
    def ocfl(self):
        return self.alg.sol.cflo.F

    @property
    def gamma(self):
        return self.alg.sol.gamma.F

   ############################################################################
    # Anchors.
    def provide(self):
        self.runanchors('provide')

    def preloop(self):
        self.runanchors('preloop')

    def postloop(self):
        self.runanchors('postloop')

    def exhaust(self):
        self.runanchors('exhaust')
    # Anchors.
    ############################################################################

    def march(self, time_current, time_increment, steps_run, worker=None):
        """
        :param time_current: Starting time of this set of marching steps.
        :type time_current: float
        :param time_increment: Temporal interval :math:`\Delta t` of the time
            step.
        :type time_increment: float
        :param steps_run: The count of time steps to run.
        :type steps_run: int
        :return: :py:attr:`marchret`

        This method performs time-marching.  The parameters *time_current* and
        *time_increment* are used to reset the instance attributes
        :py:attr:`time` and :py:attr:`time_increment`, respectively.  In each
        invokation :py:attr:`step_current` is reset to 0.

        There is a nested two-level loop in this method for time-marching.  The
        outer loop iterates for time steps, and the inner loop iterates for sub
        time steps.  The outer loop runs *steps_run* times, while the inner
        loop runs :py:attr:`substep_run` times.  In total, the inner loop runs
        *steps_run* \* :py:attr:`substep_run` times.  In each sub time step (in
        the inner loop), the increment of the attribute :py:attr:`time` is
        :py:attr:`time_increment`/:py:attr:`substep_run`.  The temporal
        increment per time step is effectively :py:attr:`time_increment`, with
        a slight error because of round-off.

        Before entering and after leaving the outer loop, :py:meth:`premarch
        <solvcon.anchor.Anchor.premarch>` and :py:meth:`postmarch
        <solvcon.anchor.Anchor.postmarch>` anchors will be run (through the
        attribute :py:attr:`runanchors`).  Similarly, before entering and after
        leaving the inner loop, :py:meth:`prefull
        <solvcon.anchor.Anchor.prefull>` and :py:meth:`postfull
        <solvcon.anchor.Anchor.postfull>` anchors will be run.  Inside the
        inner loop of sub steps, before and after executing all the marching
        methods, :py:meth:`presub <solvcon.anchor.Anchor.presub>` and
        :py:meth:`postsub <solvcon.anchor.Anchor.postsub>` anchors will be run.
        Lastly, before and after invoking every marching method, a pair of
        anchors will be run.  The anchors for a marching method are related to
        the name of the marching method itself.  For example, if a marching
        method is named "calcsome", anchor ``precalcsome`` will be run before
        the invocation, and anchor ``postcalcsome`` will be run afterward.

        Derived classes can set :py:attr:`marchret` dictionary, and
        :py:meth:`march` will return the dictionary at the end of execution.
        The dictionary is reset to empty at the begninning of the execution.
        """
        self.marchret = dict()
        self.step_current = 0
        self.runanchors('premarch')
        while self.step_current < steps_run:
            self.substep_current = 0
            self.runanchors('prefull')
            t0 = time.time()
            while self.substep_current < self.substep_run:
                # set up time.
                self.time = time_current
                self.time_increment = time_increment
                self.runanchors('presub')
                # run marching methods.
                for mmname in self.mmnames:
                    method = getattr(self, mmname)
                    t1 = time.time()
                    self.runanchors('pre'+mmname)
                    t2 = time.time()
                    method(worker=worker)
                    self.runanchors('post'+mmname)
                # increment time.
                time_current += self.time_increment/self.substep_run
                self.time = time_current
                self.time_increment = time_increment
                self.substep_current += 1
                self.runanchors('postsub')
            self.step_global += 1
            self.step_current += 1
            self.runanchors('postfull')
        self.runanchors('postmarch')
        if worker:
            worker.conn.send(self.marchret)
        return self.marchret

    def init(self, **kw):
        for arrname in self._solution_array_:
            arr = getattr(self, arrname)
            arr.fill(self.ALMOST_ZERO) # prevent initializer forgets to set!
        for bc in self.bclist:
            bc.init(**kw)

    def final(self, **kw):
        pass

    def apply_bc(self):
        self.call_non_interface_bc('soln')
        self.call_non_interface_bc('dsoln')

    def call_non_interface_bc(self, name, *args, **kw):
        """
        :param name: Name of the method of BC to call.
        :type name: str
        :return: Nothing.

        Call method of each of non-interface BC objects in my list.
        """
        bclist = [bc for bc in self.bclist
            if not isinstance(bc, boundcond.interface)]
        for bc in bclist:
            try:
                getattr(bc, name)(*args, **kw)
            except Exception as e:
                e.args = tuple([str(bc), name] + list(e.args))
                raise

    ###########################################################################
    # Begin marching algorithm.
    @property
    def mmnames(self):
        """
        Generator of time-marcher names.
        """
        for name in [
            'update',
            'calcsolt',
            'calcsoln',
            'ibcsoln',
            'bcsoln',
            'calccfl',
            'calcdsoln',
            'ibcdsoln',
            'bcdsoln',
        ]:
            yield name

    def update(self, worker=None):
        self.alg.update(self.time, self.time_increment)

    def calcsolt(self, worker=None):
        self.alg.calc_solt()

    def calcsoln(self, worker=None):
        self.alg.calc_soln()

    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    def bcsoln(self, worker=None):
        self.call_non_interface_bc('soln')

    def calccfl(self, worker=None):
        self.alg.calc_cfl()

    def calcdsoln(self, worker=None):
        self.alg.calc_dsoln()

    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)

    def bcdsoln(self, worker=None):
        self.call_non_interface_bc('dsoln')
    # End marching algorithm.
    ###########################################################################

# vim: set ff=unix fenc=utf8 ft=python nobomb et sw=4 ts=4 tw=79:
