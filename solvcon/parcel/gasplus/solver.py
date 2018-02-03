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
from solvcon import boundcond


class GasPlusSolver:
    """
    Spatial loops for the gas-dynamics solver.
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

   ############################################################################
    # Anchors.
    def provide(self):
        self.runanchors.provide()

    def preloop(self):
        self.runanchors.preloop()

    def postloop(self):
        self.runanchors.postloop()

    def exhaust(self):
        self.runanchors.exhaust()
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
        state = self.state
        state.step_current = 0
        self.runanchors.premarch()
        while state.step_current < steps_run:
            state.substep_current = 0
            self.runanchors.prefull()
            t0 = time.time()
            while state.substep_current < state.substep_run:
                # set up time.
                state.time = time_current
                state.time_increment = time_increment
                self.runanchors.presub()
                # run marching methods.
                for mmname in self.mmnames:
                    method = getattr(self, mmname)
                    method(worker=worker)
                # increment time.
                time_current += state.time_increment/state.substep_run
                state.time = time_current
                state.time_increment = time_increment
                state.substep_current += 1
                self.runanchors.postsub()
            state.step_global += 1
            state.step_current += 1
            self.runanchors.postfull()
        self.runanchors.postmarch()
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
        self.call_non_interface_bc('do0')
        self.call_non_interface_bc('do1')

    def call_non_interface_bc(self, name, *args, **kw):
        """
        :param name: Name of the method of BC to call.
        :type name: str
        :return: Nothing.

        Call method of each of non-interface Trim objects in my list.
        """
        iftype = getattr(sc.march.gas, "TrimInterface%dD" % (self.block.ndim))
        for trim in self.trims:
            if isinstance(trim, iftype):
                continue
            try:
                method = getattr(trim, "apply_"+name)
                method(*args, **kw)
            except Exception as e:
                e.args = tuple([str(trim), name] + list(e.args))
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
        self.alg.update(self.state.time, self.state.time_increment)

    def calcsolt(self, worker=None):
        self.alg.calc_solt()

    def calcsoln(self, worker=None):
        self.alg.calc_soln()

    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    def bcsoln(self, worker=None):
        self.call_non_interface_bc('do0')

    def calccfl(self, worker=None):
        self.alg.calc_cfl()

    def calcdsoln(self, worker=None):
        self.alg.calc_dsoln()

    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)

    def bcdsoln(self, worker=None):
        self.call_non_interface_bc('do1')
    # End marching algorithm.
    ###########################################################################

# vim: set ff=unix fenc=utf8 ft=python nobomb et sw=4 ts=4 tw=79:
