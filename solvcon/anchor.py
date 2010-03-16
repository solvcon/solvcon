# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Anchors attached to the solvers.
"""

class Anchor(object):
    """
    Anchor that called by solver objects at various stages.

    @ivar svr: the solver object to be attached to.
    @itype svr: solvcon.solver.Solver
    @ivar kws: excessive keywords.
    @itype kws: dict
    """

    def __init__(self, svr, **kw):
        from .solver import BaseSolver
        assert isinstance(svr, BaseSolver)
        self.svr = svr
        self.kws = dict(kw)

    def provide(self):
        pass
    def preloop(self):
        pass
    def prefull(self):
        pass
    def postfull(self):
        pass
    def prehalf(self):
        pass
    def premarchsoln(self):
        pass
    def preexsoln(self):
        pass
    def prebcsoln(self):
        pass
    def precfl(self):
        pass
    def premarchdsoln(self):
        pass
    def preexdsoln(self):
        pass
    def prebcdsoln(self):
        pass
    def posthalf(self):
        pass
    def postloop(self):
        pass
    def exhaust(self):
        pass

class AnchorList(list):
    """
    @ivar svr: solver object.
    @itype svr: solvcon.solver.BaseSolver
    """
    def __init__(self, svr, *args, **kw):
        self.svr = svr
        super(AnchorList, self).__init__(*args, **kw)
    def append(self, obj, **kw):
        if isinstance(obj, type):
            obj = obj(self.svr, **kw)
        super(AnchorList, self).append(obj)
    def __call__(self, method):
        """
        Invoke the specified method for each anchor.
        
        @param method: name of the method to run.
        @type method: str
        @return: nothing
        """
        runanchors = self.svr.runanchors
        if method == 'postloop' or method == 'exhaust':
            runanchors = reversed(runanchors)
        for anchor in runanchors:
            getattr(anchor, method)()

class ZeroIAnchor(Anchor):
    """
    Fill the solutions with zero.
    """
    def provide(self):
        self.svr.soln.fill(0.0)
        self.svr.dsoln.fill(0.0)
