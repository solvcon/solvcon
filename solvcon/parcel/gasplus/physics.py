# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING


from __future__ import absolute_import, division, print_function


import numpy as np

import solvcon as sc
from solvcon import march


class DensityInitAnchor(sc.MeshAnchor):
    """
    Initialize only density.

    FIXME: Give me doctests.
    """

    def __init__(self, svr, **kw):
        #: Density.
        self.rho = kw.pop('rho') 
        super(DensityInitAnchor, self).__init__(svr, **kw)

    def provide(self):
        self.svr.soln[:,0] = self.rho


class PhysicsAnchor(sc.MeshAnchor):
    """
    Calculates physical quantities for output.  Implements (i) provide() and
    (ii) postfull() methods.

    FIXME: I should be more integrated with :py:class:`~.solver.GasSolver`.

    :ivar gasconst: gas constant.
    :type gasconst: float
    """

    def __init__(self, svr, **kw):
        self.rsteps = kw.pop('rsteps', 1)
        self.gasconst = kw.pop('gasconst', 1.0)
        self.schk = kw.pop('schk', 1.0)
        self.schk0 = kw.pop('schk0', 0.0)
        self.schk1 = kw.pop('schk1', 1.0)
        super(PhysicsAnchor, self).__init__(svr, **kw)

    def provide(self):
        self.svr.alg.qty.update(self.gasconst, self.schk, self.schk0, self.schk1)

    def postfull(self):
        self.svr.alg.qty.update(self.gasconst, self.schk, self.schk0, self.schk1)

# vim: set ff=unix fenc=utf8 ft=python nobomb et sw=4 ts=4 tw=79:
