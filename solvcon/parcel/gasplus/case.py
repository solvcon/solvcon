# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The control interface.
"""


import solvcon as sc

from . import solver as gpsolver


class GasPlusCase(sc.MeshCase):
    """
    Temporal loop for the gas-dynamic solver.
    """

    defdict = {
        'solver.solvertype': gpsolver.GasPlusSolver,
        'solver.domaintype': sc.Domain,
        # Do no touch the following c-tau parameters.
        'solver.alpha': 1,
        'solver.sigma0': 3.0,
        'solver.taylor': 1.0,
        'solver.cnbfac': 1.0,
        'solver.sftfac': 1.0,
        'solver.taumin': None,
        'solver.tauscale': None,
        # End of c-taw parameters.
        'io.rootdir': sc.env.projdir, # Different default to MeshCase.
    }

    def make_solver_keywords(self):
        kw = super(GasPlusCase, self).make_solver_keywords()
        # time.
        neq = self.blk.ndim + 2
        kw['neq'] = self.execution.neq = neq
        kw['time'] = self.execution.time
        kw['time_increment'] = self.execution.time_increment
        # c-tau scheme parameters.
        kw['alpha'] = int(self.solver.alpha)
        for key in ('sigma0', 'taylor', 'cnbfac', 'sftfac',
                    'taumin', 'tauscale',):
            val = self.solver.get(key)
            if val != None: kw[key] = float(val)
        return kw

# vim: set ff=unix fenc=utf8 ft=python nobomb et sw=4 ts=4 tw=79:
