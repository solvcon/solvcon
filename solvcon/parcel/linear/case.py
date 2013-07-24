# -*- coding: UTF-8 -*-
#
# Copyright (C) 2012-2013 Yung-Yu Chen <yyc@solvcon.net>.
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
The control interface.
"""


from solvcon import case
from solvcon import domain

from . import solver as lsolver


class LinearCase(case.MeshCase):
    """
    Basic case with linear CESE method.
    """
    defdict = {
        'execution.verified_norm': -1.0,
        'solver.solvertype': lsolver.LinearSolver,
        'solver.domaintype': domain.Domain,
        'solver.alpha': 0,
        'solver.sigma0': 3.0,
        'solver.taylor': 1.0,
        'solver.cnbfac': 1.0,
        'solver.sftfac': 1.0,
        'solver.taumin': None,
        'solver.tauscale': None,
    }
    def make_solver_keywords(self):
        kw = super(LinearCase, self).make_solver_keywords()
        # time.
        kw['time'] = self.execution.time
        kw['time_increment'] = self.execution.time_increment
        # c-tau scheme parameters.
        kw['alpha'] = int(self.solver.alpha)
        for key in ('sigma0', 'taylor', 'cnbfac', 'sftfac',
                    'taumin', 'tauscale',):
            val = self.solver.get(key)
            if val != None: kw[key] = float(val)
        return kw


# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
