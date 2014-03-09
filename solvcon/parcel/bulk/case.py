# -*- coding: UTF-8 -*-
#
# Copyright (c) 2013, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
The control interface.
"""


from solvcon import case
from solvcon import domain

from . import solver as localsolver


class BulkCase(case.MeshCase):
    """
    Simulation case for the Navier-Stokes solver based on the bulk modulus.
    """
    defdict = {
        'solver.solvertype': localsolver.BulkSolver,
        'solver.domaintype': domain.Domain,
        'solver.alpha': 1,
        'solver.sigma0': 3.0,
        'solver.taylor': 1,
        'solver.cnbfac': 1.0,
        'solver.sftfac': 1.0,
        'solver.taumin': None,
        'solver.tauscale': None,
        'solver.p0': None,
        'solver.rho0': None,
        'solver.fluids': None,
        'solver.velocities': None,
    }
    def make_solver_keywords(self):
        kw = super(BulkCase, self).make_solver_keywords()
        # time.
        neq = 4 if 3 == self.blk.ndim else 3
        kw['neq'] = self.execution.neq = neq
        kw['time'] = self.execution.time
        kw['time_increment'] = self.execution.time_increment
        # c-tau scheme parameters.
        kw['alpha'] = int(self.solver.alpha)
        for key in ('sigma0', 'taylor', 'cnbfac', 'sftfac',
                    'taumin', 'tauscale',):
            val = self.solver.get(key)
            if val != None: kw[key] = float(val)
        # fluid and solver properties.
        kw['p0'] = self.solver.p0
        kw['rho0'] = self.solver.rho0
        kw['fluids'] = self.solver.fluids
        kw['velocities'] = self.solver.velocities
        return kw

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
