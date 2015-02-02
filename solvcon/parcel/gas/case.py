# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014, Yung-Yu Chen <yyc@solvcon.net>
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

import solvcon as sc

from . import solver as gassolver


class GasCase(sc.MeshCase):
    """
    Temporal loop for the gas-dynamic solver.
    """

    defdict = {
        'solver.solvertype': gassolver.GasSolver,
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
        kw = super(GasCase, self).make_solver_keywords()
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

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
