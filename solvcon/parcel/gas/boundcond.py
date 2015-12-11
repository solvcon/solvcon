# -*- coding: UTF-8 -*-
#
# Copyright (c), 2014 Yung-Yu Chen <yyc@solvcon.net>
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
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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
Boundary-condition treatments for :py:class:`~.solver.GasSolver`.
"""


from __future__ import absolute_import, division, print_function


import solvcon as sc

# for readthedocs to work.
sc.import_module_may_fail('._algorithm')


class GasBC(sc.BC):
    """
    Base class for all boundary conditions of the gas solver.
    """

    #: Ghost geometry calculator type.
    _ghostgeom_ = None

    def __init__(self, **kw):
        super(GasBC, self).__init__(**kw)
        self.bcd = None

    @property
    def alg(self):
        return self.svr.alg

    def init(self, **kw):
        self.bcd = self.create_bcd()
        getattr(self.alg, 'ghostgeom_'+self._ghostgeom_)(self.bcd)


class GasNonrefl(GasBC):
    _ghostgeom_ = 'mirror'
    def soln(self):
        self.alg.bound_nonrefl_soln(self.bcd)
    def dsoln(self):
        self.alg.bound_nonrefl_dsoln(self.bcd)


class GasWall(GasBC):
    _ghostgeom_ = 'mirror'
    def soln(self):
        self.alg.bound_wall_soln(self.bcd)
    def dsoln(self):
        self.alg.bound_wall_dsoln(self.bcd)


class GasInlet(GasBC):
    vnames = ['rho', 'v1', 'v2', 'v3', 'p', 'gamma']
    vdefaults = {
        'rho': 1.0, 'p': 1.0, 'gamma': 1.4, 'v1': 0.0, 'v2': 0.0, 'v3': 0.0,
    }
    _ghostgeom_ = 'mirror'
    def soln(self):
        self.alg.bound_inlet_soln(self.bcd)
    def dsoln(self):
        self.alg.bound_inlet_dsoln(self.bcd)

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
