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


import numpy as np

import solvcon as sc


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

    _varlist_ = ['v', 'rho', 'p', 'T', 'ke', 'a', 'M', 'sch']

    def __init__(self, svr, **kw):
        self.rsteps = kw.pop('rsteps', 1)
        self.gasconst = kw.pop('gasconst', 1.0)
        self.schk = kw.pop('schk', 1.0)
        self.schk0 = kw.pop('schk0', 0.0)
        self.schk1 = kw.pop('schk1', 1.0)
        super(PhysicsAnchor, self).__init__(svr, **kw)

    def _calculate_physics(self):
        svr = self.svr
        der = svr.der
        svr.alg.process_physics(
            self.gasconst, der['v'], der['w'], der['wm'],
            der['rho'], der['p'], der['T'], der['ke'], der['a'], der['M'])

    def _calculate_schlieren(self):
        from ctypes import byref, c_double
        svr = self.svr
        sch = svr.der['sch']
        svr.alg.process_schlieren_rhog(sch)
        svr.alg.process_schlieren_sch(self.schk, self.schk0, self.schk1, sch)

    def provide(self):
        svr = self.svr
        der = svr.der
        nelm = svr.ngstcell + svr.ncell
        der['v'] = np.zeros((nelm, svr.ndim), dtype='float64')
        der['w'] = np.zeros((nelm, svr.ndim), dtype='float64')
        der['wm'] = np.zeros(nelm, dtype='float64')
        der['rho'] = np.zeros(nelm, dtype='float64')
        der['p'] = np.zeros(nelm, dtype='float64')
        der['T'] = np.zeros(nelm, dtype='float64')
        der['ke'] = np.zeros(nelm, dtype='float64')
        der['a'] = np.zeros(nelm, dtype='float64')
        der['M'] = np.zeros(nelm, dtype='float64')
        der['sch'] = np.zeros(nelm, dtype='float64')
        self._calculate_physics()
        self._calculate_schlieren()

    def postfull(self):
        svr = self.svr
        istep = self.svr.step_global
        rsteps = self.rsteps
        if istep > 0 and istep%rsteps == 0:
            self._calculate_physics()
            self._calculate_schlieren()

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
