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
Logic.
"""


import numpy as np

from solvcon.parcel import linear


class VslinPWSolution(linear.PlaneWaveSolution):
    """Plane-wave solutions for the velocity-stress equations.
    """
    def _calc_eigen(self, **kw):
        wvec = kw['wvec']
        mtrl = kw['mtrl']
        idx = kw['idx']
        nml = wvec/np.sqrt((wvec**2).sum())
        jacos = mtrl.get_jacos()
        jaco = jacos[0] * nml[0]
        for idm in range(1, len(nml)):
            jaco += jacos[idm] * nml[idm]
        evl, evc = np.linalg.eig(jaco)
        srt = evl.argsort()
        evl = evl[srt[idx]].real
        evc = evc[:,srt[idx]].real
        evc *= evc[0]/abs(evc[0]+1.e-200)
        return evl, evc


class VslinSolver(linear.LinearSolver):
    """Basic elastic solver.
    """

    def __init__(self, blk, mtrldict=None, **kw):
        kw['neq'] = 9
        #: A :py:class:`dict` that maps names to :py:class:`Material
        #: <.material.Material>` object.
        self.mtrldict = mtrldict if mtrldict else {}
        #: A :py:class:`list` of all :py:class:`Material <.material.Material>`
        #: objects.
        self.mtrllist = None
        super(VslinSolver, self).__init__(blk, **kw)

    @property
    def gdlen(self):
        return 9 * 9 * self.ndim

    def _make_grpda(self):
        self.mtrllist = self._build_mtrllist(self.grpnames, self.mtrldict)
        for igrp in range(len(self.grpnames)):
            mtrl = self.mtrllist[igrp]
            jaco = self.grpda[igrp].reshape(self.neq, self.neq, self.ndim)
            mjacos = mtrl.get_jacos()
            for idm in range(self.ndim):
                jaco[:,:,idm] = mjacos[idm,:,:]

    @staticmethod
    def _build_mtrllist(grpnames, mtrldict):
        """
        Build the material list out of the mapping dict.

        @type grpnames: list
        @param mtrldict: the map from names to material objects.
        @type mtrldict: dict
        @return: the list of material object.
        @rtype: Material
        """
        mtrllist = list()
        default_mtuple = mtrldict.get(None, None)
        for grpname in grpnames:
            try:
                mtrl = mtrldict.get(grpname, default_mtuple)
            except KeyError, e:
                args = e.args[:]
                args.append('no material named %s in mtrldict'%grpname)
                e.args = args
                raise
            mtrllist.append(mtrl)
        return mtrllist


class VslinCase(linear.LinearCase):
    """Case for anisotropic elastic solids.
    """
    defdict = {
        'execution.neq': 9,
        'solver.solvertype': VslinSolver,
        'solver.mtrldict': dict,
    }

    def make_solver_keywords(self):
        kw = super(VslinCase, self).make_solver_keywords()
        # setup material mapper.
        kw['mtrldict'] = self.solver.mtrldict
        return kw

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
