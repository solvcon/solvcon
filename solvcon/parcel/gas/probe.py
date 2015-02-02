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


import os

import numpy as np

import solvcon as sc


class Probe(object):
    """
    Represent a point in the mesh.
    """

    def __init__(self, *args, **kw):
        self.speclst = kw.pop('speclst')
        self.name = kw.pop('name', None)
        self.crd = np.array(args, dtype='float64')
        self.pcl = -1
        self.vals = list()

    def __str__(self):
        crds = ','.join(['%g'%val for val in self.crd])
        return 'Pt/%s#%d(%s)%d' % (self.name, self.pcl, crds, len(self.vals))

    def locate_cell(self, svr):
        icl, ifl, jcl, jfl = svr.alg.locate_point(self.crd)
        self.pcl = icl

    def __call__(self, svr, time):
        ngstcell = svr.ngstcell
        vlist = [time]
        for spec in self.speclst:
            arr = None
            if isinstance(spec, str):
                arr = svr.der[spec]
            elif isinstance(spec, int):
                if spec >= 0 and spec < svr.neq:
                    arr = svr.soln[:,spec]
                elif spec < 0 and -1-spec < svr.neq:
                    spec = -1-spec
                    arr = svr.sol[:,spec]
            if arr is None:
                raise IndexError, 'spec %s incorrect'%str(spec)
            vlist.append(arr[ngstcell+self.pcl])
        self.vals.append(vlist)


class ProbeAnchor(sc.MeshAnchor):
    """
    Anchor for probe.
    """

    def __init__(self, svr, **kw):
        speclst = kw.pop('speclst')
        self.points = list()
        for data in kw.pop('coords'):
            pkw = {'speclst': speclst, 'name': data[0]}
            self.points.append(Probe(*data[1:], **pkw))
        super(ProbeAnchor, self).__init__(svr, **kw)

    def preloop(self):
        for point in self.points: point.locate_cell(self.svr)
        for point in self.points: point(self.svr, self.svr.time)

    def postfull(self):
        for point in self.points: point(self.svr, self.svr.time)


class ProbeHook(sc.MeshHook):
    """
    Point probe.
    """

    def __init__(self, cse, **kw):
        self.name = kw.pop('name', 'ppank')
        super(ProbeHook, self).__init__(cse, **kw)
        self.ankkw = kw
        self.points = None

    def drop_anchor(self, svr):
        ankkw = self.ankkw.copy()
        ankkw['name'] = self.name
        self._deliver_anchor(svr, ProbeAnchor, ankkw)

    def _collect(self):
        cse = self.cse
        if cse.is_parallel:
            dom = cse.solver.domainobj
            dealer = cse.solver.dealer
            allpoints = list()
            for iblk in range(dom.nblk):
                dealer[iblk].cmd.pullank(self.name, 'points', with_worker=True)
                allpoints.append(dealer[iblk].recv())
            npt = len(allpoints[0])
            points = [None]*npt
            for rpoints in allpoints:
                ipt = 0
                while ipt < npt:
                    if points[ipt] == None and rpoints[ipt].pcl >=0:
                        points[ipt] = rpoints[ipt]
                    ipt += 1
        else:
            svr = self.cse.solver.solverobj
            points = [pt for pt in svr.runanchors[self.name].points
                if pt.pcl >= 0]
        self.points = points

    def postmarch(self):
        psteps = self.psteps
        istep = self.cse.execution.step_current
        if istep%psteps != 0: return False
        self._collect()
        return True

    def postloop(self):
        for point in self.points:
            ptfn = '%s_pt_%s_%s.npy' % (
                self.cse.io.basefn, self.name, point.name)
            ptfn = os.path.join(self.cse.io.basedir, ptfn)
            np.save(ptfn, np.array(point.vals, dtype='float64'))

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
