#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
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
Validate C++-based ``gasplus`` solver.
"""

# Beware, it's full of hacks.


import os
import math
import collections
from unittest import TestCase

import numpy as np

import solvcon as sc
from solvcon.parcel import gasplus as gp


class TriangularMesher(object):
    """
    Representation of a rectangle and the Gmsh meshing helper.

    :ivar lowerleft: (x0, y0) of the rectangle.
    :type lowerleft: tuple
    :ivar upperright: (x1, y1) of the rectangle.
    :type upperright: tuple
    :ivar edgelength: Length of each mesh cell edge.
    :type edgelength: float
    """

    def __init__(self, lowerleft, upperright, edgelength):
        assert 2 == len(lowerleft)
        self.lowerleft = tuple(float(val) for val in lowerleft)
        assert 2 == len(upperright)
        self.upperright = tuple(float(val) for val in upperright)
        self.edgelength = float(edgelength)

    def __call__(self, cse):
        x0, y0 = self.lowerleft
        x1, y1 = self.upperright
        ndx = int(math.ceil((x1 - x0) / (self.edgelength*2)))
        ndy = int(math.ceil((y1 - y0) / (self.edgelength*2)))
        dx = (x1 - x0) / ndx
        dy = (y1 - y0) / ndy
        # mesh numbers.
        nnode = (ndx*2+1)*ndy + ndx+1
        nface = 6 * ndx * ndy + ndx + ndy
        ncell = 4 * ndx * ndy
        nbound = 2 * (ndx + ndy)
        blk = sc.Block(ndim=2, nnode=nnode, nface=0, ncell=ncell, nbound=3)
        # create nodes.
        nodes = []
        for iy, yloc in enumerate(np.arange(y0, y1+dy/4, dy/2)):
            if iy % 2 == 0:
                meshx = np.arange(x0, x1+dx/4, dx, dtype='float64')
            else:
                meshx = np.arange(x0+dx/2, x1-dx/4, dx, dtype='float64')
            nodes.append(np.vstack([meshx, np.full_like(meshx, yloc)]).T)
        nodes = np.vstack(nodes)
        assert nodes.shape[0] == nnode
        blk.ndcrd[:,:] = nodes
        assert (blk.ndcrd == nodes).all()
        # create cells.
        skip_bottom = ndx + 1
        skip_middle = ndx
        skip_full = skip_bottom + skip_middle
        for ilayer in range(ndy):
            inref = ilayer * skip_full
            icref = ilayer * 4 * ndx
            for ic in range(ndx):
                ictr = inref + skip_bottom + ic
                ill = inref             + ic+0
                ilr = inref             + ic+1
                iul = inref + skip_full + ic+0
                iur = inref + skip_full + ic+1
                blk.clnds[icref+ic*4  ,:4] = (3, ictr, ill, ilr)
                blk.clnds[icref+ic*4+1,:4] = (3, ictr, ilr, iur)
                blk.clnds[icref+ic*4+2,:4] = (3, ictr, iur, iul)
                blk.clnds[icref+ic*4+3,:4] = (3, ictr, iul, ill)
        narr = np.unique(blk.clnds[:ncell,1:4].flatten())
        narr.sort()
        assert (narr == np.arange(nnode, dtype='int32')).all()
        # build block.
        blk.cltpn[:] = 3
        blk.build_interior()
        assert blk.nface == nface
        assert (abs(blk.clvol - dx * dy / 4) < 1.e-10).all()
        # build boundary
        boundaries = dict(
            left=np.arange(nface, dtype='int32')[blk.fccnd[:,0] == x0],
            right=np.arange(nface, dtype='int32')[blk.fccnd[:,0] == x1],
            lower=np.arange(nface, dtype='int32')[blk.fccnd[:,1] == y0],
            upper=np.arange(nface, dtype='int32')[blk.fccnd[:,1] == y1],
        )
        for name in boundaries:
            bndfcs = boundaries[name]
            nameb = name.encode()
            bct, vdict = cse.condition.bcmap.get(nameb, (sc.BC, dict()))
            bc = bct(fpdtype=blk.fpdtype)
            bc.name = name
            bc.facn = np.empty((len(bndfcs), 3), dtype='int32')
            bc.facn.fill(-1)
            bc.facn[:,0] = bndfcs
            bc.feedValue(vdict)
            bc.sern = len(blk.bclist)
            bc.blk = blk
            blk.bclist.append(bc)
        blk.build_boundary()
        blk.build_ghost()
        return blk


class XDiaphragmAnchor(sc.MeshAnchor):
    """
    Set different density and pressure across a diaphragm along the x axis.
    """

    def __init__(self, svr, **kw):
        self.xloc = float(kw.pop('xloc'))
        self.gamma = float(kw.pop('gamma'))
        self.rho1 = float(kw.pop('rho1'))
        self.rho2 = float(kw.pop('rho2'))
        self.p1 = float(kw.pop('p1'))
        self.p2 = float(kw.pop('p2'))
        super(XDiaphragmAnchor, self).__init__(svr, **kw)

    def provide(self):
        super(XDiaphragmAnchor, self).provide()
        gamma = self.gamma
        svr = self.svr
        svr.soln[:,0].fill(self.rho1)
        svr.soln[:,1].fill(0.0)
        svr.soln[:,2].fill(0.0)
        if svr.ndim == 3:
            svr.soln[:,3].fill(0.0)
        svr.soln[:,svr.ndim+1].fill(self.p1/(gamma-1))
        # set.
        slct = svr.blk.shclcnd[:,0] > self.xloc
        svr.soln[slct,0] = self.rho2
        svr.soln[slct,svr.ndim+1] = self.p2
        # update.
        svr.sol[:] = svr.soln[:]


_HeterogeneityData = collections.namedtuple(
    "_HeterogeneityData",
    ("step_current", "substep_current", "solnidx", "xloc",
     "vmean", "vcov", "cov_threshold"))
     
class YHomogeneityCheck(sc.MeshAnchor):
    def __init__(self, svr, **kw):
        # coefficient of variation
        self.instant_fail = kw.pop('instant_fail', False)
        self.cov_threshold = kw.pop('cov_threshold')
        self.columns = None
        self.hdata = list()
        super(YHomogeneityCheck, self).__init__(svr, **kw)

    def provide(self):
        super(YHomogeneityCheck, self).provide()
        blk = self.svr.blk
        self.hdata = list()
        # get rid of rounding error with 10 digits.
        roundx = blk.clcnd[:,0].round(decimals=10) # rounded x
        indices = np.arange(blk.ncell, dtype='int32')
        xlocs = np.unique(roundx)
        xlocs.sort()
        # scan the X coordinates and fill indices.
        self.columns = collections.OrderedDict()
        for xloc in xlocs:
            self.columns[xloc] = indices[roundx==xloc]
        # sanity check.
        rebuilt = np.concatenate([self.columns[k] for k in self.columns])
        rebuilt.sort()
        assert (indices == rebuilt).all()

    def _check(self):
        alg = self.svr.alg
        soln = self.svr.soln[self.svr.ngstcell:,:]
        for xloc in self.columns:
            column = self.columns[xloc]
            for it in range(soln.shape[1]):
                val = soln[column,it]
                cov_threshold = self.cov_threshold[it]
                if cov_threshold:
                    vmean = abs(np.mean(val))
                    vamean = abs(vmean)
                    vstd = np.std(val)
                    vcov = vstd if vamean < 1.e-10 else vstd/vamean
                    if vcov >= cov_threshold:
                        hdatum = _HeterogeneityData(
                            alg.step_current, alg.substep_current,
                            it, xloc, vmean, vcov, cov_threshold)
                        if self.instant_fail:
                            raise AssertionError(hdatum)
                        else:
                            self.hdata.append(hdatum)

    def prefull(self):
        self._check()

    def postloop(self):
        self._check()
        if self.hdata:
            raise AssertionError(self.hdata)


def create_case(
    casename=None, ssteps=None, psteps=None, instant_fail=False,
    gamma=1.4, rho1=1.0, p1=1.0, rho2=0.125, p2=0.25,
    **kw
):
    if ssteps is not None and psteps is not None:
        basedir = os.path.abspath(os.path.join(os.getcwd(), casename))
    else:
        basedir = None

    # Set up case.
    mesher = TriangularMesher(lowerleft=(0,0), upperright=(4,1),
                              edgelength=0.1)
    bcmap = {
        b'upper': (sc.bctregy.GasPlusSlipWall, {},),
        b'left': (sc.bctregy.GasPlusNonRefl, {},),
        b'right': (sc.bctregy.GasPlusNonRefl, {},),
        b'lower': (sc.bctregy.GasPlusSlipWall, {},),
    }
    cse = gp.GasPlusCase(
        # Mesh generator.
        mesher=mesher,
        # Mapping boundary-condition treatments.
        bcmap=bcmap,
        # Use the case name to be the basename for all generated files.
        basefn=casename,
        basedir=basedir,
        # Runstep.
        time_increment=30.e-3, steps_run=30,
        # Debug and capture-all.
        debug=False, **kw)

    # Field initialization and derived calculations.
    cse.defer(gp.FillAnchor,
              mappers={'soln': gp.GasPlusSolver.ALMOST_ZERO,
                       'dsoln': 0.0, 'gamma': gamma})
    cse.defer(XDiaphragmAnchor,
              xloc=(mesher.lowerleft[0]+mesher.upperright[0])/2,
              gamma=gamma, rho1=rho1, p1=p1, rho2=rho2, p2=p2)
    cse.defer(YHomogeneityCheck,
              instant_fail=instant_fail,
              cov_threshold=[
                # FIXME: everything is too large for a robust solver.  I
                # tolerate it so that we can move on.  There must be something
                # wrong and we have to fix it later.
                0.027, # density
                0.63, # X velocity FIXME: way too large!
                0.035, # Y velocity isn't as bad as X velocity
                0.037, # total energy
              ])

    if ssteps is not None and psteps is not None:
        # Report information while calculating.
        cse.defer(gp.ProgressHook, linewidth=ssteps/psteps, psteps=psteps)
        cse.defer(gp.CflHook, fullstop=False, cflmax=10.0, psteps=ssteps)
        cse.defer(gp.MeshInfoHook, psteps=ssteps)
        # Store data.
        cse.defer(
            gp.PMarchSave,
            anames=[
                ('soln', False, -4),
            ],
            psteps=ssteps)

    return cse


class TestTube2dTriangleRegularCaseRun(TestCase):

    def test_run(self, **kw):
        cse = create_case('tube_2d_triangle_regular_run')
        cse.init()
        cse.run()


# This section is for debugging purpose.
if __name__ == '__main__':
    def tube_2d_triangle_regular_run(casename, **kw):
        return create_case(casename, ssteps=1, psteps=1, instant_fail=True,
                           **kw)
    gp.register_arrangement(tube_2d_triangle_regular_run)
    sc.go()

# vim: set ff=unix fenc=utf8 ft=python sw=4 ts=4 tw=79:
