#!/usr/bin/env python2.7
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2010, Yung-Yu Chen <yyc@solvcon.net>
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
See README.
"""


import sys
import os
import math
import glob
import pickle
import optparse
import functools

import numpy as np

import solvcon
from solvcon import conf
from solvcon import cmdutil
from solvcon import boundcond
from solvcon import solver
from solvcon import helper
from solvcon.parcel import vewave




###############################################################################
# Command line.
###############################################################################
class converge(cmdutil.Command):
    """
    Calculate and verify convergence.

    Must supply <delta> <M1>.
    """
    min_args = 0

    def __init__(self, env):
        super(converge, self).__init__(env)
        op = self.op

        opg = optparse.OptionGroup(op, 'Convergence')
        opg.add_option("--wdir", action="store",
            dest="wdir", default="result", help="Working directory.")
        opg.add_option("--key", action="store",
            dest="key", default="L2", help="Linf or L2 norm.")
        opg.add_option("--idx", action="store", type=int,
            dest="idx", default=0, help="Index of variable: 0--8.")
        opg.add_option("--order", action="store", type=float,
            dest="order", default=None,
            help="Error-norm should converge at the rate, if given.")
        opg.add_option("--order-tolerance", action="store", type=float,
            dest="order_tolerance", default=0.4,
            help="The variation of converge order which can be tolerated.")
        opg.add_option("--stop-on-over", action="store_true",
            dest="stop_on_over", default=False,
            help="Raise ValueError if tolerance not met.")
        op.add_option_group(opg)
        self.opg_obshock = opg

    def _convergence_test(self, mainfn):
        ops, args = self.opargs
        # collect data.
        mms = reversed(sorted([int(txt.split('_')[1]) for txt in 
            glob.glob(os.path.join(ops.wdir, '%s_*_norm.pickle'%mainfn))]))
        dat = [(mm, pickle.load(open(os.path.join(ops.wdir,
                '%s_%d_norm.pickle'%(mainfn, mm))))) for mm in mms]
        # show convergence.
        sys.stdout.write(
            '%s convergence of %s error-norm at the %dth (0--8) variable:\n' % (
            mainfn, ops.key, ops.idx))
        for ih in range(1, len(dat)):
            er = [dat[it][1][ops.key][ops.idx] for it in range(ih-1, ih+1)]
            hr = [float(dat[it][0])/1000 for it in range(ih-1, ih+1)]
            odr = math.log(er[1]/er[0])/math.log(hr[1]/hr[0])
            sys.stdout.write('  %6.4f -> %6.4f (m): %g' % (hr[0], hr[1], odr))
            if ops.order is not None:
                if odr - ops.order > -ops.order_tolerance:
                    sys.stdout.write(' GOOD. Larger than')
                else:
                    if ops.stop_on_over:
                        raise ValueError('out of tolerance')
                    else:
                        sys.stdout.write(' BAD. Out of')
                sys.stdout.write(
                    ' %g - %g = %g' % (
                        ops.order, ops.order_tolerance,
                        ops.order - ops.order_tolerance))
            sys.stdout.write('\n')

    def __call__(self):
        self._convergence_test('cvg2d')
        self._convergence_test('cvg3d')


################################################################################
# Mesh generation and boundary condition processor.
################################################################################
def mesher(cse, use_cubit=False):
    """
    Generate meshes from template files.  Note Gmsh doesn't always generate the
    same mesh with the same input.
    """
    # get dimensionality.
    ndim = int(cse.io.basefn[3])
    # determine meshing template file name.
    tmplfn = '%s.%s.tmpl' % ('cube' if 3 == ndim else 'square',
                             'cubit' if use_cubit else 'gmsh')
    # determine characteristic length of mesh.
    try:
        itv = float(cse.io.basefn.split('_')[-1])/1000
    except ValueError:
        itv = 0.2
    # load the meshing commands.
    cmds = open(tmplfn).read() % itv
    cmds = [cmd.strip() for cmd in cmds.strip().split('\n')]
    # make the original mesh object.
    mobj = helper.Cubit(cmds, ndim)() if use_cubit else helper.Gmsh(cmds)()
    # convert the mesh to block.
    blk = mobj.toblock(bcname_mapper=cse.condition.bcmap,
                       use_incenter=cse.solver.use_incenter)
    # return the converted block.
    return blk

def match_periodic(blk):
    """
    Match periodic boundary condition.
    """
    bct = boundcond.bctregy.VewavePeriodic
    bcmap = dict()
    val = -2
    bcmap.update({
        'left': (
            bct, {
                'link': 'right',
                'ref': np.array(
                    [0,val,val] if blk.ndim == 3 else [0,val], dtype='float64')
            }
        ),
    })
    bcmap.update({
        'lower': (
            bct, {
                'link': 'upper',
                'ref': np.array(
                    [val,0,val] if blk.ndim == 3 else [val,0], dtype='float64')
            }
        ),
    })
    if blk.ndim == 3:
        bcmap.update({
            'rear': (
                bct, {
                    'link': 'front',
                    'ref': np.array([val,val,0], dtype='float64'),
                }
            ),
        })
    bct.couple_all(blk, bcmap)


################################################################################
# Basic configuration.
################################################################################
def cvg_base(casename=None, mtrlname='SoftTissue', meshname=None,
    psteps=None, ssteps=None, rho=None, vp=None, sig0=None, freq=None, **kw):
    """
    Fundamental configuration of the simulation and return the case object.

    @return: the created Case object.
    @rtype: solvcon.case.BlockCase
    """
    from . import rootdir
    ndim = int(casename[3])
    # set up BCs. Options:
    # a. boundcond.bctregy.VewaveNonRefl
    # b. boundcond.bctregy.VewaveLongSineX
    bct = boundcond.bctregy.VewaveNonRefl
    longsinex = boundcond.bctregy.VewaveLongSineX
    bcmap = dict()
    bcmap.update({
        'left': (longsinex, {}),
        'right': (bct, {}),
        'upper': (bct, {}),
        'lower': (bct, {}),
    })
    if ndim == 3:
        bcmap.update({
            'front': (bct, {}),
            'rear': (bct, {}),
        })
    # set up case.
    mtrl = vewave.mltregy[mtrlname]()
    #basedir = os.path.join(os.path.abspath(os.getcwd()), 'result')
    if conf.env.command and conf.env.command.opargs[0].basedir:
        basedir = os.path.abspath(conf.env.command.opargs[0].basedir)
    else:
        basedir = os.path.abspath(os.getcwd())
    meshfn = os.path.join(rootdir, 'mesh', meshname)
    #local_mesher = functools.partial(
    #    mesher, use_cubit=os.environ.get('USE_CUBIT', False))
    cse = vewave.VewaveCase(
        basedir=basedir, rootdir=conf.env.projdir, basefn=casename,
        #mesher=local_mesher,
        meshfn=meshfn,
        bcmap=bcmap,
        mtrldict={None: mtrl}, taylor=0.0,
        use_incenter=False, **kw)
    # informative hooks.
    cse.runhooks.append(vewave.MeshInfoHook)
    cse.runhooks.append(vewave.ProgressHook, psteps=psteps,
        linewidth=ssteps/psteps)
    cse.runhooks.append(vewave.CflHook, fullstop=False, psteps=ssteps,
        cflmax=10.0, linewidth=ssteps/psteps)
    # initializer anchors.
    cse.runhooks.append(vewave.FillAnchor,
                        mappers={'soln': 0.0, #solver.ALMOST_ZERO, 
                        'dsoln': 0.0})
    cse.runhooks.append(vewave.AmscaAnchor, rho=rho, vp=vp, sig0=sig0, freq=freq)
    # analyzing/output anchors and hooks.
    cse.runhooks.append(vewave.PMarchSave, anames=[
            ('s11', True, 0),
            ('s22', True, 0),
            ('s33', True, 0),
            ('s23', True, 0),
            ('s13', True, 0),
            ('s12', True, 0),
        ], fpdtype='float64', psteps=ssteps, compressor='gz')
    return cse

def cvg2d_ve(casename, mtrlname, **kw):
    return cvg_base(casename=casename, **kw)

################################################################################
# The arrangement for 2D convergence test.
################################################################################
@vewave.VewaveCase.register_arrangement
def cvg2dv(casename, **kw):
    return cvg2d_ve(casename=casename, time_increment=0.000000012,
                    steps_run=10, ssteps=1, psteps=1,
                    mtrlname='SoftTissue', meshname='vewave2d_s.msh',
                    rho=1.06e3, vp=1578.0, sig0=10.0, freq=2e6)

@vewave.VewaveCase.register_arrangement
def cvg2d_p2(casename, **kw):
    return cvg2d_ve(casename=casename, time_increment=1.5e-9,
                    steps_run=10, ssteps=1, psteps=1,
                    mtrlname='SoftTissue', meshname='vewave2d_xxl_p2.dom',
                    rho=1.06e3, vp=1578.0, sig0=10.0, freq=2e6)

@vewave.VewaveCase.register_arrangement
def cvg2d_p3(casename, **kw):
    return cvg2d_ve(casename=casename, time_increment=1.5e-9,
                    steps_run=10, ssteps=1, psteps=1,
                    mtrlname='SoftTissue', meshname='vewave2d_xxl_p3.dom',
                    rho=1.06e3, vp=1578.0, sig0=10.0, freq=2e6)

@vewave.VewaveCase.register_arrangement
def cvg2d_p4(casename, **kw):
    return cvg2d_ve(casename=casename, time_increment=1.5e-9,
                    steps_run=10, ssteps=1, psteps=1,
                    mtrlname='SoftTissue', meshname='vewave2d_xxl_p4.dom',
                    rho=1.06e3, vp=1578.0, sig0=10.0, freq=2e6)

@vewave.VewaveCase.register_arrangement
def cvg2d_p5(casename, **kw):
    return cvg2d_ve(casename=casename, time_increment=1.5e-9,
                    steps_run=10, ssteps=1, psteps=1,
                    mtrlname='SoftTissue', meshname='vewave2d_xxl_p5.dom',
                    rho=1.06e3, vp=1578.0, sig0=10.0, freq=2e6)

################################################################################
# Invoke SOLVCON workflow.
################################################################################
if __name__ == '__main__':
    solvcon.go()

# vim: set ai et nu sw=4 ts=4 tw=79: