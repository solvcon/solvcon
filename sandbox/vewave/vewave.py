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
from solvcon import command
from solvcon.parcel import vewave


################################################################################
# Mesh generation and boundary condition processor.
################################################################################
def save_blk(blk, meshname):
    from solvcon.io import block
    bio = block.BlockIO(blk=blk)
    bio.save(stream=meshname)

def save_domain(blk, domainname, npart):
    from time import time
    from solvcon.domain import Collective
    from solvcon.io.domain import DomainIO
    from solvcon.helper import info
    info('Create domain ... ')
    timer = time()
    dom = Collective(blk)
    info('done. (%gs)\n' % (time()-timer))
    info('Partition graph into %d parts ... ' % npart)
    timer = time()
    dom.partition(npart)
    info('done. (%gs)\n' % (time()-timer))
    info('Split step 1/5: distribute into sub-domains ... ')
    timer = time()
    dom.distribute()
    info('done. (%gs)\n' % (time()-timer))
    info('Split step 2/5: compute neighbor block ... ')
    timer = time()
    clmap = dom.compute_neighbor_block()
    info('done. (%gs)\n' % (time()-timer))
    info('Split step 3/5: reindex entities ... ')
    timer = time()
    dom.reindex(clmap)
    info('done. (%gs)\n' % (time()-timer))
    info('Split step 4/5: build interface ... ')
    timer = time()
    dom.build_interface()
    info('done. (%gs)\n' % (time()-timer))
    info('Split step 5/5: supplement ... ')
    timer = time()
    dom.supplement()
    info('done. (%gs)\n' % (time()-timer))
    dio = DomainIO(dom=dom, compressor='gz')
    if not os.path.exists(domainname):
        os.makedirs(domainname)
    info('Save to directory %s/ ... ' % domainname)
    timer = time()
    dio.save(dirname=domainname)
    info('done. (%gs)\n' % (time()-timer))

def mesher(casename, bcmap):
    # determine meshing template file name.
    tmplfn = '%s.%s.tmpl' % ('vewave2d', 'gmsh')
    # determine characteristic length of mesh.
    meshfiner = casename.split('_')[1]
    npart = int(casename.split('_')[2])
    try:
        itv = 0.0001 + float(meshfiner)/10000.0
    except ValueError:
        itv = 0.0001
    # load the meshing commands.
    cmds = open(tmplfn).read() % itv
    cmds = [cmd.strip() for cmd in cmds.strip().split('\n')]
    # make the original mesh object.
    mobj = helper.Gmsh(cmds)()
    # convert the mesh to block.
    blk = mobj.toblock(bcname_mapper=bcmap,
                       use_incenter=False)
    basedir = os.path.join(os.path.abspath(os.getcwd()), 'mesh')
    if npart == 0:
        meshname = os.path.join(basedir, 'vewave2d_%d.blk' % int(meshfiner))
        save_blk(blk, meshname)
    else:
        domainname = 'vewave2d_%d_p%d.dom' % (int(meshfiner),npart)
        domainname = os.path.join(basedir, domainname)
        save_domain(blk, domainname, npart)
    

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
    #from . import rootdir
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
    basedir = os.path.join(os.path.abspath(os.getcwd()), 'result', casename)
    mesher(casename, bcmap)
    meshfn = os.path.join(os.path.abspath(os.getcwd()), 'mesh', meshname)
    local_mesher = functools.partial(
        mesher, use_cubit=os.environ.get('USE_CUBIT', False))
    cse = vewave.VewaveCase(
        basedir=basedir, rootdir=conf.env.projdir, basefn=casename,
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
# Naming Convention
# cvg2d_A_B:  A is meshfiner, larger means finer. 
#             B is number of partition
# Mesh file name:
#   Serial:  vewave2d_A.blk
#   Parallel:  vewave2d_A_pB.dom
################################################################################
@vewave.VewaveCase.register_arrangement
def cvg2d_0_0(casename, **kw):
    return cvg2d_ve(casename=casename, time_increment=0.000000012,
                    steps_run=10, ssteps=1, psteps=1,
                    mtrlname='SoftTissue', meshname='vewave2d_0.blk',
                    rho=1.06e3, vp=1578.0, sig0=10.0, freq=2e6)

@vewave.VewaveCase.register_arrangement
def cvg2d_0_2(casename, **kw):
    return cvg2d_ve(casename=casename, time_increment=1.5e-9,
                    steps_run=10, ssteps=1, psteps=1,
                    mtrlname='SoftTissue', meshname='vewave2d_0_p2.dom',
                    rho=1.06e3, vp=1578.0, sig0=10.0, freq=2e6)

@vewave.VewaveCase.register_arrangement
def cvg2d_0_3(casename, **kw):
    return cvg2d_ve(casename=casename, time_increment=1.5e-9,
                    steps_run=10, ssteps=1, psteps=1,
                    mtrlname='SoftTissue', meshname='vewave2d_0_p3.dom',
                    rho=1.06e3, vp=1578.0, sig0=10.0, freq=2e6)

@vewave.VewaveCase.register_arrangement
def cvg2d_0_4(casename, **kw):
    return cvg2d_ve(casename=casename, time_increment=1.5e-9,
                    steps_run=10, ssteps=1, psteps=1,
                    mtrlname='SoftTissue', meshname='vewave2d_0_p4.dom',
                    rho=1.06e3, vp=1578.0, sig0=10.0, freq=2e6)

@vewave.VewaveCase.register_arrangement
def cvg2d_0_5(casename, **kw):
    return cvg2d_ve(casename=casename, time_increment=1.5e-9,
                    steps_run=10, ssteps=1, psteps=1,
                    mtrlname='SoftTissue', meshname='vewave2d_0_p5.dom',
                    rho=1.06e3, vp=1578.0, sig0=10.0, freq=2e6)

################################################################################
# Invoke SOLVCON workflow.
################################################################################
if __name__ == '__main__':
    solvcon.go()

# vim: set ai et nu sw=4 ts=4 tw=79:
