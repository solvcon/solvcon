#!/usr/bin/env python2.6
# -*- coding: UTF-8 -*-
# Copyright (C) 2011 by Yung-Yu Chen.  All rights reserved.

"""
Rupture tests.
"""

from solvcon.kerpak import bulk
from solvcon.anchor import Anchor, VtkAnchor
################################################################################
# Base setting.
################################################################################
def lfos_base(casename=None, altdir='', meshname=None,
    psteps=None, vsteps=None, ssteps=None, estep=None,
    gamma=None, tempi=None, prei=None, Mi=None, aoa=0.0,
    nonslip=False, benchmark=False,
    **kw):
    import os
    from numpy import sqrt, log, exp
    from solvcon.conf import env
    from solvcon.boundcond import bctregy
    from solvcon.solver import ALMOST_ZERO
    from solvcon import hook, anchor
    from solvcon.kerpak import cese
    from . import rootdir
    # set flow properties (fp).
    p0 = 101325.0;                   # back ground pressure
    rho0 = 1.205;                    # back ground density
    bulkk = 1.42e5;                   # bulk modulus
    pref = 20e-6;                    # hearing reference pressure
    rhoi = 1.1855;                   # initial density
    pini = p0 + bulkk*log(rhoi/rho0); # initial pressure
    pi = exp(pini/bulkk);
    mu = 1.8464e-5;                  # viscosity mu
    d = 0.016;                       # diameter of cylinder
    eta = rho0 * exp(-p0/bulkk);      # eta constant for pressure base
    a = sqrt(bulkk/(eta*pi))
    v1 = 84.0515;
    fpi = {
        'rho': rhoi, 'v1':v1, 'v3': 0.0, 'v2':0.0,
        'bulk': bulkk, 'rho0': rho0, 'p0': p0, 'eta': eta, 'pref':pref,
        'pini': pini, 'd': d, 'mu': mu,
        'crd': 1,
    }
    fpri = fpi.copy()
    # set up BCs.
    bcmap = {
        'inlet': (bctregy.NSBNonreflInlet, fpi,),
        'farfield': (bctregy.NSBNonrefl, {},),
        'sphere': (bctregy.NSBNswall, {},),
        'outlet': (bctregy.NSBOutlet, fpi,),
    }
    # set up case.
    if env.command and env.command.opargs[0].basedir:
        basedir = os.path.abspath(env.command.opargs[0].basedir)
    else:
        basedir = os.path.abspath(os.getcwd())
    meshfn = os.path.join(rootdir, 'mesh.cvg', meshname)
    cse = bulk.NSBulkCase(basedir=basedir, rootdir=env.projdir,
        basefn=casename, meshfn=meshfn, bcmap=bcmap, **kw)
    # statistical anchors for solvers.
    for name in 'Runtime', 'March', 'Tpool':
        cse.runhooks.append(getattr(anchor, name+'StatAnchor'))
    # informative hooks.
    cse.runhooks.append(hook.BlockInfoHook, show_bclist=True)
    cse.runhooks.append(hook.ProgressHook, psteps=psteps,
        linewidth=vsteps/psteps)
    if not benchmark:
        cse.runhooks.append(cese.CflHook, fullstop=False, psteps=vsteps,
            cflmax=10.0, linewidth=vsteps/psteps)
        cse.runhooks.append(cese.ConvergeHook, psteps=vsteps)
    # initializer anchors.
    cse.runhooks.append(anchor.FillAnchor, keys=('soln',), value=ALMOST_ZERO)
    cse.runhooks.append(anchor.FillAnchor, keys=('dsoln',), value=0)
    cse.runhooks.append(bulk.UniformIAnchor, **fpi)
    # rupture.
    # analyzing/output anchors and hooks.
    if not benchmark:
        cse.runhooks.append(bulk.NSBulkOAnchor, rsteps=vsteps)
        cse.runhooks.append(hook.PMarchSave, anames=[
                ('soln', False, -5),
                ('rho', True, 0),
                ('p', True, 0),
                ('M', True, 0),
                ('v', True, 0.5),
                ('dB', True, 0),
                ('predif', True, 0),
                ('w', True, 0.5),
            ], psteps=ssteps,
            fpdtype='float64', compressor='gz', altdir=altdir, altsym='pvtu',
        )
    return cse

def lfos_flow(casename, **kw):
    kw.setdefault('aoa', 5.0)
    return lfos_base(casename=casename,
        rkillfn='', **kw)

################################################################################
# For results.
################################################################################

@bulk.NSBulkCase.register_arrangement
def foc_re89000(casename, **kw):
    return lfos_flow(casename=casename,
        meshname='lfos_D16mm_acoustic_400k.blk',
        time_increment=3.4e-7, nonslip=False,
        steps_run=500000, ssteps=1000, vsteps=1000, psteps=200, estep=5000,
        taylor=0.0, taumin=0.1, tauscale=6.0, viscosity = 1,
        **kw)
