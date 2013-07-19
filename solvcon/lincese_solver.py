# -*- coding: UTF-8 -*-
#
# Copyright (C) 2012-2013 Yung-Yu Chen <yyc@solvcon.net>.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
A two-/three-dimensional, second order CESE solver for generic linear PDEs. It
uses :py:mod:`solvcon.lincese_algorithm`.
"""


import os
import math
import cPickle as pickle

import numpy as np

from . import solver
from . import case
from . import boundcond
from . import anchor
from . import hook
from . import domain
from . import lincese_algorithm
from .io import vtkxml


class LinceseSolver(solver.MeshSolver):
    """
    .. inheritance-diagram:: LinceseSolver
    """

    _interface_init_ = ['cecnd', 'cevol', 'sfmrc']
    _solution_array_ = ['solt', 'sol', 'soln', 'dsol', 'dsoln']

    def __init__(self, blk, **kw):
        """
        A linear solver needs a :py:class:`Block <solvcon.block.Block>` having
        at least one group:

        >>> from .testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> blk.clgrp.fill(0)
        >>> blk.grpnames.append('blank')

        A linear solver can't be instantiated directly:

        >>> svr = LinceseSolver(blk, neq=1)
        Traceback (most recent call last):
        ...
        TypeError: data type not understood

        To instantiate the linear solver, at least :py:attr:`gdlen` needs to be
        implemented:

        >>> class SubSolver(LinceseSolver):
        ...     @property
        ...     def gdlen(self):
        ...         return 1
        >>> svr = SubSolver(blk, neq=1)
        """
        # meta data.
        self.neq = kw.pop('neq')
        super(LinceseSolver, self).__init__(blk, **kw)
        self.substep_run = 2
        ndim = blk.ndim
        ncell = blk.ncell
        ngstcell = blk.ngstcell
        fpdtype = 'float64'
        # scheme parameters.
        self.alpha = int(kw.pop('alpha', 0))
        self.sigma0 = int(kw.pop('sigma0', 3.0))
        self.taylor = float(kw.pop('taylor', 1.0))  # dirty hack.
        self.cnbfac = float(kw.pop('cnbfac', 1.0))  # dirty hack.
        self.sftfac = float(kw.pop('sftfac', 1.0))  # dirty hack.
        self.taumin = float(kw.pop('taumin', 0.0))
        self.tauscale = float(kw.pop('tauscale', 1.0))
        # dual mesh.
        self.cecnd = np.empty(
            (ngstcell+ncell, blk.CLMFC+1, ndim), dtype=fpdtype)
        self.cevol = np.empty(
            (ngstcell+ncell, blk.CLMFC+1), dtype=fpdtype)
        self.sfmrc = np.empty((ncell, blk.CLMFC, blk.FCMND, 2, ndim),
            dtype=fpdtype)
        # parameters.
        self.grpda = np.empty((self.ngroup, self.gdlen), dtype=fpdtype)
        nsca = kw.pop('nsca', 0)
        nvec = kw.pop('nvec', 0)
        self.amsca = np.empty((ngstcell+ncell, nsca), dtype=fpdtype)
        self.amvec = np.empty((ngstcell+ncell, nvec, ndim), dtype=fpdtype)
        # solutions.
        neq = self.neq
        self.sol = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.soln = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.solt = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.dsol = np.empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        self.dsoln = np.empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        self.stm = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.cfl = np.empty(ngstcell+ncell, dtype=fpdtype)
        self.ocfl = np.empty(ngstcell+ncell, dtype=fpdtype)

    @property
    def gdlen(self):
        return None

    def create_alg(self):
        """
        Create a :py:class:`.lincese_algorithm.LinceseAlgorithm` object.

        >>> # create a valid solver as the test fixture.
        >>> from .testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> blk.clgrp.fill(0)
        >>> blk.grpnames.append('blank')
        >>> class SubSolver(LinceseSolver):
        ...     @property
        ...     def gdlen(self):
        ...         return 1
        >>> svr = SubSolver(blk, neq=1)

        Create an associated algorithm object is straight-forward:

        >>> alg = svr.create_alg()
        """
        alg = lincese_algorithm.LinceseAlgorithm()
        alg.setup_mesh(self.blk)
        alg.setup_algorithm(self)
        return alg

    def init(self, **kw):
        self.create_alg().prepare_ce()
        super(LinceseSolver, self).init(**kw)
        self.create_alg().prepare_sf()

    def provide(self):
        # fill group data array.
        self._make_grpda()
        # pre-calculate CFL.
        self.create_alg().calc_cfl()
        self.ocfl[:] = self.cfl[:]
        # super method.
        super(LinceseSolver, self).provide()

    def apply_bc(self):
        super(LinceseSolver, self).apply_bc()
        self.call_non_interface_bc('soln')
        self.call_non_interface_bc('dsoln')

    def _make_grpda(self):
        raise NotImplementedError

    ###########################################################################
    # marching algorithm.
    ###########################################################################
    _MMNAMES = solver.MeshSolver.new_method_list()

    @_MMNAMES.register
    def update(self, worker=None):
        self.sol[:,:] = self.soln[:,:]
        self.dsol[:,:,:] = self.dsoln[:,:,:]

    @_MMNAMES.register
    def calcsolt(self, worker=None):
        self.create_alg().calc_solt()

    @_MMNAMES.register
    def calcsoln(self, worker=None):
        self.create_alg().calc_soln()

    @_MMNAMES.register
    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    @_MMNAMES.register
    def bcsoln(self, worker=None):
        self.call_non_interface_bc('soln')

    @_MMNAMES.register
    def calcdsoln(self, worker=None):
        self.create_alg().calc_dsoln()

    @_MMNAMES.register
    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)

    @_MMNAMES.register
    def bcdsoln(self, worker=None):
        self.call_non_interface_bc('dsoln')


class LinceseCase(case.MeshCase):
    """
    Basic case with linear CESE method.
    """
    defdict = {
        'execution.verified_norm': -1.0,
        'solver.solvertype': LinceseSolver,
        'solver.domaintype': domain.Domain,
        'solver.alpha': 0,
        'solver.sigma0': 3.0,
        'solver.taylor': 1.0,
        'solver.cnbfac': 1.0,
        'solver.sftfac': 1.0,
        'solver.taumin': None,
        'solver.tauscale': None,
    }
    def make_solver_keywords(self):
        kw = super(LinceseCase, self).make_solver_keywords()
        # time.
        kw['time'] = self.execution.time
        kw['time_increment'] = self.execution.time_increment
        # c-tau scheme parameters.
        kw['alpha'] = int(self.solver.alpha)
        for key in ('sigma0', 'taylor', 'cnbfac', 'sftfac',
                    'taumin', 'tauscale',):
            val = self.solver.get(key)
            if val != None: kw[key] = float(val)
        return kw


class LincesePeriodic(boundcond.periodic):
    """
    General periodic boundary condition for sequential runs.
    """
    def init(self, **kw):
        svr = self.svr
        blk = svr.blk
        ngstcell = blk.ngstcell
        ngstface = blk.ngstface
        facn = self.facn
        slctm = self.rclp[:,0] + ngstcell
        slctr = self.rclp[:,1] + ngstcell
        # move coordinates.
        shf = svr.cecnd[slctr,0,:] - blk.shfccnd[facn[:,2]+ngstface,:]
        svr.cecnd[slctm,0,:] = blk.shfccnd[facn[:,0]+ngstface,:] + shf
    def soln(self):
        svr = self.svr
        blk = svr.blk
        slctm = self.rclp[:,0] + blk.ngstcell
        slctr = self.rclp[:,1] + blk.ngstcell
        svr.soln[slctm,:] = svr.soln[slctr,:]
    def dsoln(self):
        svr = self.svr
        blk = svr.blk
        slctm = self.rclp[:,0] + blk.ngstcell
        slctr = self.rclp[:,1] + blk.ngstcell
        svr.dsoln[slctm,:,:] = svr.dsoln[slctr,:,:]


################################################################################
# CFL.
################################################################################

class CflAnchor(anchor.Anchor):
    """
    Counting CFL numbers.  Use svr.marchret to return results.  Implements
    postmarch() method.

    @ivar rsteps: steps to run.
    @itype rsteps: int
    """
    def __init__(self, svr, **kw):
        self.rsteps = kw.pop('rsteps')
        super(CflAnchor, self).__init__(svr, **kw)
    def postmarch(self):
        svr = self.svr
        istep = svr.step_global
        rsteps = self.rsteps
        if istep > 0 and istep%rsteps == 0:
            # download data.
            ocfl = svr.ocfl[svr.ngstcell:]
            cfl = svr.cfl[svr.ngstcell:]
            # determine extremum.
            mincfl = ocfl.min()
            maxcfl = ocfl.max()
            nadj = (cfl==1).sum()
            # store.
            lst = svr.marchret.setdefault('cfl', [0.0, 0.0, 0, 0])
            lst[0] = mincfl
            lst[1] = maxcfl
            lst[2] = nadj
            lst[3] += nadj

class CflHook(hook.Hook):
    """
    Makes sure CFL number is bounded and print averaged CFL number over time.
    Reports CFL information per time step and on finishing.  Implements (i)
    postmarch() and (ii) postloop() methods.

    @ivar name: name of the CFL tool.
    @itype name: str
    @ivar rsteps: steps to run.
    @itype rsteps: int
    @ivar cflmin: CFL number should be greater than or equal to the value.
    @itype cflmin: float
    @ivar cflmax: CFL number should be less than the value.
    @itype cflmax: float
    @ivar fullstop: flag to stop when CFL is out of bound.  Default True.
    @itype fullstop: bool
    @ivar aCFL: accumulated CFL.
    @itype aCFL: float
    @ivar mCFL: mean CFL.
    @itype mCFL: float
    @ivar hnCFL: hereditary minimal CFL.
    @itype hnCFL: float
    @ivar hxCFL: hereditary maximal CFL.
    @itype hxCFL: float
    @ivar aadj: number of adjusted CFL accumulated since last report.
    @itype aadj: int
    @ivar haadj: total number of adjusted CFL since simulation started.
    @itype haadj: int
    """
    def __init__(self, cse, **kw):
        self.name = kw.pop('name', 'cfl')
        self.cflmin = kw.pop('cflmin', 0.0)
        self.cflmax = kw.pop('cflmax', 1.0)
        self.fullstop = kw.pop('fullstop', True)
        self.aCFL = 0.0
        self.mCFL = 0.0
        self.hnCFL = 1.0
        self.hxCFL = 0.0
        self.aadj = 0
        self.haadj = 0
        rsteps = kw.pop('rsteps', None)
        super(CflHook, self).__init__(cse, **kw)
        self.rsteps = self.psteps if rsteps == None else rsteps
        self.ankkw = kw
    def drop_anchor(self, svr):
        ankkw = self.ankkw.copy()
        ankkw['name'] = self.name
        ankkw['rsteps'] = self.rsteps
        self._deliver_anchor(svr, CflAnchor, ankkw)
    def _notify(self, msg):
        from warnings import warn
        if self.fullstop:
            raise RuntimeError(msg)
        else:
            warn(msg)
    def postmarch(self):
        from numpy import isnan
        info = self.info
        istep = self.cse.execution.step_current
        mr = self.cse.execution.marchret
        isp = self.cse.is_parallel
        rsteps = self.rsteps
        psteps = self.psteps
        # collect CFL.
        if istep > 0 and istep%rsteps == 0:
            nCFL = max([m['cfl'][0] for m in mr]) if isp else mr['cfl'][0]
            xCFL = max([m['cfl'][1] for m in mr]) if isp else mr['cfl'][1]
            nadj = sum([m['cfl'][2] for m in mr]) if isp else mr['cfl'][2]
            aadj = sum([m['cfl'][3] for m in mr]) if isp else mr['cfl'][3]
            hnCFL = min([nCFL, self.hnCFL])
            self.hnCFL = hnCFL if not isnan(hnCFL) else self.hnCFL
            hxCFL = max([xCFL, self.hxCFL])
            self.hxCFL = hxCFL if not isnan(hxCFL) else self.hxCFL
            self.aCFL += xCFL*rsteps
            self.mCFL = self.aCFL/istep
            self.aadj += aadj
            self.haadj += aadj
            # check.
            if self.cflmin != None and nCFL < self.cflmin:
                self._notify("CFL = %g < %g after step: %d" % (
                    nCFL, self.cflmin, istep))
            if self.cflmax != None and xCFL >= self.cflmax:
                self._notify("CFL = %g >= %g after step: %d" % (
                    xCFL, self.cflmax, istep))
            # output information.
            if istep > 0 and istep%psteps == 0:
                info("CFL = %.2f/%.2f - %.2f/%.2f adjusted: %d/%d/%d\n" % (
                    nCFL, xCFL, self.hnCFL, self.hxCFL, nadj,
                    self.aadj, self.haadj))
                self.aadj = 0
    def postloop(self):
        self.info("Averaged maximum CFL = %g.\n" % self.mCFL)


###############################################################################
# Plane wave solution and initializer.
###############################################################################

class PlaneWaveSolution(object):
    def __init__(self, **kw):
        wvec = kw['wvec']
        ctr = kw['ctr']
        amp = kw['amp']
        assert len(wvec) == len(ctr)
        # calculate eigenvalues and eigenvectors.
        evl, evc = self._calc_eigen(**kw)
        # store data to self.
        self.amp = evc * (amp / np.sqrt((evc**2).sum()))
        self.ctr = ctr
        self.wvec = wvec
        self.afreq = evl * np.sqrt((wvec**2).sum())
        self.wsp = evl
    def _calc_eigen(self, *args, **kw):
        """
        Calculate eigenvalues and eigenvectors.

        @return: eigenvalues and eigenvectors.
        @rtype: tuple
        """
        raise NotImplementedError
    def __call__(self, svr, asol, adsol):
        svr.create_alg().calc_planewave(
            asol, adsol, self.amp, self.ctr, self.wvec, self.afreq)

class PlaneWaveAnchor(anchor.Anchor):
    def __init__(self, svr, **kw):
        self.planewaves = kw.pop('planewaves')
        super(PlaneWaveAnchor, self).__init__(svr, **kw)
    def _calculate(self, asol):
        for pw in self.planewaves:
            pw(self.svr, asol, self.adsol)
    def provide(self):
        ngstcell = self.svr.blk.ngstcell
        nacell = self.svr.blk.ncell + ngstcell
        # plane wave solution.
        asol = self.svr.der['analytical'] = np.empty(
            (nacell, self.svr.neq), dtype='float64')
        adsol = self.adsol = np.empty(
            (nacell, self.svr.neq, self.svr.blk.ndim),
            dtype='float64')
        asol.fill(0.0)
        self._calculate(asol)
        self.svr.soln[ngstcell:,:] = asol[ngstcell:,:]
        self.svr.dsoln[ngstcell:,:,:] = adsol[ngstcell:,:,:]
        # difference.
        diff = self.svr.der['difference'] = np.empty(
            (nacell, self.svr.neq), dtype='float64')
        diff[ngstcell:,:] = self.svr.soln[ngstcell:,:] - asol[ngstcell:,:]
    def postfull(self):
        ngstcell = self.svr.ngstcell
        # plane wave solution.
        asol = self.svr.der['analytical']
        asol.fill(0.0)
        self._calculate(asol)
        # difference.
        diff = self.svr.der['difference']
        diff[ngstcell:,:] = self.svr.soln[ngstcell:,:] - asol[ngstcell:,:]

class PlaneWaveHook(hook.BlockHook):
    def __init__(self, svr, **kw):
        self.planewaves = kw.pop('planewaves')
        self.norm = dict()
        super(PlaneWaveHook, self).__init__(svr, **kw)
    def drop_anchor(self, svr):
        svr.runanchors.append(
            PlaneWaveAnchor(svr, planewaves=self.planewaves)
        )
    def _calculate(self):
        neq = self.cse.execution.neq
        var = self.cse.execution.var
        asol = self._collect_interior(
            'analytical', inder=True, consider_ghost=True)
        diff = self._collect_interior(
            'difference', inder=True, consider_ghost=True)
        norm_Linf = np.empty(neq, dtype='float64')
        norm_L2 = np.empty(neq, dtype='float64')
        clvol = self.blk.clvol
        for it in range(neq):
            norm_Linf[it] = np.abs(diff[:,it]).max()
            norm_L2[it] = np.sqrt((diff[:,it]**2*clvol).sum())
        self.norm['Linf'] = norm_Linf
        self.norm['L2'] = norm_L2
    def preloop(self):
        self.postmarch()
        for ipw in range(len(self.planewaves)):
            pw = self.planewaves[ipw]
            self.info("planewave[%d]:\n" % ipw)
            self.info("  c = %g, omega = %g, T = %.15e\n" % (
                pw.wsp, pw.afreq, 2*np.pi/pw.afreq))
    def postmarch(self):
        psteps = self.psteps
        istep = self.cse.execution.step_current
        if istep%psteps == 0:
            self._calculate()
    def postloop(self):
        fname = '%s_norm.pickle' % self.cse.io.basefn
        fname = os.path.join(self.cse.io.basedir, fname)
        pickle.dump(self.norm, open(fname, 'wb'), -1)
        self.info('Linf norm in velocity:\n')
        self.info('  %e, %e, %e\n' % tuple(self.norm['Linf'][:3]))
        self.info('L2 norm in velocity:\n')
        self.info('  %e, %e, %e\n' % tuple(self.norm['L2'][:3]))

################################################################################
# Solution output.
################################################################################

class MarchSaveAnchor(anchor.Anchor):
    """
    Save solution data into VTK XML format for a solver.

    @ivar anames: the arrays in der of solvers to be saved.  True means in der.
    @itype anames: dict
    @ivar compressor: compressor for binary data.  Can only be 'gz' or ''.
    @itype compressor: str
    @ivar fpdtype: string for floating point data type (in numpy convention).
    @itype fpdtype: str
    @ivar psteps: the interval (in step) to save data.
    @itype psteps: int
    @ivar vtkfn_tmpl: the template string for the VTK file.
    @itype vtkfn_tmpl: str
    """

    def __init__(self, svr, **kw):
        self.anames = kw.pop('anames', dict())
        self.compressor = kw.pop('compressor')
        self.fpdtype = kw.pop('fpdtype')
        self.psteps = kw.pop('psteps')
        self.vtkfn_tmpl = kw.pop('vtkfn_tmpl')
        super(MarchSaveAnchor, self).__init__(svr, **kw)

    def _write(self, istep):
        ngstcell = self.svr.ngstcell
        sarrs = dict()
        varrs = dict()
        # collect data.
        for key in self.anames:
            # get the array.
            if self.anames[key]:
                arr = self.svr.der[key][ngstcell:]
            else:
                arr = getattr(self.svr, key)[ngstcell:]
            # put array in dict.
            if len(arr.shape) == 1:
                sarrs[key] = arr
            elif arr.shape[1] == self.svr.ndim:
                varrs[key] = arr
            else:
                for it in range(arr.shape[1]):
                    sarrs['%s[%d]' % (key, it)] = arr[:,it]
        # write.
        wtr = vtkxml.VtkXmlUstGridWriter(self.svr.blk, fpdtype=self.fpdtype,
            compressor=self.compressor, scalars=sarrs, vectors=varrs)
        svrn = self.svr.svrn
        wtr.write(self.vtkfn_tmpl % (istep if svrn is None else (istep, svrn)))

    def preloop(self):
        self._write(0)

    def postmarch(self):
        psteps = self.psteps
        istep = self.svr.step_global
        if istep%psteps == 0:
            self._write(istep)

    def postloop(self):
        psteps = self.psteps
        istep = self.svr.step_global
        if istep%psteps != 0:
            self._write(istep)

class PMarchSave(hook.BlockHook):
    """
    Save the geometry and variables in a case when time marching in parallel
    VTK XML format.

    @ivar anames: the arrays in der of solvers to be saved.  Format is (name,
        inder, ndim), (name, inder, ndim) ...  For ndim > 0 the
        array is a spatial vector, for ndim == 0 a simple scalar, and ndim < 0
        a list of scalar.
    @itype anames: list
    @ivar compressor: compressor for binary data.  Can only be 'gz' or ''.
    @itype compressor: str
    @ivar fpdtype: string for floating point data type (in numpy convention).
    @itype fpdtype: str
    @ivar altdir: the alternate directory to save the VTK files.
    @itype altdir: str
    @ivar altsym: the symbolic link in basedir pointing to the alternate
        directory to save the VTK files.
    @itype altsym: str
    @ivar pextmpl: template for the extension of split VTK file name.
    @itype pextmpl: str
    """

    def __init__(self, cse, **kw):
        self.anames = kw.pop('anames', list())
        self.compressor = kw.pop('compressor', 'gz')
        self.fpdtype = kw.pop('fpdtype', str(cse.execution.fpdtype))
        self.altdir = kw.pop('altdir', '')
        self.altsym = kw.pop('altsym', '')
        super(PMarchSave, self).__init__(cse, **kw)
        # override vtkfn_tmpl.
        nsteps = cse.execution.steps_run
        basefn = cse.io.basefn
        if self.altdir:
            vdir = self.altdir
            if self.altsym:
                altsym = os.path.join(cse.io.basedir, self.altsym)
                if not os.path.exists(altsym):
                    os.symlink(vdir, altsym)
        else:
            vdir = cse.io.basedir
        if not os.path.exists(vdir):
            os.makedirs(vdir)
        vtkfn_tmpl = basefn + "_%%0%dd"%int(math.ceil(math.log10(nsteps))+1)
        vtkfn_tmpl += '.pvtu'
        self.vtkfn_tmpl = os.path.join(vdir, kw.pop('vtkfn_tmpl', vtkfn_tmpl))
        # craft ext name template.
        npart = cse.execution.npart
        if npart:
            self.pextmpl = '.p%%0%dd'%int(math.ceil(math.log10(npart))+1)
        else:
            self.pextmpl = ''
        self.pextmpl += '.vtu'

    def drop_anchor(self, svr):
        basefn = os.path.splitext(self.vtkfn_tmpl)[0]
        anames = dict([(ent[0], ent[1]) for ent in self.anames])
        ankkw = dict(anames=anames, compressor=self.compressor,
            fpdtype=self.fpdtype, psteps=self.psteps,
            vtkfn_tmpl=basefn+self.pextmpl)
        self._deliver_anchor(svr, MarchSaveAnchor, ankkw)

    def _write(self, istep):
        if not self.cse.execution.npart:
            return
        # collect data.
        sarrs = dict()
        varrs = dict()
        for key, inder, ndim in self.anames:
            if ndim > 0:
                varrs[key] = self.fpdtype
            elif ndim < 0:
                for it in range(abs(ndim)):
                    sarrs['%s[%d]' % (key, it)] = self.fpdtype
            else:
                sarrs[key] = self.fpdtype
        # write.
        wtr = vtkxml.PVtkXmlUstGridWriter(self.blk, fpdtype=self.fpdtype,
            scalars=sarrs, vectors=varrs,
            npiece=self.cse.execution.npart, pextmpl=self.pextmpl)
        vtkfn = self.vtkfn_tmpl % istep
        self.info('Writing \n  %s\n... ' % vtkfn)
        wtr.write(vtkfn)
        self.info('done.\n')

    def preloop(self):
        self._write(0)

    def postmarch(self):
        psteps = self.psteps
        istep = self.cse.execution.step_current
        if istep%psteps == 0:
            self._write(istep)

    def postloop(self):
        psteps = self.psteps
        istep = self.cse.execution.step_current
        if istep%psteps != 0:
            self._write(istep)

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
