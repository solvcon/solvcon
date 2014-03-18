# -*- coding: UTF-8 -*-
#
# Copyright (c) 2012, Yung-Yu Chen <yyc@solvcon.net>
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
Input and output facilities.
"""


__all__ = [
    'AmscaAnchor', 'MeshInfoHook', 'ProgressHook', 'FillAnchor', 'CflAnchor',
    'CflHook', 'MarchSaveAnchor', 'PMarchSave',
]


import os
import time
import math
import warnings
import cPickle as pickle

import numpy as np

from solvcon import anchor
from solvcon import hook
from solvcon.io import vtkxml


class AmscaAnchor(anchor.MeshAnchor):
    """FIXME: this name is not distinguishing.  It seems I want to initialize
    the solver, so maybe I should be renamed to InitializingAnchor?
    """
    def __init__(self, svr, **kw):
        self.rho = float(kw.pop('rho', 1.06e3))
        self.vp = float(kw.pop('vp', 1578.0))
        self.sig0 = float(kw.pop('sig0', 10.0))
        self.freq = float(kw.pop('freq', 2e6))
        self.svr = svr
        super(AmscaAnchor, self).__init__(svr, **kw)

    def provide(self):
        self.svr.amsca[:,0] = self.rho
        self.svr.amsca[:,1] = self.vp
        self.svr.amsca[:,2] = self.sig0
        self.svr.amsca[:,3] = self.freq

class MeshInfoHook(hook.MeshHook):
    """Print mesh information.
    """

    def __init__(self, cse, show_bclist=False, perffn=None, **kw):
        """If keyword psteps is None, postmarch method will not output
        performance information.
        """
        #: Flag to show the list of boundary conditions.  Default is ``False``.
        self.show_bclist = show_bclist
        #: Performance file name.
        self.perffn = perffn
        super(MeshInfoHook, self).__init__(cse, **kw)

    def preloop(self):
        blk = self.blk
        self.info("Block information:\n  %s\n" % str(blk))
        if self.show_bclist:
            for bc in blk.bclist:
                self.info("  %s\n" % bc)

    def _show_performance(self):
        """Show and store performance information.
        """
        ncell = self.blk.ncell
        time = self.cse.log.time['solver_march']
        step_init = self.cse.execution.step_init
        step_current = self.cse.execution.step_current
        neq = self.cse.execution.neq
        npart = self.cse.execution.npart
        # determine filename.
        perffn = '%s_perf.txt' % self.cse.io.basefn
        perffn = self.perffn if self.perffn is not None else perffn
        perffn = os.path.join(self.cse.io.basedir, perffn)
        pf = open(perffn, 'w')
        # calculate and output performance.
        def out(msg):
            self.info(msg)
            pf.write(msg)
        perf = (step_current-step_init)*ncell / time * 1.e-6
        out('Performance of %s:\n' % self.cse.io.basefn)
        out('  %g seconds in marching solver.\n' % time)
        out('  %g seconds/step.\n' % (time/(step_current-step_init)))
        out('  %g microseconds/cell.\n' % (1./perf))
        out('  %g Mcells/seconds.\n' % perf)
        out('  %g Mvariables/seconds.\n' % (perf*neq))
        if isinstance(self.cse.execution.npart, int):
            out('  %g Mcells/seconds/computer.\n' % (perf/npart))
            out('  %g Mvariables/seconds/computer.\n' % (perf*neq/npart))
        pf.close()

    def postmarch(self):
        istep = self.cse.execution.step_current
        nsteps = self.cse.execution.steps_run
        psteps = self.psteps
        if istep > 0 and psteps and istep%psteps == 0 and istep != nsteps:
            self._show_performance()

    def postloop(self):
        self._show_performance()


class ProgressHook(hook.MeshHook):
    """Print simulation progess.
    """

    def __init__(self, cse, linewidth=50, **kw):
        #: The maximum width for progress mark.
        self.linewidth = linewidth
        super(ProgressHook, self).__init__(cse, **kw)

    def preloop(self):
        istep = self.cse.execution.step_current
        nsteps = self.cse.execution.steps_run
        info = self.info
        info("Steps %d/%d\n" % (istep, nsteps))

    def postmarch(self):
        istep = self.cse.execution.step_current
        nsteps = self.cse.execution.steps_run
        tstart = self.cse.log.time['run_march'][0]
        psteps = self.psteps
        linewidth = self.linewidth
        info = self.info
        # calculate estimated remaining time.
        tcurr = time.time()
        tleft = (tcurr-tstart) * ((float(nsteps)-float(istep))/float(istep))
        # output information.
        if istep%psteps == 0:
            info("#")
        if istep > 0 and istep%(psteps*linewidth) == 0:
            info("\nStep %d/%d, %.1fs elapsed, %.1fs left\n" % (
                istep, nsteps, tcurr-tstart, tleft,
            ))
        elif istep == nsteps:
            info("\nStep %d/%d done\n" % (istep, nsteps))


class FillAnchor(anchor.MeshAnchor):
    """Fill the specified arrays of a :py:class:`VewaveSolver
    <.solver.VewaveSolver>` with corresponding value.
    """

    def __init__(self, svr, mappers=None, **kw):
        assert None is not mappers
        #: A :py:class:`dict` maps the names of attributes of the
        #: :py:attr:`MeshAnchor.svr <solvcon.anchor.MeshAnchor.svr>` to the
        #: filling value.
        self.mappers = mappers if mappers else {}
        super(FillAnchor, self).__init__(svr, **kw)

    def provide(self):
        for key, value in self.mappers.iteritems():
            getattr(self.svr, key).fill(value)


################################################################################
# Begin CFL evaluation.
class CflAnchor(anchor.MeshAnchor):
    """Counting CFL numbers.  Use :py:attr:`MeshSolver.marchret
    <solvcon.solver.MeshSolver.marchret>` to return results.  Implements
    ``postmarch()`` method.
    """

    def __init__(self, svr, rsteps=None, **kw):
        """
        >>> from solvcon.testing import create_trivial_2d_blk
        >>> from solvcon.solver import MeshSolver
        >>> svr = MeshSolver(create_trivial_2d_blk())
        >>> ank = CflAnchor(svr)
        Traceback (most recent call last):
            ...
        TypeError: int() argument must be a string or a number, not 'NoneType'
        >>> ank = CflAnchor(svr, 1)
        >>> ank.rsteps
        1
        """
        #: Steps to run (:py:class:`int`).
        self.rsteps = int(rsteps)
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


class CflHook(hook.MeshHook):
    """
    Makes sure CFL number is bounded and print averaged CFL number over time.
    Reports CFL information per time step and on finishing.  Implements (i)
    postmarch() and (ii) postloop() methods.
    """

    def __init__(self, cse,
                 name='cfl', cflmin=0.0, cflmax=1.0,
                 fullstop=True, rsteps=None, **kw):
        #: Name of the CFL tool.
        self.name = name
        #: Miminum CFL value.
        self.cflmin = cflmin
        #: Maximum CFL value.
        self.cflmax = cflmax
        #: Flag to stop when CFL is out of bound.  Default is ``True``.
        self.fullstop = fullstop
        #: Accumulated CFL.
        self.aCFL = 0.0
        #: Mean CFL.
        self.mCFL = 0.0
        #: Hereditary minimum CFL.
        self.hnCFL = 1.0
        #: Hereditary maximum CFL.
        self.hxCFL = 0.0
        #: Number of adjusted CFL accumulated since last report.
        self.aadj = 0
        #: Total number of adjusted CFL since simulation started.
        self.haadj = 0
        rsteps = rsteps
        super(CflHook, self).__init__(cse, **kw)
        #: Steps to run.
        self.rsteps = rsteps if rsteps else self.psteps
        self.ankkw = kw

    def drop_anchor(self, svr):
        ankkw = self.ankkw.copy()
        ankkw['name'] = self.name
        ankkw['rsteps'] = self.rsteps
        self._deliver_anchor(svr, CflAnchor, ankkw)

    def _notify(self, msg):
        if self.fullstop:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)

    def postmarch(self):
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
            self.hnCFL = hnCFL if not np.isnan(hnCFL) else self.hnCFL
            hxCFL = max([xCFL, self.hxCFL])
            self.hxCFL = hxCFL if not np.isnan(hxCFL) else self.hxCFL
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
# End CFL evaluation.
################################################################################


################################################################################
# Begin solution output.
class MarchSaveAnchor(anchor.MeshAnchor):
    """Save solution data into VTK XML format for a solver.
    """

    def __init__(self, svr, anames=None, compressor=None, fpdtype=None,
                 psteps=None, vtkfn_tmpl=None, **kw): 
        assert None is not compressor
        assert None is not fpdtype
        assert None is not psteps
        assert None is not vtkfn_tmpl
        #: The arrays in :py:class:`VewaveSolver <.solver.VewaveSolver>` or
        #: :py:attr:`MeshSolver.der <solvcon.solver.MeshSolver.der>` to be
        #: saved.
        self.anames = anames if anames else dict()
        #: Compressor for binary data.  Can be either ``'gz'`` or ``''``.
        self.compressor = compressor
        #: String for floating point data type (NumPy convention).
        self.fpdtype = fpdtype
        #: The interval in step to save data.
        self.psteps = psteps
        #: The template string for the VTK file.
        self.vtkfn_tmpl = vtkfn_tmpl
        super(MarchSaveAnchor, self).__init__(svr, **kw)

    @property
    def alg(self):
        return self.svr.alg

    def _calc_physics(self):
        der = self.svr.der
        self.alg.calc_physics(der['s11'], der['s22'], der['s33'],
                              der['s23'], der['s13'], der['s12'])

    def provide(self):
        from numpy import empty
        svr = self.svr
        der = svr.der
        nelm = svr.ngstcell + svr.ncell
        der['s11'] = empty(nelm, dtype=self.fpdtype)
        der['s22'] = empty(nelm, dtype=self.fpdtype)
        der['s33'] = empty(nelm, dtype=self.fpdtype)
        der['s23'] = empty(nelm, dtype=self.fpdtype)
        der['s13'] = empty(nelm, dtype=self.fpdtype)
        der['s12'] = empty(nelm, dtype=self.fpdtype)

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
        self._calc_physics()
        self._write(0)

    def postmarch(self):
        psteps = self.psteps
        istep = self.svr.step_global
        if istep%psteps == 0:
            self._calc_physics()
            self._write(istep)

    def postloop(self):
        psteps = self.psteps
        istep = self.svr.step_global
        if istep%psteps != 0:
            self._calc_physics()
            self._write(istep)


class PMarchSave(hook.MeshHook):
    """Save the geometry and variables in a case when time marching in parallel
    VTK XML format.
    """

    def __init__(self, cse, anames=None, compressor='gz', fpdtype=None,
                 altdir='', altsym='', vtkfn_tmpl=None, **kw):
        #: The arrays in :py:class:`VewaveSolver <.solver.VewaveSolver>` or
        #: :py:attr:`MeshSolver.der <solvcon.solver.MeshSolver.der>` to be
        #: saved.  Format is (name, inder, ndim), (name, inder, ndim) ...  For
        #: ndim > 0 the array is a spatial vector, for ndim == 0 a simple
        #: scalar, and ndim < 0 a list of scalar.
        self.anames = anames if anames else list()
        #: Compressor for binary data.  Can be either ``'gz'`` or ``''``.
        self.compressor = compressor
        #: String for floating point data type (NumPy convention).
        self.fpdtype = fpdtype if fpdtype else str(cse.execution.fpdtype)
        #: The alternate directory to save the VTK files.
        self.altdir = altdir
        #: The symbolic link in basedir pointing to the alternate directory to
        #: save the VTK files.
        self.altsym = altsym
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
        if None is vtkfn_tmpl:
            vtkfn_tmpl = basefn + "_%%0%dd"%int(math.ceil(math.log10(nsteps))+1)
            vtkfn_tmpl += '.pvtu'
        #: The template string for the VTK file.
        self.vtkfn_tmpl = os.path.join(vdir, vtkfn_tmpl)
        # craft ext name template.
        npart = cse.execution.npart
        if npart:
            self.pextmpl = '.p%%0%dd'%int(math.ceil(math.log10(npart))+1)
        else:
            self.pextmpl = ''
        #: Template for the extension of split VTK file name.
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
# End solution output.
################################################################################

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
