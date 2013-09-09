# -*- coding: UTF-8 -*-
#
# Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
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
Legacy hooks for simulation :py:class:`solvcon.case_legacy.Case`.  Two
categories of hooks are defined here: (i) base hooks for subclassing and (ii)
generic hooks which can be readily installed.
"""

import os
import time
import math

import numpy as np

from . import rpc
from . import domain
from . import anchor
from .io import vtk as scvtk
from .io import vtkxml

class Hook(object):
    """
    Organizer class for hooking subroutines for BaseCase.

    @ivar cse: Case object.
    @itype cse: BaseCase
    @ivar info: information output function.
    @itype info: callable
    @ivar psteps: the interval number of steps between printing.
    @itype psteps: int
    @ivar kws: excessive keywords.
    @itype kws: dict
    """
    def __init__(self, cse, **kw):
        """
        @param cse: Case object.
        @type cse: BaseCase
        """
        from . import case # avoid cyclic importation.
        assert isinstance(cse, (case.BaseCase, case.MeshCase))
        self.cse = cse
        self.info = cse.info
        self.psteps = kw.pop('psteps', None)
        self.ankcls = kw.pop('ankcls', None)
        # save excessive keywords.
        self.kws = dict(kw)
        super(Hook, self).__init__()

    def _makedir(self, dirname, verbose=False):
        """
        Make new directory if it does not exist in prior.

        @param dirname: name of directory to be created.
        @type dirname: str
        @keyword verbose: flag if print out creation message.
        @type verbose: bool
        """
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            if verbose:
                self.info('Created %s' % dirname)

    def _depend(self, deplst, verbose=False, stop_on_false=True):
        """
        Check for dependency to another hook.

        @param deplst: list of depended hook classes.
        @type deplst: list
        @keyword verbose: flag print message.
        @type verbose: bool
        @keyword stop_on_false: flag stop on false.
        @type stop_on_false: bool
        @return: dependency met or not.
        @rtype: bool
        """
        hooks = self.cse.runhooks
        info = self.info
        # check.
        metlst = []
        msglst = []
        for ahook in deplst:
            metlst.append(False)
            for obj in hooks:
                if isinstance(obj, ahook):
                    metlst[-1] = True
                    break
            if not metlst[-1]:
                msglst.append("%s should be enabled for %s." % (
                    ahook.__name__, self.__class__.__name__))
        if verbose and msglst:
            info('\n'.join(msglst)+'\n')
        if stop_on_false and msglst:
            raise RuntimeError, '\n'.join(msglst)
        # return.
        for met in metlst:
            if not met:
                return False
        return True

    @staticmethod
    def _deliver_anchor(target, ankcls, ankkw):
        """
        Provide the information to instantiate anchor object for a solver.  The
        target object can be a real solver object or a shadow associated to a
        remote worker object with attached muscle of solver object.

        @param target: the solver or shadow object.
        @type target: solvcon.solver.Solver or solvcon.rpc.Shadow
        @param ankcls: type of the anchor to instantiate.
        @type ankcls: type
        @param ankkw: keywords to instantiate anchor object.
        @type ankkw: dict
        @return: nothing
        """
        if isinstance(target, rpc.Shadow):
            target.drop_anchor(ankcls, ankkw)
        else:
            target.runanchors.append(ankcls, **ankkw)
    def drop_anchor(self, svr):
        """
        Drop the anchor(s) to the solver object.

        @param svr: the solver object on which the anchor(s) is dropped.
        @type svr: solvon.solver.BaseSolver
        @return: nothing
        """
        if self.ankcls:
            self._deliver_anchor(svr, self.ankcls, self.kws)

    def preloop(self):
        """
        Things to do before the time-marching loop.
        """
        pass

    def premarch(self):
        """
        Things to do before the time march for a specific time step.
        """
        pass

    def postmarch(self):
        """
        Things to do after the time march for a specific time step.
        """
        pass

    def postloop(self):
        """
        Things to do after the time-marching loop.
        """
        pass

################################################################################
# Fundamental hooks.
################################################################################
class ProgressHook(Hook):
    """
    Print simulation progess.

    @ivar linewidth: the maximal width for progress symbol.  50 is upper limit.
    @itype linewidth: int
    """
    def __init__(self, cse, **kw):
        self.linewidth = kw.pop('linewidth', 50)
        super(ProgressHook, self).__init__(cse, **kw)
    def preloop(self):
        istep = self.cse.execution.step_current
        nsteps = self.cse.execution.steps_run
        info = self.info
        info("Steps %d/%d\n" % (istep, nsteps))
    def postmarch(self):
        from datetime import timedelta
        istep = self.cse.execution.step_current
        nsteps = self.cse.execution.steps_run
        tstart = self.cse.log.time['loop_march'][0]
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
            info("\nStep %d/%d, time elapsed: %s remaining: %s\n" % (
                istep, nsteps, 
                str(timedelta(seconds=int(tcurr-tstart))),
                str(timedelta(seconds=int(tleft))),
            ))
        elif istep == nsteps:
            info("\nStep %d/%d done\n" % (istep, nsteps))

################################################################################
# Hooks for BlockCase.
################################################################################
class BlockHook(Hook):
    """
    Base type for hooks needing a BlockCase.
    """
    def __init__(self, cse, **kw):
        from . import case # avoid cyclic importation.
        assert isinstance(cse, (case.BlockCase, case.MeshCase))
        super(BlockHook, self).__init__(cse, **kw)

    @property
    def blk(self):
        return self.cse.solver.domainobj.blk

    def _collect_interior(self, key, tovar=False, inder=False,
        consider_ghost=True):
        """
        @param key: the name of the array to collect in a solver object.
        @type key: str
        @keyword tovar: flag to store collect data to case var dict.
        @type tovar: bool
        @keyword inder: the array is for derived data.
        @type inder: bool
        @keyword consider_ghost: treat the array with the consideration of
            ghost cells.  Default is True.
        @type consider_ghost: bool
        @return: the interior array hold by the solver.
        @rtype: numpy.ndarray
        """
        cse = self.cse
        ncell = self.blk.ncell
        ngstcell = self.blk.ngstcell
        if cse.is_parallel:
            dom = self.cse.solver.domainobj
            # collect arrays from solvers.
            dealer = self.cse.solver.dealer
            arrs = list()
            for iblk in range(dom.nblk):
                dealer[iblk].cmd.pull(key, inder=inder, with_worker=True)
                arr = dealer[iblk].recv()
                arrs.append(arr)
            # create global array.
            shape = [it for it in arrs[0].shape]
            shape[0] = ncell
            arrg = np.empty(shape, dtype=arrs[0].dtype)
            # set global array.
            clmaps = dom.mappers[2]
            for iblk in range(dom.nblk):
                slctg = (clmaps[:,1] == iblk)
                slctl = clmaps[slctg,0]
                if consider_ghost:
                    slctl += dom.shapes[iblk,6]
                arrg[slctg] = arrs[iblk][slctl]
        else:
            if consider_ghost:
                start = ngstcell
            else:
                start = 0
            if inder:
                arrg = cse.solver.solverobj.der[key][start:].copy()
            else:
                arrg = getattr(cse.solver.solverobj, key)[start:].copy()
        if tovar:
            self.cse.execution.var[key] = arrg
        return arrg

    def _spread_interior(self, arrg, key, consider_ghost=True):
        """
        @param arrg: the global array to be spreaded.
        @type arrg: numpy.ndarray
        @param key: the name of the array to collect in a solver object.
        @type key: str
        @keyword consider_ghost: treat the arrays with the consideration of
            ghost cells.  Default is True.
        @type consider_ghost: bool
        @return: the interior array hold by the solver.
        @rtype: numpy.ndarray
        """
        cse = self.cse
        ncell = self.blk.ncell
        ngstcell = self.blk.ngstcell
        if cse.is_parallel:
            dom = self.cse.solver.domainobj
            dealer = self.cse.solver.dealer
            clmaps = dom.mappers[2]
            for iblk in range(len(dom)):
                blk = dom[iblk]
                # create subarray.
                shape = [it for it in arrg.shape]
                if consider_ghost:
                    shape[0] = blk.ngstcell+blk.ncell
                else:
                    shape[0] = blk.ncell
                arr = np.empty(shape, dtype=arrg.dtype)
                # calculate selectors.
                slctg = (clmaps[:,1] == iblk)
                slctl = clmaps[slctg,0]
                if consider_ghost:
                    slctl += blk.ngstcell
                # push data to remote solver.
                arr[slctl] = arrg[slctg]
                dealer[iblk].cmd.push(arr, key, start=blk.ngstcell)
        else:
            if consider_ghost:
                start = ngstcell
            else:
                start = 0
            getattr(cse.solver.solverobj, key)[start:] = arrg[:]

class BlockInfoHook(BlockHook):
    def __init__(self, cse, **kw):
        """
        If keyword psteps is None, postmarch method will not output performance
        information.
        """
        self.show_bclist = kw.pop('show_bclist', False)
        self.perffn = kw.pop('perffn', None)
        super(BlockInfoHook, self).__init__(cse, **kw)

    def preloop(self):
        blk = self.blk
        self.info("Block information:\n  %s\n" % str(blk))
        if self.show_bclist:
            for bc in blk.bclist:
                self.info("  %s\n" % bc)

    def _show_performance(self):
        """
        Show and store performance information.
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


class CollectHook(BlockHook):
    def __init__(self, cse, **kw):
        self.varlist = kw.pop('varlist')
        self.error_on_nan = kw.pop('error_on_nan', False)
        super(CollectHook, self).__init__(cse, **kw)
    def postmarch(self):
        psteps = self.psteps
        istep = self.cse.execution.step_current
        if istep%psteps != 0 and istep != self.cse.execution.steps_run:
            return
        vstep = self.cse.execution.varstep
        var = self.cse.execution.var
        # collect variables from solver object.
        if istep != vstep:
            for key, kw in self.varlist:
                arr = var[key] = self._collect_interior(key, **kw)
                nans = np.isnan(arr)
                msg = 'nan occurs in %s at step %d' % (key, istep)
                if nans.any():
                    if self.error_on_nan:
                        raise ValueError(msg)
                    else:
                        self.info(msg+'\n')
        self.cse.execution.varstep = istep
    preloop = postmarch

################################################################################
# Markers.
################################################################################
class SplitMarker(BlockHook):
    """
    Mark each cell with the domain index.
    """
    def preloop(self):
        cse = self.cse
        dom = cse.solver.domainobj
        if isinstance(dom, domain.Collective):
            cse.execution.var['domain'] = dom.part
        else:
            cse.execution.var['domain'] = np.zeros(dom.blk.ncell, dtype='int32')


class GroupMarker(BlockHook):
    """
    Mark each cell with the group index.
    """
    def preloop(self):
        var = self.cse.execution.var
        var['clgrp'] = self.blk.clgrp.copy()

################################################################################
# Vtk legacy writers.
################################################################################
class VtkSave(BlockHook):
    """
    Base type for writer for cse with a block.

    @ivar binary: True for BINARY format; False for ASCII.
    @itype binary: bool
    @ivar cache_grid: True to cache grid; False to forget grid every time.
    @itype cache_grid: bool
    """
    def __init__(self, cse, **kw):
        self.binary = kw.pop('binary', False)
        self.cache_grid = kw.pop('cache_grid', True)
        super(VtkSave, self).__init__(cse, **kw)


class SplitSave(VtkSave):
    """
    Save the splitted geometry.
    """

    def preloop(self):
        cse = self.cse
        if cse.is_parallel == 0:
            return  # do nothing if not in parallel.
        basefn = cse.io.basefn
        dom = cse.solver.domainobj
        nblk = len(dom)
        # build filename templates.
        vtkfn = basefn + '_decomp'
        vtksfn_tmpl = basefn + '_decomp' + '_%%0%dd'%int(math.ceil(math.log10(nblk))+1)
        if self.binary:
            vtkfn += ".bin.vtk"
            vtksfn_tmpl += ".bin.vtk"
        else:
            vtkfn += ".vtk"
            vtksfn_tmpl += ".vtk"
        # write.
        ## lumped.
        self.info("Save domain decomposition for visualization (%d parts).\n" \
            % nblk)
        scvtk.VtkLegacyUstGridWriter(dom.blk,
            binary=self.binary, cache_grid=self.cache_grid).write(vtkfn)
        ## splitted.
        iblk = 0
        for blk in dom:
            writer = scvtk.VtkLegacyUstGridWriter(blk,
                binary=self.binary, cache_grid=self.cache_grid).write(
                vtksfn_tmpl%iblk)
            iblk += 1


class MarchSave(VtkSave):
    """
    Save the geometry and variables in a case when time marching.

    @ivar vtkfn_tmpl: template for output file name(s).
    @itype vtkfn_tmpl: str
    """
    def __init__(self, cse, **kw):
        super(MarchSave, self).__init__(cse, **kw)
        nsteps = cse.execution.steps_run
        basefn = cse.io.basefn
        vtkfn_tmpl = basefn + "_%%0%dd"%int(math.ceil(math.log10(nsteps))+1)
        if self.binary:
            vtkfn_tmpl += ".bin.vtk"
        else:
            vtkfn_tmpl += ".vtk"
        self.vtkfn_tmpl = os.path.join(cse.io.basedir,
            kw.pop('vtkfn_tmpl', vtkfn_tmpl))

    @property
    def data(self):
        """
        Get dictionaries for scalar and vector data from case.

        @return: dictionaries for scalar and vector.
        @rtype: tuple
        """
        cse = self.cse
        exe = cse.execution
        var = exe.var
        # names of solution arrays.
        sarrnames = []
        if 'solverobj' in cse.solver:
            sarrnames = cse.solver.solverobj._solution_array_
        # create dictionaries for scalars and vectors.
        sarrs = dict()
        varrs = dict()
        for key in var:
            if key in sarrnames: # skip pure solution.
                continue
            arr = var[key]
            if len(arr.shape) == 1:
                sarrs[key] = arr
            elif len(arr.shape) == 2:
                varrs[key] = arr
            else:
                raise IndexError, \
                  'the dimensions of case[\'%s\'] is %d > 2' % (
                    key, len(arr.shape))
        # put soln into scalars.
        soln = var['soln']
        for i in range(soln.shape[1]):
            sarrs['soln[%d]'%i] = soln[:,i]
        # return
        return sarrs, varrs

    def _write(self, istep):
        self.writer.scalars, self.writer.vectors = self.data
        self.writer.write(self.vtkfn_tmpl % istep)

    def preloop(self):
        psteps = self.psteps
        cse = self.cse
        blk = self.blk
        # initialize writer.
        self.writer = scvtk.VtkLegacyUstGridWriter(blk,
            binary=self.binary, cache_grid=self.cache_grid)
        # write initially.
        self._write(0)

    def postmarch(self):
        psteps = self.psteps
        exe = self.cse.execution
        istep = exe.step_current
        vstep = exe.varstep
        if istep%psteps == 0 or istep == self.cse.execution.steps_run:
            assert istep == vstep   # data must be fresh.
            self._write(istep)

################################################################################
# Vtk XML parallel writers.
################################################################################
class PMarchSave(BlockHook):
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

        self.pvdf = os.path.join(vdir, cse.io.basefn+".pvd")
        vtkfn_tmpl = basefn + "_%%0%dd"%int(math.ceil(math.log10(nsteps))+1) + '.pvtu'
        self.vtkfn_tmpl = os.path.join(vdir, kw.pop('vtkfn_tmpl', vtkfn_tmpl))
        # craft ext name template.
        npart = cse.execution.npart
        self.pextmpl = '.p%%0%dd'%int(math.ceil(math.log10(npart))+1) if npart else ''
        self.pextmpl += '.vtu'

    def drop_anchor(self, svr):
        basefn = os.path.splitext(self.vtkfn_tmpl)[0]
        anames = dict([(ent[0], ent[1]) for ent in self.anames])
        ankkw = dict(anames=anames, compressor=self.compressor,
            fpdtype=self.fpdtype, psteps=self.psteps,
            vtkfn_tmpl=basefn+self.pextmpl)
        self._deliver_anchor(svr, anchor.MarchSaveAnchor, ankkw)

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

    def _write_pvd_head(self):
        outf = open(self.pvdf, 'w')
        outf.write('<?xml version="1.0"?>\n')
        outf.write('<VTKFile type="Collection" version="0.1" \
             byte_order="LittleEndian" compressor="vtkZLibDataCompressor">\n')
        outf.write('  <Collection>\n')
        outf.write('  </Collection>\n')
        outf.write('</VTKFile>')
        outf.close()

    def _write_pvd_main(self, istep):
        from numpy import ceil, log10
        nsteps = self.cse.execution.steps_run

        if self.cse.is_parallel:
            sname_tmpl = os.path.splitext(self.vtkfn_tmpl)[0]+'.pvtu'
        else:
            sname_tmpl = os.path.splitext(self.vtkfn_tmpl)[0]+'.vtu'

        sname = sname_tmpl %(istep)
        s = '    <DataSet timestep="%f" group="" part="" file="%s"/>\n' \
                    % (self.cse.execution.time, sname)
        aFile = self.pvdf
        with open(aFile) as f:
            for i, l in enumerate(f):
                pass
        nline = i +1
        
        os.rename(aFile, aFile+"~")
        destination = open(aFile, "w")
        source = open(aFile+"~", "r")
        i = 0;
        for line in source:
            i += 1
            destination.write(line)
            if i == nline-2:
                destination.write(s)

        destination.close()
        source.close()
        
    def preloop(self):
        self._write(0)
        self._write_pvd_head()
        self._write_pvd_main(0)
        
    def postmarch(self):
        psteps = self.psteps
        istep = self.cse.execution.step_current
        if istep%psteps == 0:
            self._write(istep)
            self._write_pvd_main(istep)

    def postloop(self):
        psteps = self.psteps
        istep = self.cse.execution.step_current
        if istep%psteps != 0:
            self._write(istep)
            self._write_pvd_main(istep)

################################################################################
# Hooks for in situ visualization.
################################################################################
class PVtkHook(BlockHook):
    """
    Anchor dropper and wrapping PVTP file writer.  Note, fpdtype should be set
    to single precision or parallel VTP file could be in wrong format.

    @ivar name: name of this VTK operation set; there can be multiple operation
        sets attaching on a Case object; default is None.
    @itype name: str
    @ivar anames: the arrays in der of solvers to be saved.  Format is (name,
        inder, ndim), (name, inder, ndim) ...  For ndim > 0 the
        array is a spatial vector, for ndim == 0 a simple scalar, and ndim < 0
        a list of scalar.
    @itype anames: list
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
        self.name = kw.pop('name', None)
        self.anames = kw.pop('anames', list())
        self.fpdtype = kw.pop('fpdtype', 'float32')
        self.altdir = kw.pop('altdir', '')
        self.altsym = kw.pop('altsym', '')
        self.ankkw = kw.pop('ankkw', dict())
        super(PVtkHook, self).__init__(cse, **kw)
        # override vtkfn_tmpl.
        nsteps = cse.execution.steps_run
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
        # determine VTK file name template.
        vtkfn_tmpl = cse.io.basefn
        if self.name is not None:
            vtkfn_tmpl += '_%s' % self.name
        vtkfn_tmpl += "_%%0%dd"%int(math.ceil(math.log10(nsteps))+1) + '.pvtp'
        self.vtkfn_tmpl = os.path.join(vdir, kw.pop('vtkfn_tmpl', vtkfn_tmpl))
        # craft ext name template.
        npart = cse.execution.npart
        self.pextmpl = '.p%%0%dd'%int(math.ceil(math.log10(npart))+1) if npart else ''
        self.pextmpl += '.vtp'

    def drop_anchor(self, svr):
        basefn = os.path.splitext(self.vtkfn_tmpl)[0]
        ankkw = self.ankkw.copy()
        ankkw.update(dict(anames=self.anames, fpdtype=self.fpdtype,
            psteps=self.psteps, vtkfn_tmpl=basefn+self.pextmpl))
        self._deliver_anchor(svr, self.ankcls, ankkw)

    def _write(self, istep):
        if not self.cse.execution.npart:
            return
        # collect data.
        arrs = list()
        for key, inder, ndim in self.anames:
            if ndim > 0:
                arrs.append((key, self.fpdtype, True))
            elif ndim < 0:
                for it in range(abs(ndim)):
                    arrs.append(('%s[%d]' % (key, it), self.fpdtype, False))
            else:
                arrs.append((key, self.fpdtype, False))
        # write.
        wtr = vtkxml.PVtkXmlPolyDataWriter(self.blk, fpdtype=self.fpdtype, 
            arrs=arrs, npiece=self.cse.execution.npart, pextmpl=self.pextmpl)
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
