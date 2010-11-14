# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2010 Yung-Yu Chen <yyc@solvcon.net>.
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
Hooks for simulation cases.  Two categories of hooks are defined here: (i) base
hooks for subclassing and (ii) generic hooks which can be readily installed.
"""

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
        from .case import BaseCase
        assert isinstance(cse, BaseCase)
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
        import os
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
        from .rpc import Shadow
        if isinstance(target, Shadow):
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

class HookList(list):
    """
    Hook container and invoker.

    @ivar cse: case object.
    @itype cse: solvcon.case.BaseCase
    """
    def __init__(self, cse, *args, **kw):
        self.cse = cse
        super(HookList, self).__init__(*args, **kw)
    def append(self, obj, **kw):
        """
        The object to be appended (the first and only argument) should be a 
        Hook object, but this method actually accept either a Hook type or an
        Anchor type.  The method will automatically create the necessary Hook
        object when detect acceptable type object passed as the first argument.

        All the keywords go to the creation of the Hook object if the first
        argument is a type.  If the first argument is an instantiated Hook
        object, the method accepts no keywords.

        @param obj: the hook object to be appended.
        @type obj: solvcon.hook.Hook
        """
        from .anchor import Anchor
        if isinstance(obj, type):
            if issubclass(obj, Anchor):
                kw['ankcls'] = obj
                obj = Hook
            obj = obj(self.cse, **kw)
        else:
            assert len(kw) == 0
        super(HookList, self).append(obj)
    def __call__(self, method):
        """
        Invoke the specified method for each hook object.

        @param method: name of the method to run.
        @type method: str
        """
        runhooks = self
        if method == 'postloop':
            runhooks = reversed(runhooks)
        for hook in runhooks:
            getattr(hook, method)()
    def drop_anchor(self, svr):
        for hok in self:
            hok.drop_anchor(svr)

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
        from time import time
        istep = self.cse.execution.step_current
        nsteps = self.cse.execution.steps_run
        tstart = self.cse.log.time['loop_march'][0]
        psteps = self.psteps
        linewidth = self.linewidth
        info = self.info
        # calculate estimated remaining time.
        tcurr = time()
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

################################################################################
# Hooks for BlockCase.
################################################################################
class BlockHook(Hook):
    """
    Base type for hooks needing a BlockCase.
    """
    def __init__(self, cse, **kw):
        from .case import BlockCase
        assert isinstance(cse, BlockCase)
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
        from numpy import empty
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
            arrg = empty(shape, dtype=arrs[0].dtype)
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
        from numpy import empty
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
                arr = empty(shape, dtype=arrg.dtype)
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
        self.show_bclist = kw.pop('show_bclist', False)
        self.perffn = kw.pop('perffn', None)
        super(BlockInfoHook, self).__init__(cse, **kw)
    def preloop(self):
        blk = self.blk
        self.info("Block information:\n  %s\n" % str(blk))
        if self.show_bclist:
            for bc in blk.bclist:
                self.info("  %s\n" % bc)
    def postloop(self):
        import os
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
        out('  %g microseconds/cell.\n' % (1./perf))
        out('  %g Mcells/seconds.\n' % perf)
        out('  %g Mvariables/seconds.\n' % (perf*neq))
        if isinstance(self.cse.execution.npart, int):
            out('  %g Mcells/seconds/computer.\n' % (perf/npart))
            out('  %g Mvariables/seconds/computer.\n' % (perf*neq/npart))
        pf.close()

class CollectHook(BlockHook):
    def __init__(self, cse, **kw):
        self.varlist = kw.pop('varlist')
        self.error_on_nan = kw.pop('error_on_nan', False)
        super(CollectHook, self).__init__(cse, **kw)
    def postmarch(self):
        from numpy import isnan
        psteps = self.psteps
        istep = self.cse.execution.step_current
        if istep%psteps != 0:
            return
        vstep = self.cse.execution.varstep
        var = self.cse.execution.var
        # collect variables from solver object.
        if istep != vstep:
            for key, kw in self.varlist:
                arr = var[key] = self._collect_interior(key, **kw)
                nans = isnan(arr)
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
        from numpy import zeros
        from .domain import Collective
        cse = self.cse
        dom = cse.solver.domainobj
        if isinstance(dom, Collective):
            cse.execution.var['domain'] = dom.part
        else:
            cse.execution.var['domain'] = zeros(dom.blk.ncell, dtype='int32')

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
        from math import log10, ceil
        from .io.vtk import VtkLegacyUstGridWriter
        cse = self.cse
        if cse.is_parallel == 0:
            return  # do nothing if not in parallel.
        basefn = cse.io.basefn
        dom = cse.solver.domainobj
        nblk = len(dom)
        # build filename templates.
        vtkfn = basefn + '_decomp'
        vtksfn_tmpl = basefn + '_decomp' + '_%%0%dd'%int(ceil(log10(nblk))+1)
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
        VtkLegacyUstGridWriter(dom.blk,
            binary=self.binary, cache_grid=self.cache_grid).write(vtkfn)
        ## splitted.
        iblk = 0
        for blk in dom:
            writer = VtkLegacyUstGridWriter(blk,
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
        import os
        from math import log10, ceil
        super(MarchSave, self).__init__(cse, **kw)
        nsteps = cse.execution.steps_run
        basefn = cse.io.basefn
        vtkfn_tmpl = basefn + "_%%0%dd"%int(ceil(log10(nsteps))+1)
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
        # create dictionaries for scalars and vectors.
        sarrs = dict()
        varrs = dict()
        for key in var:
            if key == 'soln' or key == 'dsoln': # skip pure solution.
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
        from .io.vtk import VtkLegacyUstGridWriter
        psteps = self.psteps
        cse = self.cse
        blk = self.blk
        # initialize writer.
        self.writer = VtkLegacyUstGridWriter(blk,
            binary=self.binary, cache_grid=self.cache_grid)
        # write initially.
        self._write(0)

    def postmarch(self):
        psteps = self.psteps
        exe = self.cse.execution
        istep = exe.step_current
        vstep = exe.varstep
        if istep%psteps == 0:
            assert istep == vstep   # data must be fresh.
            self._write(istep)

################################################################################
# Vtk XML parallel writers.
################################################################################

class PMarchSave(MarchSave):
    """
    Save the geometry and variables in a case when time marching in parallel
    VTK XML format.

    @ivar anames: the arrays in der of solvers to be saved.  Format is (name,
        inder, ndim), (name, inder, ndim) ...  For ndim > 0 the
        array is a spatial vector, for ndim == 0 a simple scalar, and ndim < 0
        a list of scalar.
    @itype anames: list
    @ivar fpdtype: string for floating point data type (in numpy convention).
    @itype fpdtype: str
    @ivar pextmpl: template for the extension of split VTK file name.
    @itype pextmpl: str
    """
    def __init__(self, cse, **kw):
        import os
        from math import log10, ceil
        self.anames = kw.pop('anames', list())
        self.fpdtype = kw.pop('fpdtype', str(cse.execution.fpdtype))
        super(PMarchSave, self).__init__(cse, **kw)
        # override vtkfn_tmpl.
        nsteps = cse.execution.steps_run
        basefn = cse.io.basefn
        vtkfn_tmpl = basefn + "_%%0%dd"%int(ceil(log10(nsteps))+1) + '.pvtu'
        self.vtkfn_tmpl = os.path.join(cse.io.basedir,
            kw.pop('vtkfn_tmpl', vtkfn_tmpl))
        # craft ext name template.
        npart = cse.execution.npart
        self.pextmpl = '.p%%0%dd'%int(ceil(log10(npart))+1) if npart else ''
        self.pextmpl += '.vtu'
    def drop_anchor(self, svr):
        import os
        from .anchor import MarchSaveAnchor
        basefn = os.path.splitext(self.vtkfn_tmpl)[0]
        anames = dict([(ent[0], ent[1]) for ent in self.anames])
        ankkw = dict(anames=anames, fpdtype=self.fpdtype,
            psteps=self.psteps, vtkfn_tmpl=basefn+self.pextmpl)
        self._deliver_anchor(svr, MarchSaveAnchor, ankkw)
    def _write(self, istep):
        from .io.vtkxml import PVtkXmlUstGridWriter
        if not self.cse.execution.npart:
            return
        sarrs = dict()
        varrs = dict()
        # collect derived.
        for key, inder, ndim in self.anames:
            if ndim > 0:
                varrs[key] = self.fpdtype
            elif ndim < 0:
                for it in range(abs(ndim)):
                    sarrs['%s[%d]' % (key, it)] = self.fpdtype
            else:
                sarrs[key] = self.fpdtype
        # collect unknowns.
        for ieq in range(self.cse.execution.neq):
            sarrs['soln[%d]'%ieq] = 'float64'
        # write.
        wtr = PVtkXmlUstGridWriter(self.blk, scalars=sarrs, vectors=varrs,
            npiece=self.cse.execution.npart, pextmpl=self.pextmpl)
        wtr.write(self.vtkfn_tmpl % istep)
    def preloop(self):
        self._write(0)
    def postmarch(self):
        psteps = self.psteps
        exe = self.cse.execution
        istep = exe.step_current
        vstep = exe.varstep
        if istep%psteps == 0:
            self._write(istep)
