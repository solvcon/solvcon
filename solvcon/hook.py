# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Hooks for simulation cases.
"""

class Hook(object):
    """
    Container class for various hooking subroutines for BaseCase.

    @ivar case: case object.
    @itype case: BaseCase
    @ivar info: information output function.
    @itype info: callable
    @ivar psteps: the interval number of steps between printing.
    @itype psteps: int
    @ivar kws: excessive keywords.
    @itype kws: dict
    """
    def __init__(self, case, **kw):
        """
        @param case: case object.
        @type case: BaseCase
        """
        from .case import BaseCase
        assert isinstance(case, BaseCase)
        self.case = case
        self.info = case.info
        self.psteps = kw.pop('psteps', None)
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
        hooks = self.case.execution.runhooks
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

    def drop_anchor(self, svr):
        """
        Drop the anchor(s) to the solver object.

        @param svr: the solver object on which the anchor(s) is dropped.
        @type svr: solvon.solver.BaseSolver
        @return: nothing
        """
        pass

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

class ProgressHook(Hook):
    """
    Print progess.

    @ivar linewidth: the maximal width for progress symbol.  50 is upper limit.
    @itype linewidth: int
    """

    def __init__(self, case, **kw):
        self.linewidth = kw.pop('linewidth', 50)
        assert self.linewidth <= 50
        super(ProgressHook, self).__init__(case, **kw)

    def preloop(self):
        istep = self.case.execution.step_current
        nsteps = self.case.execution.steps_run
        info = self.info
        info("Steps to run: %d, current step: %d\n" % (nsteps, istep))

    def postmarch(self):
        from time import time
        istep = self.case.execution.step_current
        nsteps = self.case.execution.steps_run
        tstart = self.case.log.time['loop_march'][0]
        psteps = self.psteps
        linewidth = self.linewidth
        cCFL = self.case.execution.cCFL
        info = self.info
        # calculate estimated remaining time.
        tcurr = time()
        tleft = (tcurr-tstart) * ((float(nsteps)-float(istep))/float(istep))
        # output information.
        if istep%psteps == 0:
            info("#")
        if istep > 0 and istep%(psteps*linewidth) == 0:
            info(" %d/%d (%.1fs left) %.2f\n" % (istep, nsteps, tleft, cCFL))
        elif istep == nsteps:
            info(" %d/%d done\n" % (istep, nsteps))

class CflHook(Hook):
    """
    Make sure is CFL number is bounded and print averaged CFL number over time.

    @ivar cflmin: CFL number should be greater than or equal to the value.
    @itype cflmin: float
    @ivar cflmax: CFL number should be less than the value.
    @itype cflmax: float
    @ivar fullstop: flag to stop when CFL is out of bound.  Default True.
    @itype fullstop: bool
    """

    def __init__(self, case, **kw):
        self.cflmin = kw.pop('cflmin', 0.0)
        self.cflmax = kw.pop('cflmax', 1.0)
        self.fullstop = kw.pop('fullstop', True)
        super(CflHook, self).__init__(case, **kw)

    def _notify(self, msg):
        from warnings import warn
        if self.fullstop:
            raise RuntimeError, msg
        else:
            warn(msg)

    def postmarch(self):
        cCFL = self.case.execution.cCFL
        istep = self.case.execution.step_current
        if self.cflmin != None and cCFL < self.cflmin:
            self._notify("CFL = %g < %g after step: %d" % (
                cCFL, self.cflmin, istep))
        if self.cflmax != None and cCFL >= self.cflmax:
            self._notify("CFL = %g >= %g after step: %d" % (
                cCFL, self.cflmax, istep))

    def postloop(self):
        info = self.info
        info("Averaged maximum CFL = %g.\n" % self.case.execution.mCFL)

class BlockHook(Hook):
    """
    Base type for hooks needing a BlockCase.
    """
    def __init__(self, case, **kw):
        from .case import BlockCase
        assert isinstance(case, BlockCase)
        super(BlockHook, self).__init__(case, **kw)

    @property
    def blk(self):
        return self.case.solver.domainobj.blk

    def _collect_interior(self, key, consider_ghost=True):
        """
        @param key: the name of the array to collect in a solver object.
        @type key: str
        @keyword consider_ghost: treat the arrays with the consideration of
            ghost cells.  Default is True.
        @type consider_ghost: bool
        @return: the interior array hold by the solver.
        @rtype: numpy.ndarray
        """
        from numpy import empty
        case = self.case
        ncell = self.blk.ncell
        ngstcell = self.blk.ngstcell
        if case.is_parallel:
            dom = self.case.solver.domainobj
            # collect arrays from solvers.
            dealer = self.case.solver.dealer
            arrs = list()
            for iblk in range(len(dom)):
                dealer[iblk].cmd.pull(key, with_worker=True)
                arr = dealer[iblk].recv()
                arrs.append(arr)
            # create global array.
            shape = [it for it in arrs[0].shape]
            shape[0] = ncell
            arrg = empty(shape, dtype=arrs[0].dtype)
            # set global array.
            clmaps = dom.mappers[2]
            for iblk in range(len(dom)):
                blk = dom[iblk]
                slctg = (clmaps[:,1] == iblk)
                slctl = clmaps[slctg,0]
                if consider_ghost:
                    slctl += blk.ngstcell
                arrg[slctg] = arrs[iblk][slctl]
        else:
            if consider_ghost:
                start = ngstcell
            else:
                start = 0
            arrg = getattr(case.solver.solverobj, key)[start:].copy()
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
        case = self.case
        ncell = self.blk.ncell
        ngstcell = self.blk.ngstcell
        if case.is_parallel:
            dom = self.case.solver.domainobj
            dealer = self.case.solver.dealer
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
            getattr(case.solver.solverobj, key)[start:] = arrg[:]

class BlockInfoHook(BlockHook):
    def preloop(self):
        blk = self.blk
        self.info("Block information:\n  %s\n" % str(blk))

    def postloop(self):
        ncell = self.blk.ncell
        time = self.case.log.time['loop_march'][2]
        step_init = self.case.execution.step_init
        step_current = self.case.execution.step_current
        perf = time/(step_current-step_init)/ncell * 1.e6
        self.info('Performance: %g microseconds/iteration/cell.\n' % perf)

class Initializer(BlockHook):
    """
    Base type for initializer for case with a block.

    @cvar _varnames_: variables to be set, it takes the format of:
        (
            ('arrname', True/False for putback,),
            ...,
        )
    @ctype _varnames_: tuple
    """

    _varnames_ = tuple()

    def _take_data(self):
        """
        Take data from solver or solvers.

        @return: dict (sequential) or list of dicts (parallel) for taken data.
        @rtype: dict/list
        """
        case = self.case
        if case.is_parallel > 0:
            dealer = case.solver.dealer
            datas = list()
            for sdw in dealer:
                data = dict()
                for key, putback in self._varnames_:
                    sdw.cmd.pull(key, with_worker=True)
                    data[key] = sdw.recv()
                datas.append(data)
            return datas
        else:
            solver = case.solver.solverobj
            data = dict()
            for key, putback in self._varnames_:
                arr = getattr(solver, key)
                data[key] = arr
            return data

    def _set_data(self, **kw):
        """
        Subclass must override.

        @return: nothing.
        """
        raise NotImplementedError

    def _put_data(self, datas):
        """
        Put set data to solver or solvers.

        @param datas: dict of data (sequential) or list of dicts of data
            (parallel) to put.
        @type datas: dict/list
        @return: nothing.
        """
        case = self.case
        if case.is_parallel > 0:
            dealer = case.solver.dealer
            for isvr in range(len(dealer)):
                sdw = dealer[isvr]
                data = datas[isvr]
                for key, putback in self._varnames_:
                    arr = data[key]
                    sdw.cmd.push(arr, key)
        else:
            pass    # nothing need to do for sequetial.

    def preloop(self):
        from numpy import empty
        case = self.case
        datas = self._take_data()
        if case.is_parallel > 0:
            for data in datas: self._set_data(**data)
        else:
            self._set_data(**datas)
        self._put_data(datas)

class Calculator(BlockHook):
    """
    Base type for calculator.
    """

    def _collect_solutions(self):
        """
        Collect solution variables from solver(s) to case.  This method can be
        overridden for a series of collecting.

        @return: the collected soln and dsoln.
        @rtype: tuple
        """
        from numpy import empty, isnan
        case = self.case
        exe = case.execution
        istep = exe.step_current
        vstep = exe.varstep
        # collect original variables.
        if istep == vstep:
            soln = exe.var['soln']
            dsoln = exe.var['dsoln']
        else:
            soln = exe.var['soln'] = self._collect_interior('soln')
            dsoln = exe.var['dsoln'] = self._collect_interior('dsoln')
        # check for nan.
        nans = isnan(soln)
        if nans.any():
            raise ValueError, 'nan occurs'
        nans = isnan(dsoln)
        if nans.any():
            raise ValueError, 'nan occurs'
        # update valid step.
        exe.varstep = istep
        # return
        return soln, dsoln

    def _calculate(self):
        """
        Calculate values and save them into the case.

        @return: nothing.
        """
        raise NotImplementedError

class NpySave(BlockHook):
    """
    Save data into Numpy npy/npz format.

    @ivar name: name of the array to be saved.
    @itype name: str
    @ivar compress: flag to use npz or npy.
    @itype compress: bool
    @ivar fntmpl: customizer for the filename template.
    @itype fntmpl: str
    """

    def __init__(self, case, **kw):
        from math import log10, ceil
        self.name = kw.pop('name', None)
        self.compress = kw.pop('compress', False)
        fntmpl = kw.pop('fntmpl', None)
        super(NpySave, self).__init__(case, **kw)
        if fntmpl == None:
            nsteps = case.execution.steps_run
            basefn = case.io.basefn
            fntmpl = basefn
            if self.name:
                fntmpl += '_%s'%self.name
            fntmpl += '_%%0%dd'%int(ceil(log10(nsteps))+1)
            if self.compress:
                fntmpl += '.npz'
            else:
                fntmpl += '.npy'
        self.fntmpl = fntmpl

    def _save(self, arr, istep):
        """
        @param arr: the array to be written.
        @type arr: numpy.ndarray
        @param istep: the time step.
        @type istep: int
        """
        from numpy import save, savez
        if self.compress:
            write = savez
        else:
            write = save
        fn = self.fntmpl % istep
        write(fn, arr)

class VarSave(NpySave):
    """
    Save soln to npy file.
    """

    def __init__(self, case, **kw):
        assert 'name' in kw
        assert 'fntmpl' not in kw
        super(VarSave, self).__init__(case, **kw)

    def preloop(self):
        istep = self.case.execution.step_current
        soln = self.case.execution.var[self.name]
        self._save(soln, istep)

    def postmarch(self):
        psteps = self.psteps
        istep = self.case.execution.step_current
        if istep%psteps == 0:
            soln = self.case.execution.var[self.name]
            self._save(soln, istep)

class FinalCompare(BlockHook):
    """
    Compare certain variable array after finishing the main loop.

    @ivar goldfn: file name for the gold data.
    @itype goldfn: str
    @ivar varname: variable name.
    @itype varname: str
    @ivar absdiff: the sum of absolute difference.  This variable is set (not
        None) only after the loop is finished.
    @itype absdiff: int
    """
    def __init__(self, case, **kw):
        self.goldfn = kw.pop('goldfn')
        self.varname = kw.pop('varname')
        super(FinalCompare, self).__init__(case, **kw)
        self.absdiff = None

    def postloop(self):
        from numpy import load, abs
        info = self.info
        try:
            gold = load(self.goldfn)
        except:
            info('Error in loading gold file %s.\n' % self.goldfn)
            return
        arr = self.case.execution.var[self.varname]
        absdiff = abs(arr - gold).sum()
        info('Absolute summation of difference in %s (compared against\n'
             '  %s\n'
             ') is %g .\n' % (
            self.varname, self.goldfn, absdiff))
        # save result to self.
        self.absdiff = absdiff

class VtkSave(BlockHook):
    """
    Base type for writer for case with a block.

    @ivar binary: True for BINARY format; False for ASCII.
    @itype binary: bool
    @ivar cache_grid: True to cache grid; False to forget grid every time.
    @itype cache_grid: bool
    """
    def __init__(self, case, **kw):
        self.binary = kw.pop('binary', False)
        self.cache_grid = kw.pop('cache_grid', True)
        super(VtkSave, self).__init__(case, **kw)

class SplitSave(VtkSave):
    """
    Save the splitted geometry.
    """

    def preloop(self):
        from math import log10, ceil
        from .io.vtk import VtkLegacyUstGridWriter
        case = self.case
        if case.is_parallel == 0:
            return  # do nothing if not in parallel.
        basefn = case.io.basefn
        dom = case.solver.domainobj
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
    def __init__(self, case, **kw):
        from math import log10, ceil
        super(MarchSave, self).__init__(case, **kw)
        nsteps = case.execution.steps_run
        basefn = case.io.basefn
        vtkfn_tmpl = basefn + "_%%0%dd"%int(ceil(log10(nsteps))+1)
        if self.binary:
            vtkfn_tmpl += ".bin.vtk"
        else:
            vtkfn_tmpl += ".vtk"
        self.vtkfn_tmpl = kw.pop('vtkfn_tmpl', vtkfn_tmpl)

    @property
    def data(self):
        """
        Get dictionaries for scalar and vector data from case.

        @return: dictionaries for scalar and vector.
        @rtype: tuple
        """
        case = self.case
        exe = case.execution
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
        case = self.case
        blk = self.blk
        # initialize writer.
        self.writer = VtkLegacyUstGridWriter(blk,
            binary=self.binary, cache_grid=self.cache_grid)
        # write initially.
        self._write(0)

    def postmarch(self):
        psteps = self.psteps
        exe = self.case.execution
        istep = exe.step_current
        vstep = exe.varstep
        if istep%psteps == 0:
            assert istep == vstep   # data must be fresh.
            self._write(istep)
