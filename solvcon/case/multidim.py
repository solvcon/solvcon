# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Multi-dimensional simulation cases which use block.
"""

from .core import BaseCase, Hook

class BlockCase(BaseCase):
    """
    Base class for multi-dimensional cases using block.

    Subclass must implement _load_block_for_init() private method for init()
    method to load the needed block.
    """
    defdict = {
        # execution related.
        'execution.npart': None,    # number of decomposed blocks.
        # IO.
        'io.meshfn': None,
        # conditions.
        'condition.bcmap': dict,
        # solver.
        'solver.domaintype': None,
        'solver.domainobj': None,
        'solver.dealer': None,
        'solver.outposts': list,
    }

    @property
    def is_parallel(self):
        """
        Determine if self should do parallel or not.

        @return: 0 means sequential; 1 means local parallel.
        @rtype: int
        """
        from solvcon import domain
        domaintype = self.solver.domaintype
        if domaintype == domain.Domain:
            flag_parallel = 0 # means sequential.
        elif domaintype == domain.Collective:
            flag_parallel = 1 # means local parallel.
        elif domaintype == domain.Distributed:
            flag_parallel = 2 # means network parallel.
        else:
            raise TypeError, 'domaintype shouldn\'t be %s' % domaintype
        return flag_parallel

    def load_block(self):
        """
        Return a block for init.

        @return: a block object.
        @rtype: solvcon.block.Block
        """
        import gzip
        from solvcon.io.gambit import GambitNeutral
        from solvcon.io.block import BlockIO
        meshfn = self.io.meshfn
        bcmapper = self.condition.bcmap
        if '.neu' in meshfn:
            self._log_start('read_neu_data', msg=' from %s'%meshfn)
            if meshfn.endswith('.gz'):
                stream = gzip.open(meshfn)
            else:
                stream = open(meshfn)
            data = stream.read()
            stream.close()
            self._log_end('read_neu_data')
            self._log_start('create_neu_object')
            neu = GambitNeutral(data)
            self._log_end('create_neu_object')
            self._log_start('convert_neu_to_block')
            blk = neu.toblock(bcname_mapper=bcmapper)
            self._log_end('convert_neu_to_block')
        elif '.blk' in meshfn:
            self._log_start('load_block')
            blk = BlockIO().load(stream=meshfn, bcmapper=bcmapper)
            self._log_end('load_block')
        else:
            raise ValueError
        return blk

    def init(self, force=False):
        """
        Load block and initialize solver from the geometry information in the
        block and conditions in the self case.  If parallel run is specified
        (throught domaintype), split the domain and perform corresponding tasks.
        """
        from solvcon import domain
        from solvcon.rpc import Dealer
        from solvcon.boundcond import interface
        preres = super(BlockCase, self).init(force=force)
        solvertype = self.solver.solvertype
        domaintype = self.solver.domaintype

        # load block.
        blk = self.load_block()
        # initilize the whole solver and domain.
        dom = domaintype(blk)
        self.solver.domainobj = dom

        flag_parallel = self.is_parallel
        # for serial execution.
        if flag_parallel == 0:
            assert self.execution.npart == None
            # create and initialize solver.
            solver = solvertype(blk,
                neq=self.execution.neq, fpdtype=self.execution.fpdtype)
            solver.bind()
            solver.init()
            self.solver.solverobj = solver
        # for parallel execution.
        elif flag_parallel > 0:
            assert isinstance(self.execution.npart, int)
            # split the domain.
            dom.split(nblk=self.execution.npart, interface_type=interface)
            nblk = len(dom)
            # make dealer and create workers for the dealer.
            if flag_parallel == 1:
                family = None
                create_workers = self._create_workers_local
            elif flag_parallel == 2:
                family = 'AF_INET'
                create_workers = self._create_workers_remote
            dealer = self.solver.dealer = Dealer(family=family)
            create_workers(dealer, nblk)
            # spread out decomposed solvers.
            for iblk in range(nblk):
                sbk = dom[iblk]
                svr = solvertype(sbk,
                    neq=self.execution.neq, fpdtype=self.execution.fpdtype)
                svr.blkn = iblk
                svr.nblk = nblk
                svr.unbind()    # ensure no pointers (unpicklable) in solver.
                dealer[iblk].remote_setattr('muscle', svr)
            # initialize solvers.
            for sdw in dealer: sdw.cmd.bind()
            for sdw in dealer: sdw.cmd.init()
            dealer.barrier()
            # make interconnections for rpc.
            for iblk in range(nblk):
                for jblk in range(nblk):
                    if iblk >= jblk:
                        continue
                    if dom.interfaces[iblk][jblk] != None:
                        dealer.bridge((iblk, jblk))
            dealer.barrier()
            # exchange solver metrics.
            ifacelists = dom.ifacelists
            for iblk in range(nblk):
                ifacelist = ifacelists[iblk]
                sdw = dealer[iblk]
                sdw.cmd.init_exchange(ifacelist)
            for arrname in solvertype._interface_init_:
                for sdw in dealer: sdw.cmd.exchangeibc(arrname,
                    with_worker=True)

        self._have_init = preres and True
        return self._have_init

    def _get_profiler_data(self, iblk):
        from ..conf import env
        if env.command != None:
            ops, args = env.command.opargs
            if getattr(ops, 'use_profiler'):
                return (
                    ops.profiler_dat+'%d'%iblk,
                    ops.profiler_log+'%d'%iblk,
                    ops.profiler_sort,
                )
        return None

    def _create_workers_local(self, dealer, nblk):
        from solvcon.rpc import Worker
        for iblk in range(nblk):
            dealer.hire(Worker(None,
                profiler_data=self._get_profiler_data(iblk)))

    def _create_workers_remote(self, dealer, nblk):
        import os
        try:
            from multiprocessing.connection import Client
        except ImportError:
            from processing.connection import Client
        from ..rpc import DEFAULT_AUTHKEY, Footway, Shadow
        authkey = DEFAULT_AUTHKEY
        paths = dict([(key, os.environ.get(key, '').split(':')) for key in
            'LD_LIBRARY_PATH',
            'PYTHONPATH',
        ])
        paths['PYTHONPATH'].insert(0, self.io.rootdir)
        sch = self.execution.scheduler(self)
        iworker = 0
        for node in sch:
            inetaddr = node.address
            port = Footway.build_outpost(address=inetaddr, authkey=authkey,
                paths=paths)
            ftw = Footway(address=(inetaddr, port), authkey=authkey)
            ftw.chdir(os.getcwd())
            self.solver.outposts.append(ftw)
            for iblk in range(node.ncore):
                pport = ftw.create(
                    profiler_data=self._get_profiler_data(iworker))
                dealer.appoint(inetaddr, pport, authkey)
                iworker += 1
        assert len(dealer) == nblk

    def run(self):
        """
        Run the simulation case; time marching.

        @return: nothing.
        """
        assert self._have_init
        import sys
        solvertype = self.solver.solvertype
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        # start log.
        self._log_start('run', msg=' '+self.io.basefn, postmsg=' ... \n')
        # prepare for time marching.
        aCFL = 0.0
        self.execution.step_current = self.execution.step_init
        # hook: init.
        self._runhooks('preloop')
        if flag_parallel:
            for sdw in dealer: sdw.cmd.exchangeibc('soln', with_worker=True)
            for sdw in dealer: sdw.cmd.exchangeibc('dsoln', with_worker=True)
            for sdw in dealer: sdw.cmd.boundcond()
            for sdw in dealer: sdw.cmd.update()
        else:
            self.solver.solverobj.boundcond()
            self.solver.solverobj.update()
        # start log.
        self._log_start('loop_march', postmsg='\n')
        while self.execution.step_current < self.execution.steps_run:
            # hook: premarch.
            self._runhooks('premarch')
            # march.
            cCFL = -1.0
            steps_stride = self.execution.steps_stride
            time_increment = self.execution.time_increment
            time = self.execution.step_current*time_increment
            if flag_parallel:
                for sdw in dealer: sdw.cmd.march(time, time_increment,
                    steps_stride, with_worker=True)
                cCFL = max([sdw.recv() for sdw in dealer])
            else:
                cCFL = self.solver.solverobj.march(time, time_increment,
                    steps_stride)
            self.execution.time += time_increment*steps_stride
            # process CFL.
            istep = self.execution.step_current
            aCFL += cCFL*steps_stride
            mCFL = aCFL/(istep+steps_stride)
            self.execution.cCFL = cCFL
            self.execution.aCFL = aCFL
            self.execution.mCFL = mCFL
            # increment to next time step.
            self.execution.step_current += steps_stride
            # hook: postmarch.
            self._runhooks('postmarch')
            # flush standard output/error.
            sys.stdout.flush()
            sys.stderr.flush()
        # start log.
        self._log_end('loop_march')
        # hook: final.
        self._runhooks('postloop')
        # finalize.
        if flag_parallel:
            for sdw in dealer: sdw.cmd.final()
            self.solver.dealer.terminate()
            for ftw in self.solver.outposts: ftw.terminate()
        else:
            self.solver.solverobj.final()
        # end log.
        self._log_end('run')

class BlockHook(Hook):
    """
    Base type for hooks needing a BlockCase.
    """
    def __init__(self, case, **kw):
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
    def __init__(self, case, **kw):
        assert isinstance(case, BlockCase)
        super(BlockInfoHook, self).__init__(case, **kw)

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
        from solvcon.io.vtk import VtkLegacyUstGridWriter
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
        from solvcon.io.vtk import VtkLegacyUstGridWriter
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
