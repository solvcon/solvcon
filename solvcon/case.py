# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Simulation cases.
"""

from .gendata import SingleAssignDict, AttributeDict

class ArrangementRegistry(SingleAssignDict, AttributeDict):
    """
    Arrangement registry class.  A simulation is a callable that returns a case
    object.
    """
    def __setitem__(self, key, value):
        assert callable(value)
        super(ArrangementRegistry, self).__setitem__(key, value)
arrangements = ArrangementRegistry()  # overall registry singleton.

class CaseInfoMeta(type):
    """
    Meta class for case information class.
    """
    def __new__(cls, name, bases, namespace):
        newcls = super(CaseInfoMeta, cls).__new__(cls, name, bases, namespace)
        # incremental modification of defdict.
        defdict = {}
        for base in bases:
            defdict.update(getattr(base, 'defdict', {}))
        defdict.update(newcls.defdict)
        newcls.defdict = defdict
        # create different simulation registry objects for case classes.
        newcls.arrangements = ArrangementRegistry()
        return newcls
class CaseInfo(dict):
    """
    Generic case information abstract class.  It's the base class that all case
    information classes should subclass, to form hierarchical information 
    object.
    """
    __metaclass__ = CaseInfoMeta
    defdict = {}
    def __getattr__(self, name):
        """
        Consult self dictionary for attribute.  It's a shorthand.
        """
        return self[name]
    def __setattr__(self, name, value):
        """
        Save to self dictionary first, then self object table.

        @note: This method is overriden as a stupid-preventer.  It makes
        attribute setting consistent with attribute getting.
        """
        if name in self:
            self[name] = value
        else:
            super(CaseInfo, self).__setattr__(name, value)
    def _set_through(self, key, val):
        """
        Set to self with the dot-separated key.
        """
        tokens = key.split('.', 1)
        fkey = tokens[0]
        if len(tokens) == 2:
            self[fkey]._set_through(tokens[1], val)
        else:
            self[fkey] = val
    def __init__(self, _defdict=None, *args, **kw):
        """
        Assign default values to self after initiated.

        @keyword _defdict: customized defdict; internal use only.
        @type _defdict: dict
        """
        super(CaseInfo, self).__init__(*args, **kw)
        # customize defdict.
        if _defdict is None:
            defdict = self.defdict
        else:
            defdict = dict(self.defdict)
            defdict.update(_defdict)
        # parse first hierarchy to form key groups.
        keygrp = dict()
        for key in defdict.keys():
            if key is None or key == '':
                continue
            tokens = key.split('.', 1)
            if len(tokens) == 2:
                fkey, rkey = tokens
                keygrp.setdefault(fkey, dict())[rkey] = defdict[key]
            else:
                fkey = tokens[0]
                keygrp[fkey] = defdict[fkey]
        # set up first layer keys recursively.
        for fkey in keygrp.keys():
            data = keygrp[fkey]
            if isinstance(data, dict):
                self[fkey] = CaseInfo(_defdict=data)
            elif isinstance(data, type):
                try:
                    self[fkey] = data()
                except TypeError:
                    self[fkey] = data
            else:
                self[fkey] = data

class BaseCase(CaseInfo):
    """
    Base class for simulation cases.

    @ivar runhooks: a special list containing all the hook objects to be run.
    @itype runhooks: solvcon.hook.HookList
    @ivar _have_init: flag that self was initialized or not.
    @itype _have_init: bool
    """

    from . import conf, batch
    defdict = {
        # execution related.
        'execution.fpdtype': conf.env.fpdtypestr,
        'execution.scheduler': batch.Scheduler,
        'execution.resources': dict,    # for scheduler.
        'execution.time': 0.0,
        'execution.time_increment': 0.0,
        'execution.step_init': 0,
        'execution.step_current': None,
        'execution.steps_run': None,
        'execution.steps_stride': 1,
        'execution.cCFL': 0.0,  # current.
        'execution.aCFL': 0.0,  # accumulated.
        'execution.mCFL': 0.0,  # mean.
        'execution.neq': 0, # number of unknowns.
        'execution.var': dict,  # for Calculator hooks.
        'execution.varstep': None,  # the step for which var and dvar are valid.
        # io related.
        'io.abspath': False,    # flag to use abspath or not.
        'io.rootdir': None,
        'io.basedir': None,
        'io.basefn': None,
        'io.empty_jobdir': False,
        # conditions.
        'condition.mtrllist': list,
        # solver.
        'solver.solvertype': None,
        'solver.solverobj': None,
        # logging.
        'log.time': dict,
    }
    del conf, batch
    from .helper import info
    info = staticmethod(info)

    def _log_start(self, action, msg='', postmsg=' ... '):
        """
        Print to user and record start time for certain action.

        @param action: action key.
        @type action: str
        @keyword msg: trailing message for the action key.
        @type msg: str
        @return: nothing.
        """
        from time import time
        tarr = [0,0,0]
        tarr[0] = time()
        self.log.time[action] = tarr
        self.info('Start %s%s%s' % (action, msg, postmsg))

    def _log_end(self, action, msg=''):
        """
        Print to user and record end time for certain action.

        @param action: action key.
        @type action: str
        @keyword msg: supplemental message.
        @type msg: str
        @return: nothing
        """
        from time import time
        tarr = self.log.time.setdefault(action, [0,0,0])
        tarr[1] = time()
        tarr[2] = tarr[1] - tarr[0]
        self.info('done.%s time (sec) = %g\n' % (msg, tarr[2]))

    def __init__(self, **kw):
        """
        Initiailize the basic case.  Set through keyword parameters.
        """
        import os
        from .hook import HookList
        # populate value from keywords.
        initpairs = list()
        for cinfok in self.defdict.keys():
            lkey = cinfok.split('.')[-1]
            initpairs.append((cinfok, kw.pop(lkey, None)))
        # initialize with the left keywords.
        super(BaseCase, self).__init__(**kw)
        # populate value from keywords.
        for cinfok, val in initpairs:
            if val is not None:
                self._set_through(cinfok, val)
        # create runhooks.
        self.runhooks = HookList(self)
        # expand basedir.
        if self.io.abspath:
            self.io.basedir = os.path.abspath(self.io.basedir)
        # second-phase initilization flag.
        self._have_init = False

    def init(self, force=False):
        """
        Second-phase initialization.  This initilization should be performed 
        right before the running of the case.  Subclass should set _have_init
        attribute on the end of this method.

        @keyword force: flag to force initialization no matter self was
            initialized or not.
        @type force: bool
        @return: flag initialized or not.
        @rtype: bool
        """
        assert not self._have_init or force
        self._have_init = True
        return self._have_init

    def run(self):
        """
        Run the simulation case; time marching.

        @return: nothing.
        """
        import sys
        # start log.
        self._log_start('run', msg=' '+self.io.basefn)
        self.info("\n")
        # prepare for time marching.
        aCFL = 0.0
        self.execution.step_current = 0
        self.runhooks('preloop')
        self._log_start('loop_march', postmsg='\n')
        while self.execution.step_current < self.execution.steps_run:
            self.runhooks('premarch')
            cCFL = self.solver.solverobj.march(
                self.execution.step_current*self.execution.time_increment,
                self.execution.time_increment)
            # process CFL.
            istep = self.execution.step_current
            aCFL += cCFL
            mCFL = aCFL/(istep+1)
            self.execution.cCFL = cCFL
            self.execution.aCFL = aCFL
            self.execution.mCFL = mCFL
            # increment to next time step.
            self.execution.step_current += 1
            self.runhooks('postmarch')
            sys.stdout.flush()
        self._log_start('loop_march')
        self.runhooks('postloop')
        # end log.
        self._log_end('run')

    @classmethod
    def register_arrangement(cls, func):
        """
        Decorate simulation functions.  This function assert required signature
        which is necessary for a function to be a valid simulation function.
        Moreover, all the simulation function should be decorated by this
        decorator.

        @return: simulation function.
        @rtype: callable
        """
        from .batch import Scheduler
        def simu(*args, **kw):
            kw.pop('casename', None)
            submit = kw.pop('submit', None)
            resources = kw.pop('resources', dict())
            scheduler = kw.get('scheduler', Scheduler)
            casename = func.__name__
            case = func(casename=casename, *args, **kw)
            if submit:
                sbm = scheduler(case, arnname=casename, **resources)
                sbm()
            else:
                case.init()
                case.run()
        # register self to simulation registries.
        cls.arrangements[func.__name__] = simu
        arrangements[func.__name__] = simu
        return simu

class BlockCase(BaseCase):
    """
    Base class for multi-dimensional cases using block.

    Subclass must implement _load_block_for_init() private method for init()
    method to load the needed block.
    """
    defdict = {
        # execution related.
        'execution.npart': None,    # number of decomposed blocks.
        'execution.steps_dump': None,
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
        from . import domain
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
        from .io.gambit import GambitNeutral
        from .io.block import BlockIO
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
        from . import domain
        from .rpc import Dealer
        from .boundcond import interface
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
            svr = solvertype(blk,
                neq=self.execution.neq, fpdtype=self.execution.fpdtype)
            self.runhooks.drop_anchor(svr)
            svr.bind()
            svr.init()
            self.solver.solverobj = svr
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
                svr.svrn = iblk
                svr.nsvr = nblk
                self.runhooks.drop_anchor(svr)
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
        from .conf import env
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
        from .rpc import Worker
        for iblk in range(nblk):
            dealer.hire(Worker(None,
                profiler_data=self._get_profiler_data(iblk)))

    def _create_workers_remote(self, dealer, nblk):
        import os
        try:
            from multiprocessing.connection import Client
        except ImportError:
            from processing.connection import Client
        from .rpc import DEFAULT_AUTHKEY, Footway, Shadow
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
        if flag_parallel:
            for sdw in dealer: sdw.cmd.provide()
        else:
            self.solver.solverobj.provide()
        # hook: preloop.
        self.runhooks('preloop')
        if flag_parallel:
            for sdw in dealer: sdw.cmd.preloop()
            for sdw in dealer: sdw.cmd.exchangeibc('soln', with_worker=True)
            for sdw in dealer: sdw.cmd.exchangeibc('dsoln', with_worker=True)
            for sdw in dealer: sdw.cmd.boundcond()
        else:
            self.solver.solverobj.preloop()
            self.solver.solverobj.boundcond()
        # start log.
        self._log_start('loop_march', postmsg='\n')
        while self.execution.step_current < self.execution.steps_run:
            # dump before anything.
            if self.execution.steps_dump != None and \
               self.execution.step_current != self.execution.step_init and \
               self.execution.step_current%self.execution.steps_dump == 0:
                self.dump()
            # hook: premarch.
            self.runhooks('premarch')
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
            self.runhooks('postmarch')
            # flush standard output/error.
            sys.stdout.flush()
            sys.stderr.flush()
        # start log.
        self._log_end('loop_march')
        # hook: postloop.
        if flag_parallel:
            for sdw in dealer: sdw.cmd.postloop()
        else:
            self.solver.solverobj.postloop()
        self.runhooks('postloop')
        # finalize.
        if flag_parallel:
            for sdw in dealer: sdw.cmd.exhaust()
        else:
            self.solver.solverobj.exhaust()
        if flag_parallel:
            for sdw in dealer: sdw.cmd.final()
            self.solver.dealer.terminate()
            for ftw in self.solver.outposts: ftw.terminate()
        else:
            self.solver.solverobj.final()
        # end log.
        self._log_end('run')

    def dump(self):
        import cPickle as pickle
        csefn = 'solvcon.dump.case.obj'
        svrfntmpl = 'solvcon.dump.solver%s.obj'
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        # unbind.
        if flag_parallel:
            for sdw in dealer: sdw.cmd.dump(svrfntmpl)
            outposts = self.solver.outposts
        else:
            self.solver.domainobj.unbind()
            self.solver.solverobj.unbind()
        # pickle.
        self.solver.dealer = None
        self.solver.outposts = None
        pickle.dump(self, open(csefn, 'w'), pickle.HIGHEST_PROTOCOL)
        # bind.
        if flag_parallel:
            self.solver.outposts = outposts
            self.solver.dealer = dealer
        else:
            self.solver.solverobj.bind()
            self.solver.domainobj.bind()
