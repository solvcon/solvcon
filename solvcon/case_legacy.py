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
Legacy implementation of simulation cases.

This module should not be imported directly.  Necessary interface is exposed in
:py:mod:`solvcon.case` and should be imported from there.
"""


import os
import sys
import time
import traceback
from cStringIO import StringIO
import signal
import cPickle as pickle
import gzip

from . import conf
from . import batch
from . import case_core
from . import helper
from . import domain
from . import boundcond
from . import rpc
from .io import gmsh as iogmsh
from .io import genesis as iogenesis
from .io import gambit as iogambit
from .io import block as ioblock
from .io import domain as iodomain

from . import case_core


# import core.
from .case_core import arrangements

class BaseCase(case_core.CaseInfo):
    """
    Base class for simulation cases.

    init() and run() are the two primary methods responsible for the
    execution of the simulation case object.  Both methods accept a keyword
    parameter ``level'' which indicates the run level of the run:
      - run level 0: fresh run (default),
      - run level 1: restart run,
      - run level 2: initialization only.

    @ivar runhooks: a special list containing all the hook objects to be run.
    @itype runhooks: solvcon.hook.HookList
    """

    CSEFN_DEFAULT = 'solvcon.dump.case.obj'

    defdict = {
        # execution related.
        'execution.fpdtype': conf.env.fpdtypestr,
        'execution.batch': batch.Batch,
        'execution.resources': dict,    # for batch.
        'execution.stop': False,
        'execution.time': 0.0,
        'execution.time_increment': 0.0,
        'execution.step_init': 0,
        'execution.step_current': None,
        'execution.steps_run': None,
        'execution.steps_stride': 1,
        'execution.marchret': None,
        'execution.neq': 0, # number of unknowns.
        'execution.var': dict,  # for Calculator hooks.
        'execution.varstep': None,  # the step for which var and dvar are valid.
        'execution.ncore': -1,  # number of cores to use in solver.
        # dynamic related.
        'dynamic.inputfn': 'solvcon.input',
        'dynamic.bakfn': 'solvcon.input.bak',
        'dynamic.preserve': False,
        # io related.
        'io.abspath': False,    # flag to use abspath or not.
        'io.rootdir': None,
        'io.basedir': None,
        'io.basefn': None,
        'io.empty_jobdir': False,
        'io.solver_output': False,
        # conditions.
        'condition.mtrllist': list,
        # solver.
        'solver.solvertype': None,
        'solver.solverobj': None,
        # logging.
        'log.time': dict,
    }

    info = helper.info

    def _log_start(self, action, msg='', postmsg=' ... '):
        """
        Print to user and record start time for certain action.

        @param action: action key.
        @type action: str
        @keyword msg: trailing message for the action key.
        @type msg: str
        @return: nothing.
        """
        info = self.info
        tarr = [0,0,0]
        tarr[0] = time.time()
        self.log.time[action] = tarr
        info(
            info.prefix * (info.width-info.level*info.nchar),
            travel=1
        )
        info('\nStart %s%s%s' % (action, msg, postmsg))
        info(
            '\n' + info.prefix * (info.width-info.level*info.nchar) + '\n',
        )

    def _log_end(self, action, msg='', postmsg=' . '):
        """
        Print to user and record end time for certain action.

        @param action: action key.
        @type action: str
        @keyword msg: supplemental message.
        @type msg: str
        @return: nothing
        """
        info = self.info
        tarr = self.log.time.setdefault(action, [0,0,0])
        tarr[1] = time.time()
        tarr[2] = tarr[1] - tarr[0]
        info(
            info.prefix * (info.width-info.level*info.nchar) + \
            '\nEnd %s%s%sElapsed time (sec) = %g' % (
                action, msg, postmsg, tarr[2]
            )
        )
        info(
            '\n' + info.prefix * (info.width-(info.level-1)*info.nchar) + '\n',
            travel=-1
        )

    def __init__(self, **kw):
        """
        Initiailize the basic case.  Set through keyword parameters.
        """
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
        self.runhooks = case_core.HookList(self)
        # expand basedir.
        if self.io.abspath:
            self.io.basedir = os.path.abspath(self.io.basedir)
        if self.io.basedir is not None and not os.path.exists(self.io.basedir):
            os.makedirs(self.io.basedir)

    def _dynamic_execute(self):
        """
        Dynamically execute the codes stored in the input file specified by
        the case.

        @return: nothing
        """
        if not self.dynamic.inputfn: return
        try:
            # load codes.
            if os.path.exists(self.dynamic.inputfn):
                f = open(self.dynamic.inputfn, 'r')
                codes = f.read()
                f.close()
            else:
                codes = ''
            if codes.strip():
                # clear/preserve code file.
                f = open(self.dynamic.inputfn, 'w')
                if self.dynamic.preserve:
                    f.write(codes)
                f.close()
                # backup codes.
                f = open(self.dynamic.bakfn, 'a')
                f.write('\n### %s step %d/%d\n' % (self.io.basefn,
                    self.execution.step_current, self.execution.steps_run,
                ))
                f.write(codes)
                f.close()
                # run codes.
                exec(codes)
        except Exception, e:
            f = StringIO()
            f.write('\n@@@ dynamic execution at step %d @@@' %
                self.execution.step_current)
            f.write('\nCode:\n %s\n' % codes)
            traceback.print_exc(file=f)
            self.info(f.getvalue())
            f.close()
        # reset preservation flag.
        self.dynamic.preserve = False

    def init(self, level=0):
        """
        Initialize solver.  Nothing inside now.

        @keyword level: run level; higher level does less work.
        @type level: int

        @return: nothing.
        """
        pass

    def run(self, level=0):
        """
        Run the simulation case; time marching.

        @keyword level: run level; higher level does less work.
        @type level: int

        @return: nothing.
        """
        # start log.
        self._log_start('run', msg=' '+self.io.basefn)
        self.info("\n")
        # prepare for time marching.
        self.execution.step_current = 0
        self.runhooks('preloop')
        self._log_start('loop_march')
        while self.execution.step_current < self.execution.steps_run:
            self.runhooks('premarch')
            self.execution.marchret = self.solver.solverobj.march(
                self.execution.step_current*self.execution.time_increment,
                self.execution.time_increment)
            self.execution.step_current += 1
            self.runhooks('postmarch')
        self._log_start('loop_march')
        self.runhooks('postloop')
        # end log.
        self._log_end('run')

    def cleanup(self, signum=None, frame=None):
        if signum == signal.SIGINT:
            raise KeyboardInterrupt

    @classmethod
    def register_arrangement(cls, func, casename=None):
        """
        Decorate simulation functions.  This function asserts required
        signature which is necessary for a function to be a valid simulation
        function.  Moreover, all the simulation function should be decorated by
        this decorator.

        @return: simulation function.
        @rtype: callable
        """
        if casename is None: casename = func.__name__
        def simu(*args, **kw):
            kw.pop('casename', None)
            resources = kw.pop('resources', dict())
            batcher = kw.get('batch', batch.Batch)
            submit = kw.pop('submit')
            use_mpi = kw.pop('use_mpi', False)
            postpone = kw.pop('postpone', False)
            runlevel = kw.pop('runlevel')
            # obtain the case object.
            if runlevel == 1:
                case = pickle.load(open(cls.CSEFN_DEFAULT, 'rb'))
            else:
                case = func(casename=casename, *args, **kw)
            # submit/run.
            try:
                if submit:
                    sbm = batcher(case, arnname=casename, use_mpi=use_mpi,
                        **resources)
                    sbm(runlevel=runlevel, postpone=postpone)
                else:
                    signal.signal(signal.SIGTERM, case.cleanup)
                    signal.signal(signal.SIGINT, case.cleanup)
                    case.init(level=runlevel)
                    case.info('\n')
                    case.run(level=runlevel)
                    case.cleanup()
            except:
                case.cleanup()
                raise
            return case
        # register self to simulation registries.
        cls.arrangements[casename] = simu
        arrangements[casename] = simu
        return simu

class BlockCase(BaseCase):
    """
    Base class for multi-dimensional cases using block.

    @ivar pythonpaths: extra python paths.
    @itype pythonpaths: list
    """
    defdict = {
        # execution related.
        'execution.npart': None,    # number of decomposed blocks.
        'execution.step_restart': None,
        'execution.steps_dump': None,
        # IO.
        'io.mesher': None,
        'io.meshfn': None,
        'io.domain.with_arrs': True,
        'io.domain.with_whole': True,
        'io.domain.wholefn': None,
        'io.domain.splitfns': None,
        'io.rkillfn': 'solvcon.kill.sh',
        'io.dump.csefn': 'solvcon.dump.case.obj',
        'io.dump.svrfntmpl': 'solvcon.dump.solver%s.obj',
        # conditions.
        'condition.bcmap': None,
        'condition.bcmod': None,
        # solver.
        'solver.use_incenter': False,
        'solver.domaintype': None,
        'solver.domainobj': None,
        'solver.dealer': None,
        'solver.envar': dict,
        'solver.ibcthread': False,
    }

    def __init__(self, **kw):
        self.pythonpaths = kw.pop('pythonpaths', [])
        super(BlockCase, self).__init__(**kw)

    @property
    def is_parallel(self):
        """
        Determine if self should do parallel or not.

        @return: 0 means sequential; 1 means local parallel.
        @rtype: int
        """
        domaintype = self.solver.domaintype
        if domaintype == domain.Domain:
            assert self.execution.npart == None
            flag_parallel = 0 # means sequential.
        elif domaintype == domain.Collective:
            assert isinstance(self.execution.npart, int)
            flag_parallel = 1 # means local parallel.
        elif domaintype == domain.Distributed:
            assert isinstance(self.execution.npart, int)
            flag_parallel = 2 # means network parallel.
        else:
            raise TypeError, 'domaintype shouldn\'t be %s' % domaintype
        return flag_parallel

    def dump(self):
        """
        Dump case and remote solver objects for later restart.

        @return: nothing
        """
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        # record the step can be restarted from.
        self.execution.step_restart = self.execution.step_current
        # unbind.
        if flag_parallel:
            for iblk in range(len(self.solver.domainobj)):
                dumpfn = self.io.dump.svrfntmpl % str(iblk)
                dumpfn = os.path.join(self.io.basedir, dumpfn)
                dealer[iblk].cmd.dump(dumpfn)
        else:
            self.solver.solverobj.unbind()
            svrholds = dict()
            for key in ['mesg',]:
                svrholds[key] = getattr(self.solver.solverobj, key)
                setattr(self.solver.solverobj, key, None)
        # pickle.
        self.solver.dealer = None
        dumpfn = os.path.join(self.io.basedir, self.io.dump.csefn)
        pickle.dump(self, open(dumpfn, 'wb'), pickle.HIGHEST_PROTOCOL)
        # bind.
        if flag_parallel:
            self.solver.dealer = dealer
        else:
            for key in svrholds:
                setattr(self.solver.solverobj, key, svrholds[key])
            self.solver.solverobj.bind()

    def cleanup(self, signum=None, frame=None):
        if self.solver.solverobj != None:
            self.solver.solverobj.unbind()
        super(BlockCase, self).cleanup(signum=signum, frame=frame)

    ############################################################################
    ###
    ### Begin of block of case initialization logics.
    ###
    ############################################################################

    def init(self, level=0):
        """
        Load block and initialize solver from the geometry information in the
        block and conditions in the self case.  If parallel run is specified
        (through domaintype), split the domain and perform corresponding tasks.
        """
        self._log_start('init', msg=' (level %d) %s' % (level, self.io.basefn))
        # initilize the whole solver and domain.
        if level != 1:
            self._log_start('build_domain')
            loaded = self.load_block()
            if callable(self.condition.bcmod):
                self.condition.bcmod(loaded)
            if isinstance(loaded, self.solver.domaintype):
                self.solver.domainobj = loaded
            else:
                self.solver.domainobj = self.solver.domaintype(loaded)
            self._log_end('build_domain')
        # for serial execution.
        if not self.is_parallel:
            # create and initialize solver.
            if level != 1:
                self._local_init_solver()
            else:
                self._local_bind_solver()
        # for parallel execution.
        else:
            # split the domain.
            if level != 1 and not self.solver.domainobj.presplit:
                self.info('\n')
                self._log_start('split_domain')
                self.solver.domainobj.split(
                    nblk=self.execution.npart,
                    interface_type=boundcond.interface)
                self._log_end('split_domain')
            # make dealer and create workers for the dealer.
            self.info('\n')
            self._log_start('build_dealer')
            self.solver.dealer = self._create_workers()
            self._log_end('build_dealer')
            # make interconnections for rpc.
            self.info('\n')
            self._log_start('interconnect')
            self._interconnect()
            self._log_end('interconnect')
            # spread out and initialize decomposed solvers.
            if level != 1:
                self.info('\n')
                self._log_start('remote_init_solver')
                self._remote_init_solver()
                self._log_end('remote_init_solver')
            else:
                self.info('\n')
                self._log_start('remote_load_solver')
                self._remote_load_solver()
                self._log_end('remote_load_solver')
            # initialize interfaces.
            self.info('\n')
            self._log_start('init_interface')
            self._init_interface()
            self._log_end('init_interface')
            # initialize exchange for remote solver objects.
            if level != 1:
                self.info('\n')
                self._log_start('exchange_metric')
                self._exchange_metric()
                self._log_end('exchange_metric')
        self._log_end('init', msg=' '+self.io.basefn)

    def load_block(self):
        """
        Return a block for init.

        @return: a block object.
        @rtype: solvcon.block.Block
        """
        meshfn = self.io.meshfn
        bcmapper = self.condition.bcmap
        self.info('mesh file: %s\n' % meshfn)
        if callable(self.io.mesher):
            self._log_start('create_block')
            obj = self.io.mesher(self)
            self._log_end('create_block')
        elif os.path.isdir(meshfn):
            dof = iodomain.DomainIO(dirname=meshfn)
            obj, whole, split = dof.load(bcmapper=bcmapper,
                with_arrs=self.io.domain.with_arrs,
                with_whole=self.io.domain.with_whole, with_split=False,
                return_filenames=True, domaintype=self.solver.domaintype)
            self.io.domain.wholefn = whole
            self.io.domain.splitfn = split
        elif '.msh' in meshfn:
            self._log_start('create_gmsh_object')
            if meshfn.endswith('.gz'):
                stream = gzip.open(meshfn)
            else:
                stream = open(meshfn)
            gmh = iogmsh.Gmsh(stream)
            gmh.load()
            stream.close()
            self._log_end('create_gmsh_object')
            self._log_start('convert_gmsh_to_block')
            obj = gmh.toblock(bcname_mapper=bcmapper,
                use_incenter=self.solver.use_incenter)
            self._log_end('convert_msh_to_block')
        elif meshfn.endswith('.g'):
            self._log_start('create_genesis_object')
            gn = iogenesis.Genesis(meshfn)
            gn.load()
            gn.close_file()
            self._log_end('create_genesis_object')
            self._log_start('convert_genesis_to_block')
            obj = gn.toblock(bcname_mapper=bcmapper,
                use_incenter=self.solver.use_incenter)
            self._log_end('convert_genesis_to_block')
        elif '.neu' in meshfn:
            self._log_start('read_neu_data', msg=' from %s'%meshfn)
            if meshfn.endswith('.gz'):
                stream = gzip.open(meshfn)
            else:
                stream = open(meshfn)
            data = stream.read()
            stream.close()
            self._log_end('read_neu_data')
            self._log_start('create_neu_object')
            neu = iogambit.GambitNeutral(data)
            self._log_end('create_neu_object')
            self._log_start('convert_neu_to_block')
            obj = neu.toblock(bcname_mapper=bcmapper,
                use_incenter=self.solver.use_incenter)
            self._log_end('convert_neu_to_block')
        elif '.blk' in meshfn:
            self._log_start('load_block')
            obj = ioblock.BlockIO().load(stream=meshfn, bcmapper=bcmapper)
            self._log_end('load_block')
        else:
            raise ValueError(meshfn)
        return obj

    def make_solver_keywords(self):
        """
        Return keywords to initialize solvers.

        @return: keywords
        @rtype: dict
        """
        return dict(
            ncore=self.execution.ncore,
            neq=self.execution.neq,
            fpdtype=self.execution.fpdtype,
            enable_mesg=self.io.solver_output,
        )

    # solver object initialization/binding/loading.
    def _local_init_solver(self):
        """
        @return: nothing
        """
        svr = self.solver.solvertype(
            self.solver.domainobj.blk, **self.make_solver_keywords())
        self.runhooks.drop_anchor(svr)
        svr.bind()
        svr.ibcthread = self.solver.ibcthread
        svr.init()
        self.solver.solverobj = svr
    def _local_bind_solver(self):
        """
        @return: nothing
        """
        self.solver.solverobj.bind()
        self.solver.solverobj.ibcthread = self.solver.ibcthread
        self.solver.domainobj.bind()
    def _remote_init_solver(self):
        """
        @return: nothing
        """
        dealer = self.solver.dealer
        solvertype = self.solver.solvertype
        dom = self.solver.domainobj
        nblk = dom.nblk
        for iblk in range(nblk):
            svrkw = self.make_solver_keywords()
            self.info('solver #%d/(%d-1): ' % (iblk, nblk))
            if dom.presplit:
                dealer[iblk].create_solver(self.condition.bcmap,
                    self.io.meshfn, self.io.domain.splitfn[iblk],
                    iblk, nblk, solvertype, svrkw)
                self.runhooks.drop_anchor(dealer[iblk])
            else:
                sbk = dom[iblk]
                svr = solvertype(sbk, **svrkw)
                self.info('sending ... ')
                svr.svrn = iblk
                svr.nsvr = nblk
                self.runhooks.drop_anchor(svr)
                svr.unbind()    # ensure no pointers (unpicklable) in solver.
                dealer[iblk].remote_setattr('muscle', svr)
            self.info('done.\n')
        self.info('Bind/Init ... ')
        for sdw in dealer: sdw.cmd.bind()
        for sdw in dealer: sdw.cmd.remote_setattr('ibcthread',
            self.solver.ibcthread)
        for sdw in dealer: sdw.cmd.init()
        dealer.barrier()
        self.info('done.\n')
    def _remote_load_solver(self):
        """
        @return: nothing
        """
        dealer = self.solver.dealer
        nblk = self.solver.domainobj.nblk
        for iblk in range(nblk):
            dealer[iblk].remote_loadobj('muscle',
                self.io.dump.svrfntmpl % str(iblk))
        for sdw in dealer: sdw.cmd.bind()
        for sdw in dealer: sdw.cmd.remote_setattr('ibcthread',
            self.solver.ibcthread)
        dealer.barrier()

    # workers and worker manager (dealer) creation.
    def _create_workers(self):
        """
        Make dealer and create workers for the dealer.

        @return: worker manager.
        @rtype: solvcon.rpc.Dealer
        """
        nblk = self.solver.domainobj.nblk
        flag_parallel = self.is_parallel
        if flag_parallel == 1:
            family = None
            create_workers = self._create_workers_local
        elif flag_parallel == 2:
            family = 'AF_INET'
            create_workers = self._create_workers_remote
        dealer = rpc.Dealer(family=family)
        create_workers(dealer, nblk)
        return dealer
    def _get_profiler_data(self, iblk):
        if conf.env.command != None:
            ops, args = conf.env.command.opargs
            if getattr(ops, 'use_profiler'):
                return (
                    ops.profiler_dat+'%d'%iblk,
                    ops.profiler_log+'%d'%iblk,
                    ops.profiler_sort,
                )
        return None
    def _create_workers_local(self, dealer, nblk):
        for iblk in range(nblk):
            dealer.hire(rpc.Worker(None,
                profiler_data=self._get_profiler_data(iblk)))
    def _create_workers_remote(self, dealer, nblk):
        info = self.info
        authkey = rpc.DEFAULT_AUTHKEY
        paths = dict([(key, os.environ.get(key, '').split(':')) for key in
            'LD_LIBRARY_PATH',
            'PYTHONPATH',
        ])  # TODO: make sure VTK in LD_LIBRARY_PATH.
        paths['PYTHONPATH'].extend(self.pythonpaths)
        paths['PYTHONPATH'].insert(0, self.io.rootdir)
        # appoint remote worker objects.
        info('Appoint remote workers')
        bat = self.execution.batch(self)
        nodelist = bat.nodelist()
        if conf.env.command != None and conf.env.command.opargs[0].compress_nodelist:
            info(' (compressed)')
        if conf.env.mpi:
            info(' (head excluded for MPI)')
        info(':\n')
        iworker = 0 
        for node in nodelist:
            info('  %s' % node.name)
            port = bat.create_worker(node, authkey,
                envar=self.solver.envar, paths=paths,
                profiler_data=self._get_profiler_data(iworker))
            info(' worker #%d created' % iworker)
            dealer.appoint(node.address, port, authkey)
            info(' and appointed.\n')
            iworker += 1
        if len(dealer) != nblk:
            raise IndexError('%d != %d' % (len(dealer), nblk))
        # create remote killer script.
        if self.io.rkillfn:
            f = open(self.io.rkillfn, 'w')
            f.write("""#!/bin/sh
nodes="
%s
"
for node in $nodes; do rsh $node killall %s; done
""" % (
                '\n'.join([node.name for node in nodelist]),
                os.path.split(sys.executable)[-1],
            ))

    # interconnection.
    def _interconnect(self):
        """
        Make interconnections for distributed solver objects.

        @return: nothing
        """
        dom = self.solver.domainobj
        dealer = self.solver.dealer
        dwidth = len(str(dom.nblk-1))
        oblk = -1
        for iblk, jblk in dom.ifparr:
            if iblk != oblk:
                if oblk != -1:
                    self.info('.\n')
                self.info(('%%0%dd ->' % dwidth) % iblk)
                oblk = iblk
            dealer.bridge((iblk, jblk))
            self.info((' %%0%dd'%dwidth) % jblk)
        self.info('.\n')
        dealer.barrier()

    # interface.
    def _init_interface(self):
        """
        Exchange meta data.

        @return: nothing
        """
        dom = self.solver.domainobj
        dealer = self.solver.dealer
        nblk = dom.nblk
        iflists = dom.make_iflist_per_block()
        self.info('Interface exchanging pairs (%d phases):\n' % len(
            iflists[0]))
        dwidth = len(str(nblk-1))
        for iblk in range(nblk):
            ifacelist = iflists[iblk]
            sdw = dealer[iblk]
            sdw.cmd.init_exchange(ifacelist)
            # print.
            self.info(('%%0%dd ->' % dwidth) % iblk)
            for pair in ifacelist:
                if pair < 0:
                    stab = '-' * (2*dwidth+1)
                else:
                    stab = '-'.join([('%%0%dd'%dwidth)%item for item in pair])
                self.info(' %s' % stab)
            self.info('\n')

    def _exchange_metric(self):
        """
        Exchange metric data for solver.

        @return: nothing
        """
        dealer = self.solver.dealer
        for arrname in self.solver.solvertype._interface_init_:
            for sdw in dealer: sdw.cmd.exchangeibc(arrname,
                with_worker=True)

    ############################################################################
    ###
    ### End of block of case initialization logics.
    ###
    ############################################################################

    ############################################################################
    ###
    ### Begin of block of case execution.
    ###
    ############################################################################

    def run(self, level=0):
        """
        Run the simulation case; time marching.

        @keyword level: run level; higher level does less work.
        @type level: int

        @return: nothing.
        """
        self._log_start('run', msg=' (level %d) %s' % (level, self.io.basefn))
        self.execution.step_current = self.execution.step_init
        if level < 1:
            self._run_provide()
            self._run_preloop()
        if level < 2:
            self._run_march()
            self._run_postloop()
            self._run_exhaust()
        else:   # level == 2.
            self.dump()
        self._run_final()
        self._log_end('run', msg=' '+self.io.basefn)

    # logics before entering main loop (march).
    def _run_provide(self):
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        # anchor: provide.
        if flag_parallel:
            for sdw in dealer: sdw.cmd.provide()
        else:
            self.solver.solverobj.provide()
    def _run_preloop(self):
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        # hook: preloop.
        self.runhooks('preloop')
        if flag_parallel:
            for sdw in dealer: sdw.cmd.preloop()
            for arrname in self.solver.solvertype._solution_array_:
                for sdw in dealer:
                    sdw.cmd.exchangeibc(arrname, with_worker=True)
            for sdw in dealer: sdw.cmd.boundcond()
        else:
            self.solver.solverobj.preloop()
            self.solver.solverobj.boundcond()

    def _run_march(self):
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        self.log.time['solver_march'] = 0.0
        self.info('\n')
        self._log_start('loop_march')
        while self.execution.step_current < self.execution.steps_run:
            # the first thing is detecting for dynamic codes.
            self._dynamic_execute()
            if self.execution.stop: break
            # dump before hooks.
            if self.execution.steps_dump != None and \
               self.execution.step_current != self.execution.step_restart and \
               self.execution.step_current%self.execution.steps_dump == 0:
                self.dump()
            # hook: premarch.
            self.runhooks('premarch')
            # march.
            solver_march_marker = time.time()
            steps_stride = self.execution.steps_stride
            time_increment = self.execution.time_increment
            time_current = self.execution.step_current*time_increment
            if flag_parallel:
                for sdw in dealer: sdw.cmd.march(
                    time_current, time_increment, steps_stride,
                    with_worker=True)
                self.execution.marchret = [sdw.recv() for sdw in dealer]
            else:
                self.execution.marchret = self.solver.solverobj.march(
                    time_current, time_increment, steps_stride)
            self.execution.time += time_increment*steps_stride
            self.log.time['solver_march'] += time.time() - solver_march_marker
            self.execution.step_current += steps_stride
            # hook: postmarch.
            self.runhooks('postmarch')
        # end log.
        self._log_end('loop_march')
        self.info('\n')

    # logics after exiting main loop (march).
    def _run_postloop(self):
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        # hook: postloop.
        if flag_parallel:
            for sdw in dealer: sdw.cmd.postloop()
        else:
            self.solver.solverobj.postloop()
        self.runhooks('postloop')
    def _run_exhaust(self):
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        # anchor: exhaust.
        if flag_parallel:
            for sdw in dealer: sdw.cmd.exhaust()
        else:
            self.solver.solverobj.exhaust()
    def _run_final(self):
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        # finalize.
        if flag_parallel:
            for sdw in dealer: sdw.cmd.final()
            self.solver.dealer.terminate()
        else:
            self.solver.solverobj.final()

    ############################################################################
    ###
    ### End of block of case execution.
    ###
    ############################################################################

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
