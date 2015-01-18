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
Simulation cases.
"""


import os
import sys
import signal
import cPickle as pickle
import time
import gzip

from . import hook
from . import anchor
from . import helper
from . import domain
from . import rpc
from . import conf
from . import boundcond
from .io import gmsh as iogmsh
from .io import genesis as iogenesis
from .io import gambit as iogambit
from .io import block as ioblock
from .io import domain as iodomain

from . import case_core


# import legacy.
from .case_core import arrangements, CaseInfo
from .case_legacy import BaseCase, BlockCase


class MeshHookList(list):
    """
    Hook container and invoker.

    :ivar cse: case object.
    :type cse: solvcon.case.BaseCase

    >>> MeshHookList()
    Traceback (most recent call last):
        ...
    TypeError: __init__() takes at least 2 arguments (1 given)
    >>> # Give the constructor a MeshCase.
    >>> from solvcon.testing import create_trivial_2d_blk
    >>> cse = MeshCase() # No arguments because of demonstration.
    >>> hook_list = MeshHookList(cse)

    The input *cse* will be kept in the list:

    >>> cse is hook_list.cse
    True
    """

    def __init__(self, cse, *args, **kw):
        """
        :py:class:`MeshHookList` requires an argument to :py:class:`MeshCase`:
        """
        self.cse = cse
        super(MeshHookList, self).__init__(*args, **kw)

    def append(self, obj, **kw):
        """
        The input :py:class:`~.hook.MeshHook` *obj* will be appended into the
        list.  The *obj* argument is usually a :py:class:`~.hook.MeshHook` type
        or a :py:class:`~.anchor.MeshAnchor` type, instead of an instance of
        them.  The method will automatically create the necessary Hook object
        when detect acceptable type object passed as the first argument.

        See :py:meth:`sanitize` for conditions of the input *obj* and *kw*.

        >>> import solvcon as sc
        >>> from solvcon.testing import create_trivial_2d_blk
        >>> cse = MeshCase() # No arguments because of demonstration.
        >>> hook_list = MeshHookList(cse)
        >>> len(hook_list)
        0
        >>> # Create two hooks of the same type and content.
        >>> hook_list.append(sc.MeshHook, dummy="name")
        >>> hook_list.append(sc.MeshHook, dummy="name")
        >>> hook_list[0].kws['dummy'], hook_list[1].kws['dummy']
        ('name', 'name')
        >>> # The two hooks aren't identical.
        >>> hook_list[0] is hook_list[1]
        False
        """
        obj = self.sanitize(obj, **kw)
        return super(MeshHookList, self).append(obj)

    def replace(self, obj, **kw):
        """
        Replace the object if one of the same type already existed, otherwise,
        call :py:meth:`append`.

        See :py:meth:`sanitize` for conditions of the input *obj* and *kw*.

        :return: Nothing.

        >>> import solvcon as sc
        >>> from solvcon.testing import create_trivial_2d_blk
        >>> cse = MeshCase() # No arguments because of demonstration.
        >>> hook_list = MeshHookList(cse)
        >>> len(hook_list)
        0
        >>> # Create a hook.
        >>> hook_list.replace(sc.MeshHook, dummy="name1")
        >>> hook_list[0].kws['dummy']
        'name1'
        >>> # Create the second hook to replace the first one.
        >>> hook_list.replace(sc.MeshHook, dummy="name2")
        >>> hook_list[0].kws['dummy'] # Got replaced.
        'name2'
        >>> len(hook_list) # Still only one hook in the list.
        1
        """
        # Check if there's an existed object of the same type.
        index = None
        for it, existed in enumerate(self):
            good_types = [obj]
            ankcls = getattr(existed, 'ankcls', None)
            if ankcls:
                good_types.append(ankcls)
            if issubclass(type(existed), tuple(good_types)):
                index = it
                # If there area multiple matches, ignore them.
                break
        # Append if no match.
        if None is index:
            return self.append(obj, **kw)
        # Replace if there's a match.
        else:
            obj = self.sanitize(obj, **kw)
            self[index] = obj

    def sanitize(self, obj, **kw):
        """
        All the keywords go to the creation of the :py:class:`~.hook.MeshHook`
        object if the first argument is a type.  If the first argument is an
        instantiated :py:class:`~.hook.MeshHook` object, the method accepts no
        keywords.

        :param obj: the hook object to be appended.
        :type obj: solvcon.hook.MeshHook or solvcon.anchor.MeshAnchor
        :return: The original or the newly instantiated
                 :py:class:`~.hook.MeshHook` object.

        The input *obj* is either a :py:class:`~.hook.MeshHook` or a
        :py:class:`~.anchor.MeshAnchor` type.  Other input results into
        TypeError.

        >>> from solvcon.testing import create_trivial_2d_blk
        >>> cse = MeshCase() # No arguments because of demonstration.
        >>> hook_list = MeshHookList(cse)
        >>> hook_list.sanitize(tuple)
        Traceback (most recent call last):
            ...
        TypeError: not solvcon.MeshHook or solvcon.MeshAnchor
        >>> import solvcon as sc
        >>> type(hook_list.sanitize(sc.MeshHook)).__name__
        'MeshHook'

        When :py:class:`~.anchor.MeshAnchor` is given, the constructed
        object will be a :py:class:`~.hook.MeshHook` used as a dummy.

        >>> type(hook_list.sanitize(sc.MeshAnchor)).__name__
        'MeshHook'

        If the input *obj* is an instance, no *kw* can be given.

        >>> hook_list.sanitize(sc.MeshHook(cse), dummy=None)
        Traceback (most recent call last):
            ...
        AssertionError
        >>> type(hook_list.sanitize(sc.MeshHook(cse))).__name__
        'MeshHook'
        """
        if isinstance(obj, type):
            if issubclass(obj, hook.MeshHook):
                hookcls = obj
            elif issubclass(obj, anchor.MeshAnchor):
                kw['ankcls'] = obj
                hookcls = hook.MeshHook # Use a dummy MeshHook.
            else:
                raise TypeError("not solvcon.MeshHook or solvcon.MeshAnchor")
            obj = hookcls(self.cse, **kw)
        else: # Got an instance.
            assert len(kw) == 0
        return obj

    def __call__(self, method):
        """
        Invoke the specified method for each hook object.

        :param method: name of the method to run.
        :type method: str
        """
        runhooks = self
        if method == 'postloop':
            runhooks = reversed(runhooks)
        for hook in runhooks:
            getattr(hook, method)()

    def drop_anchor(self, svr):
        for hok in self:
            hok.drop_anchor(svr)


class MeshCase(case_core.CaseInfo):
    """
    Base class for simulation cases based on :py:class:`solvcon.mesh.Mesh`.
    """

    defdict = {
        # execution related.
        'execution.fpdtype': 'float64',
        'execution.npart': None,    # number of decomposed blocks.
        'execution.stop': False,
        'execution.time': 0.0,
        'execution.time_increment': 0.0,
        'execution.step_init': 0,
        'execution.step_current': None,
        'execution.steps_run': None,
        'execution.steps_stride': 1,
        'execution.marchret': None,
        'execution.var': dict,  # for Calculator hooks.
        'execution.varstep': None,  # the step for which var and dvar are valid.
        # io related.
        'io.mesher': None,
        'io.meshfn': None,
        'io.domain.with_arrs': True,
        'io.domain.with_whole': True,
        'io.domain.wholefn': None,
        'io.domain.splitfns': None,
        'io.abspath': False,    # flag to use abspath or not.
        'io.rootdir': None,
        'io.basedir': None,
        'io.basefn': None,
        'io.empty_jobdir': False,
        'io.solver_output': False,
        # conditions.
        'condition.bcmap': None,
        'condition.bcmod': None,
        'condition.mtrllist': list,
        # solver.
        'solver.debug': False,
        'solver.use_incenter': False,
        'solver.domaintype': None,
        'solver.domainobj': None,
        'solver.solvertype': None,
        'solver.solverobj': None,
        'solver.dealer': None,
        # logging.
        'log.time': dict,
    }

    def __init__(self, **kw):
        # populate value from keywords.
        initpairs = list()
        for cinfok in self.defdict.keys():
            lkey = cinfok.split('.')[-1]
            initpairs.append((cinfok, kw.pop(lkey, None)))
        # initialize with the left keywords.
        super(MeshCase, self).__init__(**kw)
        # populate value from keywords.
        for cinfok, val in initpairs:
            if val is not None:
                self._set_through(cinfok, val)
        # create runhooks.
        self.runhooks = MeshHookList(self)
        # expand basedir.
        if self.io.abspath:
            self.io.basedir = os.path.abspath(self.io.basedir)
        if self.io.basedir is not None and not os.path.exists(self.io.basedir):
            os.makedirs(self.io.basedir)
        # message logger.
        self.info = helper.Information()

    def _log_start(self, action, msg='', postmsg=' ... '):
        """
        :param action: Action key.
        :type action: str
        :keyword msg: Trailing message for the action key.
        :type msg: str
        :return: Nothing.

        Print to user and record start time for certain action.
        """
        info = self.info
        tarr = [0,0,0]
        tarr[0] = time.time()
        self.log.time[action] = tarr
        # header.
        prefix = info.prefix * (info.width-info.level*info.nchar)
        info(prefix, travel=1)
        # content.
        info('\nStart %s%s%s' % (action, msg, postmsg))
        prefix = info.prefix * (info.width-info.level*info.nchar)
        info('\n' + prefix + '\n')

    def _log_end(self, action, msg='', postmsg=' . '):
        """
        :param action: Action key.
        :type action: str
        :keyword msg: Supplemental message.
        :type msg: str
        :return: Nothing

        Print to user and record end time for certain action.
        """
        info = self.info
        tarr = self.log.time.setdefault(action, [0,0,0])
        tarr[1] = time.time()
        tarr[2] = tarr[1] - tarr[0]
        # footer.
        prefix = info.prefix * (info.width-info.level*info.nchar)
        info(prefix + '\nEnd %s%s%sElapsed time (sec) = %g' % (
            action, msg, postmsg, tarr[2]))
        # up a level.
        prefix = info.prefix * (info.width-(info.level-1)*info.nchar)
        info('\n' + prefix + '\n', travel=-1)

    @property
    def is_parallel(self):
        """
        Determine if the self object should do parallel or not.  0 means
        sequential; 1 means local parallel; 2 means networked parallel.
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

    ############################################################################
    ###
    ### Begin of block of case initialization logics.
    ###
    ############################################################################

    def init(self, level=0):
        """
        :keyword level: Run level; higher level does less work.
        :type level: :py:class:`int`
        :return: Nothing

        Load a block and initialize the solver from the geometry information in
        the block and conditions in the self case.  If parallel run is
        specified (through domaintype), split the domain and perform
        corresponding tasks.
        
        For a :py:class:`MeshCase` to be initialized, some information needs to
        be supplied to the constructor:

        >>> cse = MeshCase()
        >>> cse.info.muted = True
        >>> cse.init()
        Traceback (most recent call last):
            ...
        TypeError: coercing to Unicode: need string or buffer, NoneType found

        #. Mesh information.  We can provide *meshfn* that specifying the path
           of a valid mesh file, or provide *mesher*, which is a function that
           generates the mesh and returns the :py:class:`solvcon.block.Block`
           object, like the following code:

            >>> from solvcon.testing import create_trivial_2d_blk
            >>> blk = create_trivial_2d_blk()
            >>> cse = MeshCase(mesher=lambda *arg: blk)
            >>> cse.info.muted = True
            >>> cse.init() # doctest: +NORMALIZE_WHITESPACE
            Traceback (most recent call last):
                ...
            TypeError: isinstance() arg 2 must be a class, type, or tuple of
            classes and types

        #. Type of the spatial domain.  This information is used for detemining
           sequential or parallel execution, and performing related operations:

            >>> cse = MeshCase(mesher=lambda *arg: blk, domaintype=domain.Domain)
            >>> cse.info.muted = True
            >>> cse.init()
            Traceback (most recent call last):
                ...
            TypeError: 'NoneType' object is not callable

        #. The type of solver.  It is used to specify the underlying numerical
           method:

            >>> from solvcon.solver import MeshSolver
            >>> cse = MeshCase(mesher=lambda *arg: blk,
            ...                domaintype=domain.Domain,
            ...                solvertype=MeshSolver)
            >>> cse.info.muted = True
            >>> cse.init()
            Traceback (most recent call last):
                ...
            TypeError: cannot concatenate 'str' and 'NoneType' objects

        #. The base name.  It is used to name its output files:

            >>> cse = MeshCase(
            ...     mesher=lambda *arg: blk, domaintype=domain.Domain,
            ...     solvertype=MeshSolver, basefn='meshcase')
            >>> cse.info.muted = True
            >>> cse.init()
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
        :return: a block object.
        :rtype: solvcon.block.Block

        Return a block for init.
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

    @property
    def blk(self):
        """
        >>> from solvcon.testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> from solvcon.solver import MeshSolver
        >>> cse = MeshCase(
        ...     mesher=lambda *arg: blk, domaintype=domain.Domain,
        ...     solvertype=MeshSolver, basefn='meshcase')
        >>> cse.info.muted = True

        This property is only valid after :py:meth:`init`.

        >>> str(cse.blk)
        Traceback (most recent call last):
            ...
        KeyError: 'blk'
        >>> cse.init()
        >>> str(cse.blk)
        '[Block (2D/centroid): 4 nodes, 6 faces (3 BC), 3 cells]'
        """
        return self.solver.domainobj.blk

    def make_solver_keywords(self):
        """
        :return: keywords
        :rtype: dict

        Return keywords to initialize solvers.
        """
        return dict(
            enable_mesg=self.io.solver_output,
            debug=self.solver.debug,
        )

    # solver object initialization/loading.
    def _local_init_solver(self):
        """
        @return: nothing
        """
        svr = self.solver.solvertype(
            self.solver.domainobj.blk, **self.make_solver_keywords())
        self.runhooks.drop_anchor(svr)
        svr.init()
        self.solver.solverobj = svr
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
                dealer[iblk].remote_setattr('muscle', svr)
            self.info('done.\n')
        self.info('Bind/Init ... ')
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
            if self.solver.debug:
                debug = iblk, self.io.basedir
            else:
                debug = None
            dealer.hire(rpc.Worker(
                None,
                profiler_data=self._get_profiler_data(iblk),
                debug=debug))
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
        if conf.env.command != None and \
           conf.env.command.opargs[0].compress_nodelist:
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
        :keyword level: Run level; higher level does less work.
        :type level: :py:class:`int`
        :return: Nothing

        Temporal loop for the incorporated solver.  A simple example:

        >>> from .testing import create_trivial_2d_blk
        >>> from .solver import MeshSolver
        >>> blk = create_trivial_2d_blk()
        >>> cse = MeshCase(basefn='meshcase', mesher=lambda *arg: blk,
        ...                domaintype=domain.Domain, solvertype=MeshSolver)
        >>> cse.info.muted = True
        >>> cse.init()
        >>> cse.run()
        """
        self._log_start('run')
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
        self._log_start('run_provide')
        if flag_parallel:
            for sdw in dealer: sdw.cmd.provide()
        else:
            self.solver.solverobj.provide()
        self._log_end('run_provide')
    def _run_preloop(self):
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        # hook: preloop.
        self._log_start('run_preloop')
        self.runhooks('preloop')
        if flag_parallel:
            for sdw in dealer: sdw.cmd.preloop()
            for arrname in self.solver.solvertype._solution_array_:
                for sdw in dealer:
                    sdw.cmd.exchangeibc(arrname, with_worker=True)
            for sdw in dealer: sdw.cmd.apply_bc()
        else:
            self.solver.solverobj.preloop()
            self.solver.solverobj.apply_bc()
        self._log_end('run_preloop')

    def _run_march(self):
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        self.log.time['solver_march'] = 0.0
        self.info('\n')
        self._log_start('run_march')
        while self.execution.step_current < self.execution.steps_run:
            if self.execution.stop: break
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
        self._log_end('run_march')
        self.info('\n')

    # logics after exiting main loop (march).
    def _run_postloop(self):
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        # hook: postloop.
        self._log_start('run_postloop')
        if flag_parallel:
            for sdw in dealer: sdw.cmd.postloop()
        else:
            self.solver.solverobj.postloop()
        self.runhooks('postloop')
        self._log_end('run_postloop')
    def _run_exhaust(self):
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        # anchor: exhaust.
        self._log_start('run_exhaust')
        if flag_parallel:
            for sdw in dealer: sdw.cmd.exhaust()
        else:
            self.solver.solverobj.exhaust()
        self._log_end('run_exhaust')

    def _run_final(self):
        dealer = self.solver.dealer
        flag_parallel = self.is_parallel
        # finalize.
        self._log_start('run_final')
        if flag_parallel:
            for sdw in dealer: sdw.cmd.final()
            dealer.terminate()
        else:
            self.solver.solverobj.final()
        self._log_end('run_final')

    ############################################################################
    ###
    ### End of block of case execution.
    ###
    ############################################################################

    def cleanup(self, signum=None, frame=None):
        """
        :keyword signum: Signal number.
        :keyword frame: Current stack frame.

        A signal handler for cleaning up the simulation case on termination or
        when errors occur.  This method can be overridden in subclasses.  The
        base implementation is trivial, but usually doesn't need to be
        overridden.
        
        An example to connect this method to a signal:

        >>> from .testing import create_trivial_2d_blk
        >>> from .solver import MeshSolver
        >>> blk = create_trivial_2d_blk()
        >>> cse = MeshCase(basefn='meshcase', mesher=lambda *arg: blk,
        ...                domaintype=domain.Domain, solvertype=MeshSolver)
        >>> cse.info.muted = True
        >>> signal.signal(signal.SIGTERM, cse.cleanup)
        0

        An example to call this method explicitly:

        >>> cse.init()
        >>> cse.run()
        >>> cse.cleanup()
        """
        if signum == signal.SIGINT:
            raise KeyboardInterrupt

    @classmethod
    def register_arrangement(cls, func, casename=None):
        """
        :return: Simulation function.
        :rtype: callable

        This class method is a decorator that creates a closure (internal
        function) that turns the decorated function to an arrangement, and
        registers the arrangement into the module-level registry and the
        class-level registry.  The decorator function should return a
        :py:class:`MeshCase` object ``cse``, and the closure performs a
        simulation run by the following code:

        .. code-block:: python

            try:
                signal.signal(signal.SIGTERM, cse.cleanup)
                signal.signal(signal.SIGINT, cse.cleanup)
                cse.init(level=runlevel)
                cse.run(level=runlevel)
                cse.cleanup()
            except:
                cse.cleanup()
                raise
        
        The usage of this decorator can be exemplified by the following code,
        which creates four arrangements (although the first three are
        erroneous):

        >>> @MeshCase.register_arrangement
        ... def arg1():
        ...     return None
        >>> @MeshCase.register_arrangement
        ... def arg2(wrongname):
        ...     return None
        >>> @MeshCase.register_arrangement
        ... def arg3(casename):
        ...     return None
        >>> @MeshCase.register_arrangement
        ... def arg4(casename):
        ...     from .testing import create_trivial_2d_blk
        ...     from .solver import MeshSolver
        ...     blk = create_trivial_2d_blk()
        ...     cse = MeshCase(basefn='meshcase', mesher=lambda *arg: blk,
        ...                    domaintype=domain.Domain, solvertype=MeshSolver)
        ...     cse.info.muted = True
        ...     return cse

        The created arrangements are collected to a class attribute
        :py:attr:`arrangements`, i.e., the class-level registry:

        >>> sorted(MeshCase.arrangements.keys())
        ['arg1', 'arg2', 'arg3', 'arg4']

        The arrangements in the class attribute :py:attr:`arrangements` are
        also put into a module-level attribute
        :py:data:`solvcon.case.arrangements`:

        >>> arrangements == MeshCase.arrangements
        True

        The first example arrangement is a bad one, because it allows no
        argument:

        >>> arrangements.arg1()
        Traceback (most recent call last):
          ...
        TypeError: arg1() takes no arguments (1 given)

        The second example arrangement is still a bad one, because although it
        has an argument, the name of the argument is incorrect:

        >>> arrangements.arg2()
        Traceback (most recent call last):
          ...
        TypeError: arg2() got an unexpected keyword argument 'casename'

        The third example arrangement is a bad one for another reason.  It
        doesn't return a :py:class:`MeshCase`:

        >>> arrangements.arg3()
        Traceback (most recent call last):
          ...
        AttributeError: 'NoneType' object has no attribute 'cleanup'

        The fourth example arrangement is finally good:

        >>> arrangements.arg4()
        """
        if casename is None: casename = func.__name__
        def simu(*args, **kw):
            kw.setdefault('casename', casename)
            runlevel = kw.pop('runlevel', 0)
            # obtain the case object.
            if runlevel == 1:
                cse = pickle.load(open(cls.CSEFN_DEFAULT, 'rb'))
            else:
                cse = func(*args, **kw)
            # run.
            try:
                signal.signal(signal.SIGTERM, cse.cleanup)
                signal.signal(signal.SIGINT, cse.cleanup)
                cse.init(level=runlevel)
                cse.run(level=runlevel)
                cse.cleanup()
            except:
                cse.cleanup()
                raise
        # register self to simulation registries.
        cls.arrangements[casename] = arrangements[casename] = simu
        # return the simulation function.
        return simu

    def defer(self, delayed, replace=False, **kw):
        """
        Insert (append or replace) hooks.

        :param delayed: The delayed construct.
        :type delayed: :py:class:`solvcon.MeshHook` or
                       :py:class:`solvcon.MeshAnchor`.
        :keyword replace: True if existing object should be replaced.
        :type replace: bool
        :return: Nothing.

        >>> import solvcon as sc
        >>> from solvcon.testing import create_trivial_2d_blk
        >>> cse = MeshCase() # No arguments because of demonstration.
        >>> len(cse.runhooks)
        0
        >>> # Insert a hook.
        >>> cse.defer(sc.MeshHook, dummy='name1')
        >>> cse.runhooks[0].kws['dummy']
        'name1'
        >>> # Insert the second hook to replace the first one.
        >>> cse.defer(sc.MeshHook, replace=True, dummy='name2')
        >>> cse.runhooks[0].kws['dummy'] # Got replaced.
        'name2'
        >>> len(cse.runhooks) # Still only one hook in the list.
        1
        >>> # Insert the third hook without replace.
        >>> cse.defer(sc.MeshHook, dummy='name3')
        >>> cse.runhooks[1].kws['dummy'] # Got replaced.
        'name3'
        """
        if replace: # Intentional duplication.
            self.runhooks.replace(delayed, **kw)
        else: # Replace the original one.
            self.runhooks.append(delayed, **kw)

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
