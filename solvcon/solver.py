# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2013 Yung-Yu Chen <yyc@solvcon.net>.
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
Definition of the structure of solvers.
"""

from .solver_core import ALMOST_ZERO
from .solver_legacy import (BaseSolverExedata, BaseSolver, FakeBlockVtk,
    BlockSolver)

class _MethodList(list):
    def register(self, func):
        self.append(func.__name__)
        return func

class MeshSolver(object):
    """
    Base class for all solvers that need :py:class:`solvcon.mesh.Mesh`.
    """

    _interface_init_ = []
    _solution_array_ = []

    def __init__(self, blk, time=0.0, time_increment=0.0, enable_mesg=False,
            **kw):
        """
        A :py:class:`solvcon.block.Block` object must be provided to set the
        :py:attr:`blk` attribute.  The attribute holds the mesh data.
        """
        from .anchor import AnchorList
        from .gendata import Timer
        super(MeshSolver, self).__init__()
        # set mesh and BCs.
        self.blk = blk
        for bc in self.blk.bclist:
            bc.svr = self
        # set time.
        self.time = time
        self.time_increment = time_increment
        # set step.
        self.step_current = 0
        self.step_global = 0
        self.substep_run = 1
        self.substep_current = 0
        # set meta data.
        self.svrn = self.blk.blkn
        self.nsvr = None
        # marching facilities.
        self._mmnames = None
        self.runanchors = AnchorList(self)
        self.marchret = None
        # derived data container.
        self.der = dict()
        # reporting facility.
        self.timer = Timer(vtype=float)
        self.enable_mesg = enable_mesg
        self._mesg = None

    @property
    def bclist(self):
        return self.blk.bclist

    @property
    def grpnames(self):
        return self.blk.grpnames

    @property
    def ngroup(self):
        return len(self.grpnames)

    @property
    def ndim(self):
        return self.blk.ndim

    @property
    def nnode(self):
        return self.blk.nnode

    @property
    def nface(self):
        return self.blk.nface

    @property
    def ncell(self):
        return self.blk.ncell

    @property
    def ngstnode(self):
        return self.blk.ngstnode

    @property
    def ngstface(self):
        return self.blk.ngstface

    @property
    def ngstcell(self):
        return self.blk.ngstcell

    @staticmethod
    def detect_ncore():
        """
        This utility method returns the number of cores on this machine.  It
        only works under Linux.
        """
        f = open('/proc/stat')
        data = f.read()
        f.close()
        cpulist = [line for line in data.split('\n') if
            line.startswith('cpu')]
        cpulist = [line for line in cpulist if line.split()[0] != 'cpu']
        return len(cpulist)

    @property
    def mesg(self):
        """
        Create the message outputing object, which is intended for debugging
        and outputing messages related to the solver.  The outputing device is
        most useful when running distributed solvers.  The created device will
        be attach to self.
        """
        import os
        from .helper import Printer
        if None is self._mesg:
            if self.enable_mesg:
                if self.svrn != None:
                    dfn = 'solvcon.solver%d.log' % self.svrn
                    dprefix = 'SOLVER%d: ' % self.svrn
                else:
                    dfn = 'solvcon.solver.log'
                    dprefix = ''
            else:
                dfn = os.devnull
                dprefix = ''
            self._mesg = Printer(dfn, prefix=dprefix, override=True)
        return self._mesg

    _MMNAMES = None
    @staticmethod
    def new_method_list():
        """
        :return: An object for :py:attr:`_MMNAMES`.
        :rtype: :py:class:`_MethodList`

        In subclasses, implementors can use this utility method to creates an
        instance of :py:class:`_MethodList`, which can be set for
        :py:attr:`_MMNAMES`.
        """
        return _MethodList()

    @property
    def mmnames(self):
        if not self._mmnames:
            self._mmnames = self._MMNAMES[:]
        return self._mmnames

    def provide(self):
        self.runanchors('provide')
    def preloop(self):
        self.runanchors('preloop')
    def postloop(self):
        self.runanchors('postloop')
    def exhaust(self):
        self.runanchors('exhaust')

    def march(self, time, time_increment, steps_run, worker=None):
        """
        :param time: Starting time of this set of marching steps.
        :type time: float
        :param time_increment: Temporal interval :math:`\Delta t` of the time
            step.
        :type time_increment: float
        :param steps_run: The count of time steps to run.
        :type steps_run: int
        :return: Arbitrary return value.
        """
        from time import time as _time
        self.marchret = dict()
        self.step_current = 0
        self.runanchors('premarch')
        while self.step_current < steps_run:
            self.substep_current = 0
            self.runanchors('prefull')
            t0 = _time()
            while self.substep_current < self.substep_run:
                # set up time.
                self.time = time
                self.time_increment = time_increment
                self.runanchors('presub')
                # run marching methods.
                for mmname in self.mmnames:
                    method = getattr(self, mmname)
                    t1 = _time()
                    self.runanchors('pre'+mmname)
                    t2 = _time()
                    method(worker=worker)
                    self.timer.increase(mmname, _time() - t2)
                    self.runanchors('post'+mmname)
                    self.timer.increase(mmname+'_a', _time() - t1)
                # increment time.
                time += self.time_increment/self.substep_run
                self.time = time
                self.time_increment = time_increment
                self.substep_current += 1
                self.runanchors('postsub')
            self.timer.increase('march', _time() - t0)
            self.step_global += 1
            self.step_current += 1
            self.runanchors('postfull')
        self.runanchors('postmarch')
        if worker:
            worker.conn.send(self.marchret)
        return self.marchret

    def init(self, **kw):
        """
        :return: Nothing.

        Check and initialize BCs.
        """
        for arrname in self._solution_array_:
            arr = getattr(self, arrname)
            arr.fill(ALMOST_ZERO)   # prevent initializer forgets to set!
        for bc in self.bclist:
            bc.init(**kw)

    def final(self, **kw):
        pass

    def apply_bc(self):
        """
        :return: Nothing.

        Update the boundary conditions.
        """
        pass

    def call_non_interface_bc(self, name, *args, **kw):
        """
        :param name: Name of the method of BC to call.
        :type name: str
        :return: Nothing.

        Call method of each of non-interface BC objects in my list.
        """
        from .boundcond import interface
        bclist = [bc for bc in self.bclist if not isinstance(bc, interface)]
        for bc in bclist:
            try:
                getattr(bc, name)(*args, **kw)
            except Exception as e:
                e.args = tuple([str(bc), name] + list(e.args))
                raise

    ##################################################
    # parallelization.
    ##################################################
    def remote_setattr(self, name, var):
        """
        Remotely set attribute of worker.
        """
        return setattr(self, name, var)

    def pull(self, arrname, inder=False, worker=None):
        """
        :param arrname: The namd of the array to pull to master.
        :type arrname: str
        :param inder: The data array is derived data array.  Default is False.
        :type inder: bool
        :keyword worker: The worker object for communication.  Default is None.
        :type worker: solvcon.rpc.Worker
        :return: Nothing.

        Pull data array to dealer (rpc) through worker object.
        """
        conn = worker.conn
        if inder:
            arr = self.der[arrname]
        else:
            arr = getattr(self, arrname)
        conn.send(arr)

    def push(self, marr, arrname, start=0, inder=False):
        """
        :param marr: The array passed in.
        :type marr: numpy.ndarray
        :param arrname: The array to pull to master.
        :type arrname: str
        :param start: The starting index of pushing.  Default is 0.
        :type start: int
        :param inder: The data array is derived data array.  Default is False.
        :type inder: bool
        :return: Nothing.

        Push data array received from dealer (rpc) into self.
        """
        if inder:
            arr = self.der[arrname]
        else:
            arr = getattr(self, arrname)
        arr[start:] = marr[start:]

    def pullank(self, ankname, objname, worker=None):
        """
        :param ankname: The name of related anchor.
        :type ankname: str
        :param objname: The object to pull to master.
        :type objname: str
        :keyword worker: The worker object for communication.  Default is None.
        :type worker: solvcon.rpc.Worker
        :return: Nothing.

        Pull data array to dealer (rpc) through worker object.
        """
        conn = worker.conn
        obj = getattr(self.runanchors[ankname], objname)
        conn.send(obj)

    def init_exchange(self, ifacelist):
        from .boundcond import interface
        # grab peer index.
        ibclist = list()
        for pair in ifacelist:
            if pair < 0:
                ibclist.append(pair)
            else:
                assert len(pair) == 2
                assert self.svrn in pair
                ibclist.append(sum(pair)-self.svrn)
        # replace with BC object, sendn and recvn.
        for bc in self.bclist:
            if not isinstance(bc, interface):
                continue
            it = ibclist.index(bc.rblkn)
            sendn, recvn = ifacelist[it]
            ibclist[it] = bc, sendn, recvn
        self.ibclist = ibclist

    def exchangeibc(self, arrname, worker=None):
        from time import sleep
        from threading import Thread
        threads = list()
        for ibc in self.ibclist:
            # check if sleep or not.
            if ibc < 0:
                continue 
            bc, sendn, recvn = ibc
            # determine callable and arguments.
            if self.svrn == sendn:
                target = self.pushibc
                args = arrname, bc, recvn
            elif self.svrn == recvn:
                target = self.pullibc
                args = arrname, bc, sendn
            else:
                raise ValueError, 'bc.rblkn = %d != %d or %d' % (
                    bc.rblkn, sendn, recvn) 
            kwargs = {'worker': worker}
            # call to data transfer.
            target(*args, **kwargs)

    def pushibc(self, arrname, bc, recvn, worker=None):
        """
        :param arrname: The name of the array in the object to exchange.
        :type arrname: str
        :param bc: The interface BC to push.
        :type bc: solvcon.boundcond.interface
        :param recvn: Serial number of the peer to exchange data with.
        :type recvn: int
        :keyword worker: The wrapping worker object for parallel processing.
            Default is None.
        :type worker: solvcon.rpc.Worker

        Push data toward selected interface which connect to blocks with larger
        serial number than myself.
        """
        from numpy import empty
        conn = worker.pconns[bc.rblkn]
        ngstcell = self.ngstcell
        arr = getattr(self, arrname)
        # ask the receiver for data.
        shape = list(arr.shape)
        shape[0] = bc.rclp.shape[0]
        rarr = empty(shape, dtype=arr.dtype)
        conn.recvarr(rarr)  # comm.
        slct = bc.rclp[:,0] + ngstcell
        arr[slct] = rarr[:]
        # provide the receiver with data.
        slct = bc.rclp[:,2] + ngstcell
        conn.sendarr(arr[slct]) # comm.

    def pullibc(self, arrname, bc, sendn, worker=None):
        """
        :param arrname: The name of the array in the object to exchange.
        :type arrname: str
        :param bc: The interface BC to pull.
        :type bc: solvcon.boundcond.interface
        :param sendn: Serial number of the peer to exchange data with.
        :type sendn: int
        :keyword worker: The wrapping worker object for parallel processing.
            Default is None.
        :type worker: solvcon.rpc.Worker

        Pull data from the interface determined by the serial of peer.
        """
        from numpy import empty
        conn = worker.pconns[bc.rblkn]
        ngstcell = self.ngstcell
        arr = getattr(self, arrname)
        # provide sender the data.
        slct = bc.rclp[:,2] + ngstcell
        conn.sendarr(arr[slct]) # comm.
        # ask data from sender.
        shape = list(arr.shape)
        shape[0] = bc.rclp.shape[0]
        rarr = empty(shape, dtype=arr.dtype)
        conn.recvarr(rarr)  # comm.
        slct = bc.rclp[:,0] + ngstcell
        arr[slct] = rarr[:]
