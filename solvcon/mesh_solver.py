# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2012 Yung-Yu Chen <yyc@solvcon.net>.
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
Solvers that base on :py:class:`.mesh.Mesh`.
"""

class MeshSolver(object):
    """
    Base class for all solvers that base on :py:class:`.mesh.Mesh`.
    """
    MESG_FILENAME_DEFAULT = 'solvcon.solver.log'
    MMNAMES = []

    _interface_init_ = []
    _solution_array_ = []

    def __init__(self, blk, *args, **kw):
        from .anchor import AnchorList
        from .gendata import Timer
        super(MeshSolver, self).__init__()
        # set reporting facility.
        self.enable_mesg = kw.pop('enable_mesg', False)
        self.mesg = None
        # set mesh.
        self.blk = blk
        # set meta data.
        self.svrn = blk.blkn
        self.nsvr = None
        # set time.
        self.time = kw.pop('time', 0.0)
        self.time_increment = kw.pop('time_increment', 0.0)
        # set step.
        self.step_global = 0
        self.step_current = 0
        self.substep_current = 0
        self.substep_run = kw.pop('substep_run', 2)
        # BCs.
        for bc in self.blk.bclist:
            bc.svr = self
        self.ibclist = None
        # anchor list.
        self.runanchors = AnchorList(self)
        # marching methods name.
        self.mmnames = self.MMNAMES[:]
        self.marchret = None
        # timer.
        self.timer = Timer(vtype=float)
        # derived data.
        self.der = dict()

    @staticmethod
    def detect_ncore():
        """
        :return: Number of cores.
        :rtype: int

        Only works under Linux.
        """
        f = open('/proc/stat')
        data = f.read()
        f.close()
        cpulist = [line for line in data.split('\n') if
            line.startswith('cpu')]
        cpulist = [line for line in cpulist if line.split()[0] != 'cpu']
        return len(cpulist)

    def __create_mesg(self, force=False):
        """
        :keyword force: flag to force the creation.  Default is False,
        :type force: bool
        :return: nothing

        Create the message outputing device, which is intended for debugging
        and outputing messages related to the solver.  The outputing device is
        most useful when running distributed solvers.  The created device will
        be attach to self.
        """
        import os
        from .helper import Printer
        if force: self.enable_mesg = True
        if self.enable_mesg:
            if self.svrn != None:
                main, ext = os.path.splitext(self.MESG_FILENAME_DEFAULT)
                tmpl = main + '%d' + ext
                dfn = tmpl % self.svrn
                dprefix = 'SOLVER%d: ' % self.svrn
            else:
                dfn = self.MESG_FILENAME_DEFAULT
                dprefix = ''
        else:
            dfn = os.devnull
            dprefix = ''
        self.mesg = Printer(dfn, prefix=dprefix, override=True)

    def provide(self):
        self.runanchors('provide')
    def preloop(self):
        self.runanchors('preloop')
    def postloop(self):
        self.runanchors('postloop')
    def exhaust(self):
        self.runanchors('exhaust')

    def _set_time(self, time, time_increment):
        """
        :param time: Starting time of marching.
        :type time: float
        :param time_increment: Temporal interval :math:`\Delta t` for the time
            step.
        :type time_increment: float

        Set the time for self and structures.
        """
        self.time = time
        self.time_increment = time_increment

    def march(self, time, time_increment, steps_run, worker=None):
        """
        :param time: Starting time of marching.
        :type time: float
        :param time_increment: Temporal interval :math:`\Delta t` for the time
            step.
        :type time_increment: float
        :param steps_run: The count of time steps to run.
        :type steps_run: int
        :return: arbitrary return value.
        :rtype: float

        Default marcher for the solver object.
        """
        from time import time as _time
        self.marchret = dict()
        self.step_current = 0
        self.runanchors('premarch')
        while self.step_current < steps_run:
            self.substep_current = 0
            self.runanchors('prefull')
            t0= _time()
            while self.substep_current < self.substep_run:
                self._set_time(time, time_increment)
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
                time += time_increment/self.substep_run
                self._set_time(time, time_increment)
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
        Check and initialize BCs.
        """
        for arrname in self._solution_array_:
            arr = getattr(self, arrname)
            arr.fill(ALMOST_ZERO)   # prevent initializer forgets to set!
        for bc in self.bclist:
            bc.init(**kw)

    def boundcond(self):
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

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
