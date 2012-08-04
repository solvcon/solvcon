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

    from .block import Block
    FCMND = Block.FCMND
    CLMND = Block.CLMND
    CLMFC = Block.CLMFC
    del Block

    def __init__(self, blk, *args, **kw):
        from numpy import empty
        from .conf import env
        from .anchor import AnchorList
        from .gendata import Timer
        super(MeshSolver, self).__init__()
        kw.pop('fpdtype', None)
        self.ibcthread = kw.pop('ibcthread', False)
        self.enable_mesg = kw.pop('enable_mesg', False)
        self.mesg = None
        self.enable_tpool = kw.pop('enable_tpool', True)
        self.ncore = kw.pop('ncore', -1)
        self.neq = kw.pop('neq')
        self.time = kw.pop('time', 0.0)
        self.time_increment = kw.pop('time_increment', 0.0)
        self.step_global = 0
        self.step_current = 0
        self.substep_current = 0
        self.substep_run = kw.pop('substep_run', 2)
        # anchor list.
        self.runanchors = AnchorList(self)
        # marching methods name.
        self.mmnames = self.MMNAMES[:]
        self.marchret = None
        # timer.
        self.timer = Timer(vtype=float)
        self.ticker = dict()
        # derived data.
        self.der = dict()
        # block data.
        self.all_simplex = blk.check_simplex()
        self.use_incenter = blk.use_incenter
        # index.
        self.svrn = blk.blkn
        self.nsvr = None
        # group.
        self.grpnames = blk.grpnames
        self.ngroup = len(self.grpnames)
        # BCs.
        self.bclist = blk.bclist
        for bc in self.bclist:
            bc.blk = None
            bc.svr = self
        self.ibclist = None
        # mesh shape.
        self.ndim = blk.ndim
        self.nnode = blk.nnode
        self.nface = blk.nface
        self.ncell = blk.ncell
        self.nbound = blk.nbound
        self.ngstnode = blk.ngstnode
        self.ngstface = blk.ngstface
        self.ngstcell = blk.ngstcell
        # meta array.
        self.fctpn = blk.shfctpn
        self.cltpn = blk.shcltpn
        self.clgrp = blk.shclgrp
        ## connectivity.
        self.clnds = blk.shclnds
        self.clfcs = blk.shclfcs
        self.fcnds = blk.shfcnds
        self.fccls = blk.shfccls
        ## geometry.
        self.ndcrd = blk.shndcrd
        self.fccnd = blk.shfccnd
        self.fcara = blk.shfcara
        self.fcnml = blk.shfcnml
        self.clcnd = blk.shclcnd
        self.clvol = blk.shclvol
        # in situ visualization by VTK.
        self._ust = None

    @property
    def fpdtype(self):
        from numpy import float64
        return float64
    @property
    def fpdtypestr(self):
        return 'float64'

    @staticmethod
    def detect_ncore():
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

    def init(self, **kw):
        pass
    def final(self):
        pass

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
        Set the time for self and structures.  TODO: should be property setter.
        """
        self.time = time
        self.time_increment = time_increment

    def march(self, time, time_increment, steps_run, worker=None):
        """
        Default marcher for the solver object.

        @param time: starting time of marching.
        @type time: float
        @param time_increment: temporal interval for the time step.
        @type time_increment: float
        @param steps_run: the count of time steps to run.
        @type steps_run: int
        @return: arbitrary return value.
        @rtype: float
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

    @property
    def ust(self):
        from .visual_vtk import make_ust_from_blk
        _ust = self._ust
        if _ust is None:
            fbk = FakeBlockVtk(self)
            _ust = make_ust_from_blk(fbk)
        self._ust = _ust
        return _ust

    def bind(self):
        """
        Bind all the boundary condition objects.

        @note: BC must be bound AFTER solver "pointers".  Overridders to the
            method should firstly bind all pointers, secondly super binder, and 
            then methods/subroutines.
        """
        super(BlockSolver, self).bind()
        # boundary conditions.
        for bc in self.bclist:
            bc.bind()

    def unbind(self):
        """
        Unbind all the boundary condition objects.
        """
        super(BlockSolver, self).unbind()
        for bc in self.bclist:
            bc.unbind()

    @property
    def is_bound(self):
        """
        Check boundness for solver as well as BC objects.
        """
        if not super(BlockSolver, self).is_bound:
            return False
        else:
            for bc in self.bclist:
                if not bc.is_bound:
                    return False
            return True

    @property
    def is_unbound(self):
        """
        Check unboundness for solver as well as BC objects.
        """
        if not super(BlockSolver, self).is_unbound:
            return False
        else:
            for bc in self.bclist:
                if not bc.is_unbound:
                    return False
                return True

    def init(self, **kw):
        """
        Check and initialize BCs.
        """
        for arrname in self._solution_array_:
            arr = getattr(self, arrname)
            arr.fill(ALMOST_ZERO)   # prevent initializer forgets to set!
        for bc in self.bclist:
            bc.init(**kw)
        super(BlockSolver, self).init(**kw)

    def boundcond(self):
        """
        Update the boundary conditions.

        @return: nothing.
        """
        pass

    def call_non_interface_bc(self, name, *args, **kw):
        """
        Call method of each of non-interface BC objects in my list.

        @param name: name of the method of BC to call.
        @type name: str
        @return: nothing
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
        Pull data array to dealer (rpc) through worker object.

        @param arrname: the array to pull to master.
        @type arrname: str
        @param inder: the data array is derived data array.
        @type inder: bool
        @keyword worker: the worker object for communication.
        @type worker: solvcon.rpc.Worker
        @return: nothing.
        """
        conn = worker.conn
        if inder:
            arr = self.der[arrname]
        else:
            arr = getattr(self, arrname)
        conn.send(arr)

    def push(self, marr, arrname, start=0, inder=False):
        """
        Push data array received from dealer (rpc) into self.

        @param marr: the array passed in.
        @type marr: numpy.ndarray
        @param arrname: the array to pull to master.
        @type arrname: str
        @param start: the starting index of pushing.
        @type start: int
        @param inder: the data array is derived data array.
        @type inder: bool
        @return: nothing.
        """
        if inder:
            arr = self.der[arrname]
        else:
            arr = getattr(self, arrname)
        arr[start:] = marr[start:]

    def pullank(self, ankname, objname, worker=None):
        """
        Pull data array to dealer (rpc) through worker object.

        @param ankname: the name of related anchor.
        @type ankname: str
        @param objname: the object to pull to master.
        @type objname: str
        @keyword worker: the worker object for communication.
        @type worker: solvcon.rpc.Worker
        @return: nothing.
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
            if self.ibcthread:
                threads.append(Thread(
                    target=target,
                    args=args,
                    kwargs=kwargs,
                ))
                threads[-1].start()
            else:
                target(*args, **kwargs)
        if self.ibcthread:
            for thread in threads:
                thread.join()

    def pushibc(self, arrname, bc, recvn, worker=None):
        """
        Push data toward selected interface which connect to blocks with larger
        serial number than myself.

        @param arrname: name of the array in the object to exchange.
        @type arrname: str
        @param bc: the interface BC to push.
        @type bc: solvcon.boundcond.interface
        @param recvn: serial number of the peer to exchange data with.
        @type recvn: int
        @keyword worker: the wrapping worker object for parallel processing.
        @type worker: solvcon.rpc.Worker
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
        Pull data from the interface determined by the serial of peer.

        @param arrname: name of the array in the object to exchange.
        @type arrname: str
        @param bc: the interface BC to pull.
        @type bc: solvcon.boundcond.interface
        @param sendn: serial number of the peer to exchange data with.
        @type sendn: int
        @keyword worker: the wrapping worker object for parallel processing.
        @type worker: solvcon.rpc.Worker
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
