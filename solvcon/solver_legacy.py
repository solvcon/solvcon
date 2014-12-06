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
Definition of the structure of solvers.
"""

from ctypes import Structure
from .gendata import TypeWithBinder
from .solver_core import ALMOST_ZERO

class BaseSolverExedata(Structure):
    """
    Execution information for BaseSolver.
    """
    from ctypes import c_int, c_double
    _fields_ = [
        ('ncore', c_int), ('neq', c_int),
        ('time', c_double), ('time_increment', c_double),
    ]
    del c_int, c_double 
    def __init__(self, *args, **kw):
        svr = kw.pop('svr', None)
        super(BaseSolverExedata, self).__init__(*args, **kw)
        if svr == None:
            return
        for key in ('ncore', 'neq', 'time', 'time_increment'):
            setattr(self, key, getattr(svr, key))

class BaseSolver(object):
    """
    Generic solver definition.  It is an abstract class and should not be used
    to any concrete simulation case.  The concrete solver sub-classes should
    override the empty init and final methods for initialization and 
    finalization, respectively.

    @cvar _clib_solve: the external dll (accessible through ctypes) which do
        the cell loop.  Subclass should override it.
    @ctype _clib_solve: ctypes.CDLL
    @cvar _exedatatype_: the C struct definition in ctypes.Structure.
    @ctype _exedatatype_: ctypes.Structure

    @cvar MESG_FILENAME_DEFAULT = the default file name for serial solver
        object.

    @ivar _fpdtype: dtype for the floating point data in the block instance.
    @itype _fpdtype: numpy.dtype

    @ivar enable_mesg: flag if mesg device should be enabled.
    @itype enable_mesg: bool
    @ivar mesg: message printer attached to a certain solver object; designed
        and mainly used for parallel solver.
    @itype mesg: solvcon.helper.Printer

    @ivar runanchors: the list for the anchor objects to be run.
    @itype runanchors: solvcon.anchor.AnchorList

    @ivar exd: execution information for the solver.
    @itype exd: ctypes.Structure
    @ivar enable_tpool: flag to enable thread pool on binding.
    @itype enable_tpool: bool
    @ivar tpool: thread pool for solver.
    @itype tpool: solvcon.mthread.ThreadPool
    @ivar arglists: argument lists for C functions to be executed in the
        thread pool.
    @itype arglists: list
    @ivar mmnames: marching methods name.
    @itype mmnames: list
    @ivar marchret: return value set for march.

    @ivar der: the dictionary to put derived data arrays.  Mostly used by
        Anchors.
    @itype der: dict
    """

    __metaclass__ = TypeWithBinder
    _pointers_ = ['exd', 'tpool', 'arglists']

    _exedatatype_ = BaseSolverExedata
    _clib_solve = None  # subclass should override.

    MESG_FILENAME_DEFAULT = 'solvcon.solver.log'
    MMNAMES = []

    def __init__(self, **kw):
        from .conf import env
        from .anchor import AnchorList
        from .gendata import Timer
        self._fpdtype = kw.pop('fpdtype', env.fpdtype)
        self._fpdtype = env.fpdtype if self._fpdtype==None else self._fpdtype
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
        # data structure for C/FORTRAN.
        self.exd = None
        self.tpool = None
        self.arglists = None
        # marching methods name.
        self.mmnames = self.MMNAMES[:]
        self.marchret = None
        # timer.
        self.timer = Timer(vtype=float)
        self.ticker = dict()
        # derived data.
        self.der = dict()

    @property
    def fpdtype(self):
        import numpy
        _fpdtype = self._fpdtype
        if isinstance(_fpdtype, str):
            return getattr(numpy, _fpdtype)
        else:
            return self._fpdtype
    @property
    def fpdtypestr(self):
        from .dependency import str_of
        return str_of(self.fpdtype)
    @property
    def _clib_solvcon(self):
        from .dependency import _clib_solvcon_of
        return _clib_solvcon_of(self.fpdtype)

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
        Create the message outputing device, which is intended for debugging
        and outputing messages related to the solver.  The outputing device is
        most useful when running distributed solvers.  The created device will
        be attach to self.

        @keyword force: flag to force the creation.  Default False,
        @type force: bool

        @return: nothing
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

    def dump(self, objfn):
        """
        Pickle self into the given filename.

        @parameter objfn: the output filename.
        @type objfn: str
        """
        import cPickle as pickle
        holds = dict()
        self.unbind()
        for key in ['mesg',]:
            holds[key] = getattr(self, key)
            setattr(self, key, None)
        pickle.dump(self, open(objfn, 'wb'), pickle.HIGHEST_PROTOCOL)
        for key in holds:
            setattr(self, key, holds[key])
        self.bind()

    def _tcall(self, *args, **kw):
        """
        Use thread pool to call C functions in parallel (shared-memory).
        """
        if not self.tpool:
            raise RuntimeError('tpool is not available in %s'%str(self))
        from ctypes import byref, c_int
        from numpy import zeros
        ncore = self.ncore
        cfunc = args[0]
        iter_start = args[1]
        iter_end = args[2]
        tickerkey = kw.pop('tickerkey', None)
        if ncore > 0:
            if len(args)>3:
                alsts = list()
                for it in range(self.ncore):
                    alst = [byref(self.exd), c_int(0), c_int(0)]
                    alst.extend(args[3:])
                    alsts.append(alst)
            else:
                alsts = self.arglists
            incre = (iter_end-iter_start)/ncore + 1
            istart = iter_start
            for it in range(ncore):
                iend = min(istart+incre, iter_end)
                alsts[it][1].value = istart
                alsts[it][2].value = iend
                istart = iend
            ret = self.tpool(cfunc, alsts)
        else:
            alst = [byref(self.exd), c_int(iter_start), c_int(iter_end)]
            alst.extend(args[3:])
            ret = [cfunc(*alst)]
        if tickerkey != None:
            if tickerkey not in self.ticker:
                self.ticker[tickerkey] = zeros(len(ret), dtype='int32')
            for it in range(len(ret)):
                self.ticker[tickerkey][it] += ret[it]
        return ret

    def bind(self):
        """
        Put everything that cannot be pickled, such as file objects, ctypes
        pointers, etc., into self.

        @return: nothing
        """
        import sys
        from ctypes import byref, c_int
        from solvcon.mthread import ThreadPool
        # create message device.
        if self.mesg == None: self.__create_mesg()
        # detect number of cores.
        if self.ncore == -1 and sys.platform.startswith('linux2'):
            self.ncore = self.detect_ncore()
        # create executional data.
        exdtype = self._exedatatype_
        self.exd = exdtype(svr=self)
        # create thread pool.
        if self.enable_tpool:
            self.tpool = ThreadPool(nthread=self.ncore)
        self.arglists = list()
        for it in range(self.ncore):
            self.arglists.append([byref(self.exd), c_int(0), c_int(0)])

    def init(self, **kw):
        pass
    def final(self):
        self.unbind()

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
        Set the time for self and structures.
        """
        self.exd.time = self.time = time
        self.exd.time_increment = self.time_increment = time_increment
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

class FakeBlockVtk(object):
    """
    Faked block from solver for being used by VTK.
    """
    def __init__(self, svr):
        self.ndim = svr.ndim
        self.nnode = svr.nnode
        self.ncell = svr.ncell
        self.ndcrd = svr.ndcrd[svr.ngstnode:]
        self.clnds = svr.clnds[svr.ngstcell:]
        self.cltpn = svr.cltpn[svr.ngstcell:]
        self.fpdtype = svr.fpdtype

class BlockSolver(BaseSolver):
    """
    Generic class for multi-dimensional (implemented with Block)
    sequential/parallel solvers.  Meta, metric, and connectivity data arrays
    are absorbed into the instance of this class.

    Before the invocation of init() method, bind() method must be called.

    @note: When subclass BlockSolver, in the init() method in the subclass must
    be initilized first, and the super().init() can then be called.  Otherwise
    the BCs can't set correct information to the solver.

    @cvar _interface_init_: list of attributes (arrays) to be exchanged on
        interface when initialized.
    @ctype _interface_init_: list

    @ivar ibcthread: flag if using threads.
    @itype ibcthread: bool

    @ivar svrn: serial number of solver object.
    @itype svrn: int
    @ivar nsvr: number of solver objects.
    @itype nsvr: int

    @ivar grpnames: list of names of groups.
    @itype grpnames: list
    @ivar ngroup: number of groups.
    @itype ngroup: int

    @ivar bclist: list of BCs.
    @itype bclist: list
    @ivar ibclist: list of interface BCs.
    @itype ibclist: list
    @ivar all_simplex: True if the mesh is all-simplex, False otherwise.
    @itype all_simplex: bool
    @ivar use_incenter: True if the mesh uses incenters, False otherwise.
    @itype use_incenter: bool
    """

    _interface_init_ = []
    _solution_array_ = []

    from .block import Block
    FCMND = Block.FCMND
    CLMND = Block.CLMND
    CLMFC = Block.CLMFC
    del Block

    def __init__(self, blk, *args, **kw):
        from numpy import empty
        self.ibcthread = kw.pop('ibcthread', False)
        super(BlockSolver, self).__init__(*args, **kw)
        assert self.fpdtype == blk.fpdtype
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
