# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Definition of the structure of solvers.
"""

from .gendata import TypeWithBinder
from .dependency import FortranType

class BaseSolver(object):
    """
    Generic solver definition.  It is an abstract class and should not be used
    to any concrete simulation case.  The concrete solver sub-classes should
    override the empty init and final methods for initialization and 
    finalization, respectively.

    @ivar _fpdtype: dtype for the floating point data in the block instance.
    @itype _fpdtype: numpy.dtype
    @ivar runanchors: the list for the anchor objects to be run.
    @itype runanchors: solvcon.anchor.AnchorList
    @ivar ankdict: the container of anchor object for communication with hook
        objects.
    @itype ankdict: dict
    """

    __metaclass__ = TypeWithBinder

    _pointers_ = [] # for binder.

    def __init__(self, **kw):
        """
        @keyword fpdtype: dtype for the floating point data.
        """
        from .conf import env
        from .anchor import AnchorList
        self._fpdtype = kw.pop('fpdtype', env.fpdtype)
        self._fpdtype = env.fpdtype if self._fpdtype==None else self._fpdtype
        # anchor list.
        self.runanchors = AnchorList(self)
        self.ankdict = dict()

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
    def fpptr(self):
        from .dependency import pointer_of
        return pointer_of(self.fpdtype)
    @property
    def _clib_solvcon(self):
        from .dependency import _clib_solvcon_of
        return _clib_solvcon_of(self.fpdtype)

    def init(self, **kw):
        """
        An empty initializer for the solver object.

        @return: nothing.
        """
        pass

    def provide(self):
        self.runanchors('provide')
    def preloop(self):
        self.runanchors('preloop')

    def march(self, time, time_increment, steps_run):
        """
        An empty marcher for the solver object.

        @param time: starting time of marching.
        @type time: float
        @param time_increment: temporal interval for the time step.
        @type time_increment: float
        @param steps_run: the count of time steps to run.
        @type steps_run: int
        @return: maximum CFL number.
        @rtype: float
        """
        return -2.0

    def postloop(self):
        self.runanchors('postloop')
    def exhaust(self):
        self.runanchors('exhaust')

    def final(self):
        """
        An empty finalizer for the solver object.

        @return: nothing.
        """
        pass

class BlockSolverExeinfo(FortranType):
    """
    Execution information for BlockSolver.

    """
    _fortran_name_ = 'execution'
    from ctypes import c_int, c_double
    _fields_ = [
        ('neq', c_int),
        ('time', c_double), ('time_increment', c_double),
    ]
    del c_int, c_double

class BlockSolver(BaseSolver):
    """
    Generic class for multi-dimensional (implemented with Block)
    sequential/parallel solvers.

    Before the invocation of init() method, bind() method must be called.

    @note: When subclass BlockSolver, in the init() method in the subclass must
    be initilized first, and the super().init() can then be called.  Otherwise
    the BCs can't set correct information to the solver.

    @cvar _clib_solve: the external dll (accessible through ctypes) which do
        the cell loop.  Subclass should override it.
    @ctype _clib_solve: ctypes.CDLL
    @cvar _exeinfotype_: the type of Exeinfo (solvcon.dependency.FortranType) 
        for the solver.
    @ctype _exeinfotype_: type
    @cvar _interface_init_: list of attributes (arrays) to be exchanged on
        interface when initialized.
    @ctype _interface_init_: list

    @ivar enable_mesg: flag if mesg device should be enabled.
    @itype enable_mesg: bool
    @ivar mesg: message printer attached to a certain solver object; designed
        and mainly used for parallel solver.
    @itype mesg: solvcon.helper.Printer

    @ivar svrn: serial number of block.
    @itype svrn: int
    @ivar msh: shape information of the Block.
    @itype msh: solvecon.block.BlockShape
    @ivar exn: execution information for the solver.  This should be redefined
        in child classes.
    @itype exn: BlockSolverExeinfo

    @ivar cecnd: coordinates of center of CCE/BCEs.
    @itype cecnd: numpy.ndarray
    @ivar cevol: volumes of CCE/BCEs.
    @itype cevol: numpy.ndarray

    @ivar sol: solution variables (for last time-step).
    @itype sol: numpy.ndarray
    @ivar soln: solution variables (to be updated for the marching time-step).
    @itype soln: numpy.ndarray
    @ivar dsol: gradient of solution variables (for last time-step).
    @itype dsol: numpy.ndarray
    @ivar dsoln: gradient of solution variables (to be updated for the marching
        time-step).
    @itype dsoln: numpy.ndarray

    @ivar der: the dictionary to put derived data arrays.  Mostly used by
        Anchors.
    @itype der: dict

    @ivar _calc_soln_args: a list of ctypes entities for marchsol() method.
    @itype _calc_soln_args: list
    @ivar _calc_dsoln_args: a list of ctypes entities for dmarchsol() method.
    @itype _calc_dsoln_args: list
    """

    _pointers_ = ['exn', 'msh', 'solptr', 'solnptr', 'dsolptr', 'dsolnptr',
        '_calc_soln_args', '_calc_dsoln_args']
    _clib_solve = None  # subclass should override.
    _exeinfotype_ = BlockSolverExeinfo
    _interface_init_ = ['cecnd', 'cevol']

    from .block import Block
    FCMND = Block.FCMND
    CLMND = Block.CLMND
    CLMFC = Block.CLMFC
    del Block

    DEBUG_FILENAME_TEMPLATE = 'solvcon.solver%d.log'
    DEBUG_FILENAME_DEFAULT = 'solvcon.solver.log'

    def pop_exnkw(self, blk, kw):
        exnkw = dict()
        exnkw['neq'] = kw.pop('neq')
        # just placeholder for marchers.
        exnkw['time'] = 0.0
        exnkw['time_increment'] = 0.0
        return exnkw

    def __init__(self, blk, *args, **kw):
        """
        @keyword neq: number of equations (variables).
        @type neq: int
        @keyword enable_mesg: flag if mesg device should be enabled.
        @type enable_mesg: bool
        """
        from numpy import empty
        self.enable_mesg = kw.pop('enable_mesg', False)
        self.exnkw = self.pop_exnkw(blk, kw)
        neq = self.exnkw['neq']
        super(BlockSolver, self).__init__(*args, **kw)
        assert self.fpdtype == blk.fpdtype
        # list of tuples for interfaces.
        self.ibclist = list()
        # absorb block.
        ## meta-data.
        self.svrn = blk.blkn
        self.nsvr = None
        ### shape.
        self.ndim = blk.ndim
        self.nnode = blk.nnode
        self.nface = blk.nface
        self.ncell = blk.ncell
        self.nbound = blk.nbound
        self.ngstnode = blk.ngstnode
        self.ngstface = blk.ngstface
        self.ngstcell = blk.ngstcell
        ### cell grouping and BCs.
        self.grpnames = blk.grpnames
        self.clgrp = blk.shclgrp
        self.bclist = blk.bclist
        for bc in self.bclist:
            bc.blk = None
            bc.svr = self
        ## connectivity.
        self.clnds = blk.shclnds
        self.clfcs = blk.shclfcs
        self.fcnds = blk.shfcnds
        self.fccls = blk.shfccls
        ## metrics.
        self.ndcrd = blk.shndcrd
        self.fccnd = blk.shfccnd
        self.fcnml = blk.shfcnml
        self.clcnd = blk.shclcnd
        self.clvol = blk.shclvol
        # data structure for C/FORTRAN.
        self.msh = None
        self.exn = None
        # create arrays.
        ndim = self.ndim
        ncell = self.ncell
        ngstcell = self.ngstcell
        ## metrics.
        self.cecnd = empty(
            (ngstcell+ncell, blk.CLMFC+1, ndim), dtype=self.fpdtype)
        self.cevol = empty((ngstcell+ncell, blk.CLMFC+1), dtype=self.fpdtype)
        ## solutions.
        self.sol = empty((ngstcell+ncell, neq), dtype=self.fpdtype)
        self.soln = empty((ngstcell+ncell, neq), dtype=self.fpdtype)
        self.dsol = empty((ngstcell+ncell, neq, ndim), dtype=self.fpdtype)
        self.dsoln = empty((ngstcell+ncell, neq, ndim), dtype=self.fpdtype)
        ## derived data.
        self.der = dict()
        # placeholders.
        self.mesg = None
        self.solptr = None
        self.solnptr = None
        self.dsolptr = None
        self.dsolnptr = None
        self._calc_soln_args = None
        self._calc_dsoln_args = None

    @property
    def args_struct(self):
        from ctypes import byref
        assert self.msh != None and self.exn != None
        return [byref(self.msh), byref(self.exn)]

    def create_mesg(self, force=False):
        import os
        from .helper import Printer
        if force: self.enable_mesg = True
        if self.enable_mesg:
            if self.svrn != None:
                dfn = self.DEBUG_FILENAME_TEMPLATE % self.svrn
                dprefix = 'SOLVER%d: '%self.svrn
            else:
                dfn = self.DEBUG_FILENAME_DEFAULT
                dprefix = ''
        else:
            dfn = os.devnull
            dprefix = ''
        self.mesg = Printer(dfn, prefix=dprefix, override=True)

    def bind(self):
        """
        Bind all the boundary condition objects.

        @note: BC must be bound AFTER solver "pointers".  Overridders to the
            method should firstly bind all pointers, secondly super binder, and 
            then methods/subroutines.
        """
        from .block import BlockShape
        # create debug printer.
        if self.mesg == None: self.create_mesg()
        # structures.
        self.msh = BlockShape(
            ndim=self.ndim,
            fcmnd=self.FCMND, clmnd=self.CLMND, clmfc=self.CLMFC,
            nnode=self.nnode, nface=self.nface, ncell=self.ncell,
            nbound=self.nbound,
            ngstnode=self.ngstnode, ngstface=self.ngstface,
            ngstcell=self.ngstcell,
        )
        self.exn = self._exeinfotype_(**self.exnkw)
        # pointers.
        fpptr = self.fpptr
        self.solptr = self.sol.ctypes.data_as(fpptr)
        self.solnptr = self.soln.ctypes.data_as(fpptr)
        self.dsolptr = self.dsol.ctypes.data_as(fpptr)
        self.dsolnptr = self.dsoln.ctypes.data_as(fpptr)
        self._calc_soln_args = self.args_struct + [
            self.clvol.ctypes.data_as(fpptr),
            self.solptr,
            self.solnptr,
        ]
        self._calc_dsoln_args = self.args_struct + [
            self.clcnd.ctypes.data_as(fpptr),
            self.dsolptr,
            self.dsolnptr,
        ]
        # boundary conditions.
        for bc in self.bclist:
            bc.bind()
        super(BlockSolver, self).bind()

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

        @note: BC must be initialized AFTER solver itself.
        """
        for bc in self.bclist:
            bc.init(**kw)
        super(BlockSolver, self).init(**kw)

    def dump(self, objfn):
        import cPickle as pickle
        self.unbind()
        mesg = self.mesg
        self.mesg = None
        pickle.dump(self, open(objfn, 'w'), pickle.HIGHEST_PROTOCOL)
        self.mesg = mesg
        self.bind()

    ##################################################
    # CESE solving algorithm.
    ##################################################
    def calc_soln(self):
        self._clib_solvcon.calc_soln_(*self._calc_soln_args)
    def calc_dsoln(self):
        self._clib_solvcon.calc_dsoln_(*self._calc_dsoln_args)

    def boundcond(self):
        """
        Update the boundary conditions.

        @return: nothing.
        """
        for bc in self.bclist: bc.sol()
        for bc in self.bclist: bc.dsol() 
    def update(self):
        """
        Copy from old solution arrays to new solution arrays.

        @return: nothing.
        """
        self.sol[:,:] = self.soln[:,:]
        self.dsol[:,:,:] = self.dsoln[:,:,:]

    def marchsol(self, time, time_increment):
        """
        March the solution U vector in the solver and BCs.

        @return: nothing.
        """
        self.exn.time = time
        self.exn.time_increment = time_increment
        self.calc_soln()
    def estimatecfl(self):
        return -2.0
    def marchdsol(self, time, time_increment):
        """
        March the gradient of solution dU vector in the solver and BCs.

        @return: nothing.
        """
        self.exn.time = time
        self.exn.time_increment = time_increment
        self.calc_dsoln()

    def boundsol(self):
        for bc in self.bclist: bc.sol()
    def bounddsol(self):
        for bc in self.bclist: bc.dsol()

    def march(self, time, time_increment, steps_run, worker=None):
        maxCFL = -2.0
        istep = 0
        while istep < steps_run:
            self.runanchors('prefull')
            for ihalf in range(2):
                self.runanchors('prehalf')
                self.update()
                # solutions.
                self.runanchors('premarchsoln')
                self.marchsol(time, time_increment)
                self.runanchors('preexsoln')
                if worker: self.exchangeibc('soln', worker=worker)
                self.runanchors('prebcsoln')
                for bc in self.bclist: bc.sol()
                self.runanchors('precfl')
                cCFL = self.estimatecfl()
                maxCFL = cCFL if cCFL > maxCFL else maxCFL
                # solution gradients.
                self.runanchors('premarchdsoln')
                self.marchdsol(time, time_increment)
                self.runanchors('preexdsoln')
                if worker: self.exchangeibc('dsoln', worker=worker)
                self.runanchors('prebcdsoln')
                for bc in self.bclist: bc.dsol()
                # increment time.
                time += time_increment/2
                self.runanchors('posthalf')
            istep += 1
            self.runanchors('postfull')
        if worker:
            worker.conn.send(maxCFL)
        return maxCFL

    ##################################################
    # below are for parallelization.
    ##################################################
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
        obj = getattr(self.ankdict[ankname], objname)
        conn.send(obj)

    def init_exchange(self, ifacelist):
        from .boundcond import interface
        # grab peer index.
        ibclist = list()
        for pair in ifacelist:
            assert len(pair) == 2
            assert self.svrn in pair
            ibclist.append(sum(pair)-self.svrn)
        # replace with bc plus peer indices.
        for bc in self.bclist:
            if not isinstance(bc, interface):
                continue
            it = ibclist.index(bc.rblkn)
            ibclist[it] = (bc, ifacelist[it][0], ifacelist[it][1])
        self.ibclist = ibclist

    def exchangeibc(self, arrname, worker=None):
        ibclist = self.ibclist
        for bc, sendn, recvn in ibclist:
            if self.svrn == sendn:
                self.pushibc(arrname, bc, recvn, worker=worker)
            elif self.svrn == recvn:
                self.pullibc(arrname, bc, sendn, worker=worker)
            else:
                raise ValueError, 'bc.rblkn = %d != %d or %d' % (
                    bc.rblkn, sendn, recvn)

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
        ngstcell = self.ngstcell
        conn = worker.pconns[bc.rblkn]
        arr = getattr(self, arrname)
        # ask the receiver for data.
        rarr = conn.recv()  # comm.
        slct = bc.rclp[:,0] + ngstcell
        arr[slct] = rarr[:]
        # provide the receiver with data.
        slct = bc.rclp[:,2] + ngstcell
        conn.send(arr[slct])    # comm.

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
        ngstcell = self.ngstcell
        conn = worker.pconns[bc.rblkn]
        arr = getattr(self, arrname)
        # provide sender the data.
        slct = bc.rclp[:,2] + ngstcell
        conn.send(arr[slct])    # comm.
        # ask data from sender.
        rarr = conn.recv()  # comm.
        slct = bc.rclp[:,0] + ngstcell
        arr[slct] = rarr[:]
