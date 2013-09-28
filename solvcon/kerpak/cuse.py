# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2011 Yung-Yu Chen <yyc@solvcon.net>.
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
Second-order, multi-dimensional CESE method with CUDA enabled.

Four functionalities are defined: (i) CuseGlue, (ii) CFL (CflHook), (iii)
Convergence (ConvergeAnchor, ConvergeHook), and (iv) Prober (Probe,
ProbeAnchor, ProbeHook).
"""

CUDA_RAISE_ON_FAIL = False

from ctypes import Structure
from solvcon.gendata import AttributeDict
from solvcon.solver import BlockSolver
from solvcon.case import BlockCase
from solvcon.boundcond import BC, periodic
from solvcon.anchor import Anchor, GlueAnchor
from solvcon.hook import Hook, BlockHook

class CudaDataManager(AttributeDict):
    """
    Customized dictionary managing solver data on GPU memory.  Each item in
    the object represents a block of allocated memory on GPU, i.e.,
    solvcon.scuda.GpuMemory object.
    """
    def __init__(self, *args, **kw):
        from ctypes import sizeof
        self.svr = kw.pop('svr')
        super(CudaDataManager, self).__init__(*args, **kw)
        self.exd = self.svr._exedatatype_(svr=self.svr)
        self.gexd = self.svr.scu.alloc(sizeof(self.exd))
        # allocate all arrays.
        for name in self.svr.cuarr_map:
            shf = self.svr.cuarr_map[name]
            carr = getattr(self.svr, name)
            self[name] = self.svr.scu.alloc(carr.nbytes)
    def __set_cuda_pointer(self, exc, svr, aname, shf):
        from ctypes import c_void_p
        arr = getattr(svr, aname)
        gptr = self[aname].gptr
        nt = 1
        if len(arr.shape) > 1:
            for width in arr.shape[1:]:
                nt *= width
        shf *= nt * arr.itemsize
        addr = gptr.value + shf if gptr.value is not None else None
        setattr(exc, aname, c_void_p(addr))

    def update_exd(self, exd=None):
        if exd is None:
            exd = self.svr.exd
        # copy everything.
        for key, ctp in exd._fields_:
            setattr(self.exd, key, getattr(exd, key))
        # shift pointers.
        for name in self.svr.cuarr_map:
            shf = self.svr.cuarr_map[name]
            self.__set_cuda_pointer(self.exd, self.svr, name, shf)
        # send to GPU.
        self.exd_to_gpu()

    def exd_to_gpu(self):
        from ctypes import byref, sizeof
        scu = self.svr.scu
        scu.cudaMemcpy(self.gexd.gptr, byref(self.exd),
            sizeof(self.exd), scu.cudaMemcpyHostToDevice)

    def arr_to_gpu(self, *args):
        scu = self.svr.scu
        names = self if not args else args
        for name in names:
            scu.memcpy(self[name], getattr(self.svr, name))
    def arr_from_gpu(self, *args):
        scu = self.svr.scu
        names = self if not args else args
        for name in names:
            scu.memcpy(getattr(self.svr, name), self[name])

    def free_all(self):
        scu = self.svr.scu
        scu.free(self.gexd)
        for name in self:
            scu.free(self[name])

###############################################################################
# Solver.
###############################################################################

class CuseSolverExedata(Structure):
    """
    Data structure to interface with C.
    """
    from ctypes import c_int, c_double, c_void_p
    _fields_ = [
        # inherited.
        ('ncore', c_int), ('neq', c_int),
        ('time', c_double), ('time_increment', c_double),
        # mesh shape.
        ('ndim', c_int), ('nnode', c_int), ('nface', c_int), ('ncell', c_int),
        ('nbound', c_int),
        ('ngstnode', c_int), ('ngstface', c_int), ('ngstcell', c_int),
        # group shape.
        ('ngroup', c_int), ('gdlen', c_int),
        # parameter shape.
        ('nsca', c_int), ('nvec', c_int),
        # scheme.
        ('alpha', c_int), ('sigma0', c_double),
        ('taylor', c_double), ('cnbfac', c_double), ('sftfac', c_double),
        ('taumin', c_double), ('tauscale', c_double),
        # function pointer.
        ('jacofunc', c_void_p),
        # meta array.
        ('fctpn', c_void_p), ('cltpn', c_void_p), ('clgrp', c_void_p),
        ('grpda', c_void_p),
        # metric array.
        ('ndcrd', c_void_p),
        ('fccnd', c_void_p), ('fcnml', c_void_p), ('fcara', c_void_p),
        ('clcnd', c_void_p), ('clvol', c_void_p),
        ('cecnd', c_void_p), ('cevol', c_void_p), ('sfmrc', c_void_p),
        # connectivity array.
        ('fcnds', c_void_p), ('fccls', c_void_p),
        ('clnds', c_void_p), ('clfcs', c_void_p),
        # solution array.
        ('amsca', c_void_p), ('amvec', c_void_p),
        ('sol', c_void_p), ('dsol', c_void_p), ('solt', c_void_p),
        ('soln', c_void_p), ('dsoln', c_void_p),
        ('stm', c_void_p), ('cfl', c_void_p), ('ocfl', c_void_p),
    ]
    del c_int, c_double, c_void_p

    def __set_pointer(self, svr, aname, shf):
        from ctypes import c_void_p
        ptr = getattr(svr, aname)[shf:].ctypes._as_parameter_
        setattr(self, aname, ptr)

    def __init__(self, *args, **kw):
        from ctypes import c_int, c_double, POINTER, c_void_p, byref, cast
        svr = kw.pop('svr', None)
        super(CuseSolverExedata, self).__init__(*args, **kw)
        if svr == None:
            return
        # function pointer.
        self.jacofunc = cast(svr._jacofunc_, c_void_p).value
        for key in ('ncore', 'neq', 'time', 'time_increment',
                    'ndim', 'nnode', 'nface', 'ncell', 'nbound', 'ngstnode',
                    'ngstface', 'ngstcell', 'ngroup', 'gdlen', 'nsca', 'nvec',
                    'alpha', 'sigma0', 'taylor', 'cnbfac', 'sftfac',
                    'taumin', 'tauscale'):
            setattr(self, key, getattr(svr, key))
        # arrays.
        for aname in ('grpda', 'sfmrc'):
            self.__set_pointer(svr, aname, 0)
        for aname in ('ndcrd',):
            self.__set_pointer(svr, aname, svr.ngstnode)
        for aname in ('fctpn', 'fcnds', 'fccls', 'fccnd', 'fcnml', 'fcara'):
            self.__set_pointer(svr, aname, svr.ngstface)
        for aname in ('cltpn', 'clgrp', 'clcnd', 'clvol', 'cecnd', 'cevol',
                      'clnds', 'clfcs', 'amsca', 'amvec',
                      'sol', 'dsol', 'solt', 'soln', 'dsoln',
                      'stm', 'cfl', 'ocfl'):
            self.__set_pointer(svr, aname, svr.ngstcell)

class CuseSolver(BlockSolver):
    """
    The base solver class for second-order, multi-dimensional CESE code with
    CUDA enabled.

    @cvar _gdlen_: length per group data.  Must be overridden.
    @ctype _gdlen_: int
    @cvar _jacofunc_: ctypes function to Jacobian calculator.  Must be
        overridden.
    @ctype _jacofunc_: ctypes.FuncPtr
    @cvar _clib_mcu: ctypes library for physical model on GPU.
    @ctype _clib_mcu: ctypes.CDLL

    @ivar debug: flag for debugging.
    @itype debug: bool

    @ivar scu: CUDA wrapper.
    @itype scu: solvcon.scuda.Scuda
    @ivar ncuth: number of thread per block for CUDA.
    @itype ncuth: int

    @ivar alpha: parameter to the weighting function.
    @itype alpha: int
    @ivar sigma0: constant parameter for W-3 scheme; should be of order of 1.
        Default is 3.0.
    @itype sigma0: float
    @ivar taylor: factor for Taylor's expansion; 0 off, 1 on.
    @itype taylor: float
    @ivar cnbfac: factor to use BCE centroid, othersize midpoint; 0 off, 1 on.
    @itype cnbfac: float
    @ivar sftfac: factor to shift gradient shape; 0 off, 1 on.
    @itype sftfac: float
    @ivar taumin: the lower bound of tau.
    @itype taumin: float
    @ivar tauscale: scaling of tau.
    @itype tauscale: float

    @ivar grpda: group data.
    @ivar cecnd: solution points for CCEs and BCEs.
    @ivar cevol: CCE and BCE volumes.
    @ivar sfmrc: sub-face geometry.  It is a 5-dimensional array, and the shape
        is (ncell, CLMFC, FCMND, 2, NDIM).  sfmrc[...,0,:] are centers,
        while sfmrc[...,1,:] are normal vectors.
    @ivar amsca: Parameter scalar array.
    @ivar amvec: Parameter vector array.
    @ivar solt: temporal diffrentiation of solution.
    @ivar sol: current solution.
    @ivar soln: next solution.
    @ivar dsol: current gradient of solution.
    @ivar dsoln: next gradient of solution.
    @ivar cfl: CFL number.
    @ivar ocfl: original CFL number.
    """

    _exedatatype_ = CuseSolverExedata
    _interface_init_ = ['cecnd', 'cevol']
    _solution_array_ = ['sol', 'soln', 'dsol', 'dsoln']

    _gdlen_ = None
    _jacofunc_ = None
    _clib_mcu = None

    def __init__(self, blk, *args, **kw):
        from numpy import empty
        self.debug = kw.pop('debug', False)
        # shape parameters.
        nsca = kw.pop('nsca', 0)
        nvec = kw.pop('nvec', 0)
        # CUDA parameters.
        self.ncuth = kw.pop('ncuth',  0)
        self.scu = None
        # scheme parameters.
        self.alpha = int(kw.pop('alpha', 0))
        self.sigma0 = int(kw.pop('sigma0', 3.0))
        self.taylor = float(kw.pop('taylor', 1.0))  # dirty hack.
        self.cnbfac = float(kw.pop('cnbfac', 1.0))  # dirty hack.
        self.sftfac = float(kw.pop('sftfac', 1.0))  # dirty hack.
        self.taumin = float(kw.pop('taumin', 0.0))
        self.tauscale = float(kw.pop('tauscale', 1.0))
        # super call.
        kw.setdefault('enable_tpool', False)
        super(CuseSolver, self).__init__(blk, *args, **kw)
        fpdtype = self.fpdtype
        ndim = self.ndim
        ncell = self.ncell
        ngstcell = self.ngstcell
        neq = self.neq
        ngroup = self.ngroup
        # CUDA manager.
        self.cumgr = None
        self.cuarr_map = dict()
        for key in ('ndcrd',):
            self.cuarr_map[key] = self.ngstnode
        for key in ('fcnds', 'fccls', 'fccnd', 'fcnml', 'fcara'):
            self.cuarr_map[key] = self.ngstface
        for key in ('clfcs', 'clcnd', 'cltpn'):
            self.cuarr_map[key] = self.ngstcell
        # meta array.
        self.grpda = empty((ngroup, self._gdlen_), dtype=fpdtype)
        self.cuarr_map['grpda'] = 0
        # dual mesh.
        self.cecnd = empty((ngstcell+ncell, self.CLMFC+1, ndim), dtype=fpdtype)
        self.cevol = empty((ngstcell+ncell, self.CLMFC+1), dtype=fpdtype)
        self.cuarr_map['cecnd'] = self.cuarr_map['cevol'] = self.ngstcell
        self.sfmrc = empty((ncell, self.CLMFC, self.FCMND, 2, ndim),
            dtype=fpdtype)
        self.cuarr_map['sfmrc'] = 0
        # solutions.
        self.amsca = empty((ngstcell+ncell, nsca), dtype=fpdtype)
        self.amvec = empty((ngstcell+ncell, nvec, ndim), dtype=fpdtype)
        self.solt = empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.sol = empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.soln = empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.dsol = empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        self.dsoln = empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        self.stm = empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.cfl = empty(ngstcell+ncell, dtype=fpdtype)
        self.ocfl = empty(ngstcell+ncell, dtype=fpdtype)
        for key in ('amsca', 'amvec', 'solt', 'sol', 'soln', 'dsol', 'dsoln',
            'stm', 'cfl', 'ocfl'):
            self.cuarr_map[key] = self.ngstcell

    @property
    def gdlen(self):
        """
        Length per group data.
        """
        return self._gdlen_
    @property
    def nsca(self):
        return self.amsca.shape[1]
    @property
    def nvec(self):
        return self.amvec.shape[1]

    def bind(self):
        from solvcon.conf import env
        self.scu = env.scu if self.ncuth else None
        if self.scu:
            self.cumgr = CudaDataManager(svr=self)
        super(CuseSolver, self).bind()
    def unbind(self):
        if self.scu and self.cumgr is not None:
            self.cumgr.free_all()
            self.cumgr = None
        super(CuseSolver, self).unbind()

    def init(self, **kw):
        from ctypes import byref
        self._clib_cuse_c.prepare_ce(byref(self.exd))
        super(CuseSolver, self).init(**kw)
        self._clib_cuse_c.prepare_sf(byref(self.exd))
        if self.scu: self.cumgr.arr_to_gpu()
        self.mesg('cuda is %s\n'%('on' if self.scu else 'off'))

    def boundcond(self):
        if self.scu:
            self.cumgr.update_exd()
            self.cumgr.arr_to_gpu('sol', 'soln', 'dsol', 'dsoln')
            if self.nsca: self.cumgr.arr_to_gpu('amsca')
            if self.nvec: self.cumgr.arr_to_gpu('amvec')
        super(CuseSolver, self).boundcond()
        self.call_non_interface_bc('soln')
        if self.scu:
            self.cumgr.arr_from_gpu('sol', 'soln')
        self.call_non_interface_bc('dsoln')
        if self.scu:
            self.cumgr.arr_from_gpu('dsol', 'dsoln')

    ###########################################################################
    # parallelization.
    ###########################################################################
    def pushibc(self, arrname, bc, recvn, worker=None):
        """
        Push data toward selected interface which connect to blocks with larger
        serial number than myself.  If CUDA is present, data are first uploaded
        to and then downloaded from GPU.

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
        shape = list(arr.shape)
        shape[0] = bc.rclp.shape[0]
        # for CUDA up/download.
        if self.scu:
            stride = arr.itemsize
            for size in arr.shape[1:]:
                stride *= size
        # ask the receiver for data.
        rarr = empty(shape, dtype=arr.dtype)
        conn.recvarr(rarr)  # comm.
        # set array and upload to GPU.
        slct = bc.rclp[:,0] + ngstcell
        if self.scu:
            gslct = self.scu.alloc(slct.nbytes)
            self.scu.memcpy(gslct, slct)
            gbrr = self.scu.alloc(rarr.nbytes)
            self.scu.memcpy(gbrr, rarr)
            garr = self.cumgr[arrname]
            self._clib_cuse_cu.slct_io(self.ncuth, 0, len(slct), stride,
                gslct.gptr, garr.gptr, gbrr.gptr)
        else:
            arr[slct] = rarr[:]
        # download from GPU and get array.
        slct = bc.rclp[:,2] + ngstcell
        if self.scu:
            self.scu.memcpy(gslct, slct)
            self._clib_cuse_cu.slct_io(self.ncuth, 1, len(slct), stride,
                gslct.gptr, garr.gptr, gbrr.gptr)
            self.scu.memcpy(rarr, gbrr)
            self.scu.free(gbrr)
            self.scu.free(gslct)
        else:
            rarr[:] = arr[slct]
        # provide the receiver with data.
        conn.sendarr(rarr)  # comm.

    def pullibc(self, arrname, bc, sendn, worker=None):
        """
        Pull data from the interface determined by the serial of peer.  If CUDA
        is present, data are first downloaded from GPU and then uploaded to
        GPU.

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
        shape = list(arr.shape)
        shape[0] = bc.rclp.shape[0]
        # for CUDA up/download.
        if self.scu:
            stride = arr.itemsize
            for size in arr.shape[1:]:
                stride *= size
        # download from GPU and get array.
        slct = bc.rclp[:,2] + ngstcell
        rarr = empty(shape, dtype=arr.dtype)
        if self.scu:
            gslct = self.scu.alloc(slct.nbytes)
            self.scu.memcpy(gslct, slct)
            gbrr = self.scu.alloc(rarr.nbytes)
            garr = self.cumgr[arrname]
            self._clib_cuse_cu.slct_io(self.ncuth, 1, len(slct), stride,
                gslct.gptr, garr.gptr, gbrr.gptr)
            self.scu.memcpy(rarr, gbrr)
        else:
            rarr[:] = arr[slct]
        # provide sender the data.
        conn.sendarr(rarr)  # comm.
        # ask data from sender.
        conn.recvarr(rarr)  # comm.
        # set array and upload to GPU.
        slct = bc.rclp[:,0] + ngstcell
        if self.scu:
            self.scu.memcpy(gslct, slct)
            self.scu.memcpy(gbrr, rarr)
            self._clib_cuse_cu.slct_io(self.ncuth, 0, len(slct), stride,
                gslct.gptr, garr.gptr, gbrr.gptr)
            self.scu.free(gbrr)
            self.scu.free(gslct)
        else:
            arr[slct] = rarr[:]

    ###########################################################################
    # utility.
    ###########################################################################
    def locate_point(self, *args):
        """
        Locate the cell index where the input coordinate is.
        """
        from ctypes import byref, c_int
        from numpy import array
        crd = array(args, dtype='float64')
        picl = c_int(0)
        pifl = c_int(0)
        pjcl = c_int(0)
        pjfl = c_int(0)
        self._clib_cuse_c.locate_point(byref(self.exd),
            crd.ctypes._as_parameter_,
            byref(picl), byref(pifl), byref(pjcl), byref(pjfl))
        return picl.value, pifl.value, pjcl.value, pjfl.value

    ###########################################################################
    # library.
    ###########################################################################
    from solvcon.dependency import getcdll
    __clib_cuse_c = {
        2: getcdll('cuse2d_c', raise_on_fail=False),
        3: getcdll('cuse3d_c', raise_on_fail=False),
    }
    __clib_cuse_cu = {
        2: getcdll('cuse2d_cu', raise_on_fail=CUDA_RAISE_ON_FAIL),
        3: getcdll('cuse3d_cu', raise_on_fail=CUDA_RAISE_ON_FAIL),
    }
    del getcdll
    @property
    def _clib_cuse_c(self):
        return self.__clib_cuse_c[self.ndim]
    @property
    def _clib_cuse_cu(self):
        return self.__clib_cuse_cu[self.ndim]

    ###########################################################################
    # marching algorithm.
    ###########################################################################
    MMNAMES = list()

    MMNAMES.append('update')
    def update(self, worker=None):
        if self.debug: self.mesg('update')
        # exchange solution and gradient.
        self.sol, self.soln = self.soln, self.sol
        self.dsol, self.dsoln = self.dsoln, self.dsol
        # exchange pointers in execution data.
        exd = self.exd
        exd.sol, exd.soln = exd.soln, exd.sol
        exd.dsol, exd.dsoln = exd.dsoln, exd.dsol
        # exchange items in GPU execution data.
        if self.scu:
            cumgr = self.cumgr
            cumgr.sol, cumgr.soln = cumgr.soln, cumgr.sol
            cumgr.dsol, cumgr.dsoln = cumgr.dsoln, cumgr.dsol
            cumgr.update_exd()
        if self.debug: self.mesg(' done.\n')

    MMNAMES.append('ibcam')
    def ibcam(self, worker=None):
        if self.debug: self.mesg('ibcam')
        if worker:
            if self.nsca: self.exchangeibc('amsca', worker=worker)
            if self.nvec: self.exchangeibc('amvec', worker=worker)
        if self.debug: self.mesg(' done.\n')

    MMNAMES.append('calcsolt')
    def calcsolt(self, worker=None):
        from ctypes import byref
        if self.debug: self.mesg('calcsolt')
        if self.scu:
            self._clib_mcu.calc_solt(self.ncuth,
                byref(self.cumgr.exd), self.cumgr.gexd.gptr)
        else:
            self._clib_cuse_c.calc_solt(byref(self.exd),
                -self.ngstcell, self.ncell)
        if self.debug: self.mesg(' done.\n')
    MMNAMES.append('calcsoln')
    def calcsoln(self, worker=None):
        from ctypes import byref
        if self.debug: self.mesg('calcsoln')
        if self.scu:
            self._clib_mcu.calc_soln(self.ncuth,
                byref(self.cumgr.exd), self.cumgr.gexd.gptr)
        else:
            self._clib_cuse_c.calc_soln(byref(self.exd))
        if self.debug: self.mesg(' done.\n')

    MMNAMES.append('ibcsoln')
    def ibcsoln(self, worker=None):
        if self.debug: self.mesg('ibcsoln')
        if worker: self.exchangeibc('soln', worker=worker)
        if self.debug: self.mesg(' done.\n')
    MMNAMES.append('bcsoln')
    def bcsoln(self, worker=None):
        if self.debug: self.mesg('bcsoln')
        self.call_non_interface_bc('soln')
        if self.debug: self.mesg(' done.\n')

    MMNAMES.append('calccfl')
    def calccfl(self, worker=None):
        raise NotImplementedError

    MMNAMES.append('calcdsoln')
    def calcdsoln(self, worker=None):
        from ctypes import byref
        if self.debug: self.mesg('calcdsoln')
        if self.scu:
            self._clib_mcu.calc_dsoln_w3(self.ncuth,
                byref(self.cumgr.exd), self.cumgr.gexd.gptr)
        else:
            self._clib_cuse_c.calc_dsoln_w3(byref(self.exd))
        if self.debug: self.mesg(' done.\n')

    MMNAMES.append('ibcdsoln')
    def ibcdsoln(self, worker=None):
        if self.debug: self.mesg('ibcdsoln')
        if worker: self.exchangeibc('dsoln', worker=worker)
        if self.debug: self.mesg(' done.\n')
    MMNAMES.append('bcdsoln')
    def bcdsoln(self, worker=None):
        if self.debug: self.mesg('bcdsoln')
        self.call_non_interface_bc('dsoln')
        if self.debug: self.mesg(' done.\n')

###############################################################################
# Case.
###############################################################################

class CuseCase(BlockCase):
    """
    Inviscid aerodynamic case for the Euler equations.
    """
    defdict = {
        'execution.verified_norm': -1.0,
        'solver.debug_cese': False,
        'solver.ncuth': 0,
        'solver.alpha': 1,
        'solver.sigma0': 3.0,
        'solver.taylor': 1.0,
        'solver.cnbfac': 1.0,
        'solver.sftfac': 1.0,
        'solver.taumin': None,
        'solver.tauscale': None,
    }
    def make_solver_keywords(self):
        kw = super(CuseCase, self).make_solver_keywords()
        kw['debug'] = self.solver.debug_cese
        kw['ncuth'] = int(self.solver.ncuth)
        kw['alpha'] = int(self.solver.alpha)
        for key in ('sigma0', 'taylor', 'cnbfac', 'sftfac',
                    'taumin', 'tauscale',):
            val = self.solver.get(key)
            if val != None: kw[key] = float(val)
        return kw

###############################################################################
# Boundary conditions.
###############################################################################

class CuseBC(BC):
    """
    Basic BC class for the cuse series solvers.  This class support glue BCs.

    @cvar _ghostgeom_: indicate which ghost geometry processor to use.
    @ctype _ghostgeom_: str
    """

    _ghostgeom_ = None

    ###########################################################################
    # library.
    ###########################################################################
    from solvcon.dependency import getcdll
    __clib_cuseb_c = {
        2: getcdll('cuseb2d_c', raise_on_fail=False),
        3: getcdll('cuseb3d_c', raise_on_fail=False),
    }
    __clib_cuseb_cu = {
        2: getcdll('cuseb2d_cu', raise_on_fail=CUDA_RAISE_ON_FAIL),
        3: getcdll('cuseb3d_cu', raise_on_fail=CUDA_RAISE_ON_FAIL),
    }
    del getcdll
    @property
    def _clib_cuseb_c(self):
        return self.__clib_cuseb_c[self.svr.ndim]
    @property
    def _clib_cuseb_cu(self):
        return self.__clib_cuseb_cu[self.svr.ndim]

    def bind(self):
        super(CuseBC, self).bind()
        if self.svr.scu:
            for key in ('facn', 'value',):
                nbytes = getattr(self, key).nbytes
                setattr(self, 'cu'+key, self.svr.scu.alloc(nbytes))
    def unbind(self):
        if self.svr.scu:
            for key in ('facn', 'value',):
                gptr = getattr(self, 'cu'+key, None)
                if gptr is not None:
                    self.svr.scu.free(getattr(self, 'cu'+key))

    def init(self, **kw):
        from ctypes import byref, c_int
        super(CuseBC, self).init(**kw)
        # process ghost geometry.
        getattr(self._clib_cuseb_c, 'ghostgeom_'+self._ghostgeom_)(
            byref(self.svr.exd), c_int(self.facn.shape[0]),
            self.facn.ctypes._as_parameter_)
        # initialize GPU data.
        if self.svr.scu:
            for key in ('facn', 'value',):
                self.svr.scu.memcpy(getattr(self, 'cu'+key), getattr(self, key))

    def soln(self):
        """
        Update ghost cells after self.svr.calcsoln.
        """
        pass
    def dsoln(self):
        """
        Update ghost cells after self.svr.calcdsoln.
        """
        pass

class CuseNonrefl(CuseBC):
    _ghostgeom_ = 'mirror'
    def soln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_cuseb_cu.bound_nonrefl_soln(svr.ncuth,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_cuseb_c.bound_nonrefl_soln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)
    def dsoln(self):
        from ctypes import byref
        svr = self.svr
        if svr.scu:
            self._clib_cuseb_cu.bound_nonrefl_dsoln(svr.ncuth,
                byref(svr.cumgr.exd),
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_cuseb_c.bound_nonrefl_dsoln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)

class CusePeriodic(periodic):
    """
    General periodic boundary condition for sequential runs.
    """
    def init(self, **kw):
        svr = self.svr
        ngstcell = svr.ngstcell
        ngstface = svr.ngstface
        facn = self.facn
        slctm = self.rclp[:,0] + ngstcell
        slctr = self.rclp[:,1] + ngstcell
        # move coordinates.
        shf = svr.cecnd[slctr,0,:] - svr.fccnd[facn[:,2]+ngstface,:]
        svr.cecnd[slctm,0,:] = svr.fccnd[facn[:,0]+ngstface,:] + shf
    def soln(self):
        svr = self.svr
        slctm = self.rclp[:,0] + svr.ngstcell
        slctr = self.rclp[:,1] + svr.ngstcell
        if svr.scu:
            svr.cumgr.arr_from_gpu('soln')
        svr.soln[slctm,:] = svr.soln[slctr,:]
        if svr.scu:
            svr.cumgr.arr_to_gpu('soln')
    def dsoln(self):
        svr = self.svr
        slctm = self.rclp[:,0] + svr.ngstcell
        slctr = self.rclp[:,1] + svr.ngstcell
        if svr.scu:
            svr.cumgr.arr_from_gpu('dsoln')
        svr.dsoln[slctm,:,:] = svr.dsoln[slctr,:,:]
        if svr.scu:
            svr.cumgr.arr_to_gpu('dsoln')

################################################################################
# CUDA downloader.
################################################################################

class CudaUpDownAnchor(Anchor):
    """
    Upload and download variable arrays between GPU device memory and CPU host
    memory.  By default preloop() and premarch() callbacks do uploading, and
    postmarch() and postloop() do downloading.  The default behavior
    is compatible to solvcon.anchor.VtkAnchor.  Subclasses can override
    callback methods for more complicated operations.

    @ivar rsteps: steps to run.
    @itype rsteps: int
    @ivar uparrs: names of arrays to be uploaded from host to device.
    @itype uparrs: list
    @ivar downarrs: names of arrays to be downloaded from device to host.
    @itype downarrs: list
    """
    def __init__(self, svr, **kw):
        self.rsteps = kw.pop('rsteps', 1)
        self.uparrs = kw.pop('uparrs', list())
        self.downarrs = kw.pop('downarrs', list())
        super(CudaUpDownAnchor, self).__init__(svr, **kw)
    def _upload(self):
        if self.svr.scu and self.uparrs:
            self.svr.cumgr.arr_to_gpu(*self.uparrs)
    def _download(self):
        if self.svr.scu and self.downarrs:
            print self.downarrs
            self.svr.cumgr.arr_from_gpu(*self.downarrs)
    def preloop(self):
        self._upload()
    def premarch(self):
        istep = self.svr.step_global
        rsteps = self.rsteps
        if istep > 0 and istep%rsteps == 0:
            self._upload()
    def postmarch(self):
        istep = self.svr.step_global
        rsteps = self.rsteps
        if istep > 0 and istep%rsteps == 0:
            self._download()
    def postloop(self):
        istep = self.svr.step_global
        rsteps = self.rsteps
        if istep%rsteps != 0:
            self.process(istep)

################################################################################
# Glue.
################################################################################

class CuseGlue(GlueAnchor):
    """
    Use Glue class to glue specified BC objects of a solver object.  The class
    is only valid for CuseSolver.
    """
    KEYS_ENABLER = ('cecnd',)
    def __init__(self, svr, **kw):
        assert isinstance(svr, CuseSolver)
        super(CuseGlue, self).__init__(svr, **kw)

################################################################################
# CFL.
################################################################################

class CflAnchor(Anchor):
    """
    Counting CFL numbers.  Use svr.marchret to return results.  Implements
    postmarch() method.

    @ivar rsteps: steps to run.
    @itype rsteps: int
    """
    def __init__(self, svr, **kw):
        self.rsteps = kw.pop('rsteps')
        super(CflAnchor, self).__init__(svr, **kw)
    def postmarch(self):
        svr = self.svr
        istep = svr.step_global
        rsteps = self.rsteps
        if istep > 0 and istep%rsteps == 0:
            # download data.
            if svr.scu:
                svr.cumgr.arr_from_gpu('cfl', 'ocfl')
            ocfl = svr.ocfl[svr.ngstcell:]
            cfl = svr.cfl[svr.ngstcell:]
            # determine extremum.
            mincfl = ocfl.min()
            maxcfl = ocfl.max()
            nadj = (cfl==1).sum()
            # store.
            lst = svr.marchret.setdefault('cfl', [0.0, 0.0, 0, 0])
            lst[0] = mincfl
            lst[1] = maxcfl
            lst[2] = nadj
            lst[3] += nadj

class CflHook(Hook):
    """
    Makes sure CFL number is bounded and print averaged CFL number over time.
    Reports CFL information per time step and on finishing.  Implements (i)
    postmarch() and (ii) postloop() methods.

    @ivar name: name of the CFL tool.
    @itype name: str
    @ivar rsteps: steps to run.
    @itype rsteps: int
    @ivar cflmin: CFL number should be greater than or equal to the value.
    @itype cflmin: float
    @ivar cflmax: CFL number should be less than the value.
    @itype cflmax: float
    @ivar fullstop: flag to stop when CFL is out of bound.  Default True.
    @itype fullstop: bool
    @ivar aCFL: accumulated CFL.
    @itype aCFL: float
    @ivar mCFL: mean CFL.
    @itype mCFL: float
    @ivar hnCFL: hereditary minimal CFL.
    @itype hnCFL: float
    @ivar hxCFL: hereditary maximal CFL.
    @itype hxCFL: float
    @ivar aadj: number of adjusted CFL accumulated since last report.
    @itype aadj: int
    @ivar haadj: total number of adjusted CFL since simulation started.
    @itype haadj: int
    """
    def __init__(self, cse, **kw):
        self.name = kw.pop('name', 'cfl')
        self.cflmin = kw.pop('cflmin', 0.0)
        self.cflmax = kw.pop('cflmax', 1.0)
        self.fullstop = kw.pop('fullstop', True)
        self.aCFL = 0.0
        self.mCFL = 0.0
        self.hnCFL = 1.0
        self.hxCFL = 0.0
        self.aadj = 0
        self.haadj = 0
        rsteps = kw.pop('rsteps', None)
        super(CflHook, self).__init__(cse, **kw)
        self.rsteps = self.psteps if rsteps == None else rsteps
        self.ankkw = kw
    def drop_anchor(self, svr):
        ankkw = self.ankkw.copy()
        ankkw['name'] = self.name
        ankkw['rsteps'] = self.rsteps
        self._deliver_anchor(svr, CflAnchor, ankkw)
    def _notify(self, msg):
        from warnings import warn
        if self.fullstop:
            raise RuntimeError(msg)
        else:
            warn(msg)
    def postmarch(self):
        from numpy import isnan
        info = self.info
        istep = self.cse.execution.step_current
        mr = self.cse.execution.marchret
        isp = self.cse.is_parallel
        rsteps = self.rsteps
        psteps = self.psteps
        # collect CFL.
        if istep > 0 and istep%rsteps == 0:
            nCFL = max([m['cfl'][0] for m in mr]) if isp else mr['cfl'][0]
            xCFL = max([m['cfl'][1] for m in mr]) if isp else mr['cfl'][1]
            nadj = sum([m['cfl'][2] for m in mr]) if isp else mr['cfl'][2]
            aadj = sum([m['cfl'][3] for m in mr]) if isp else mr['cfl'][3]
            hnCFL = min([nCFL, self.hnCFL])
            self.hnCFL = hnCFL if not isnan(hnCFL) else self.hnCFL
            hxCFL = max([xCFL, self.hxCFL])
            self.hxCFL = hxCFL if not isnan(hxCFL) else self.hxCFL
            self.aCFL += xCFL*rsteps
            self.mCFL = self.aCFL/istep
            self.aadj += aadj
            self.haadj += aadj
            # check.
            if self.cflmin != None and nCFL < self.cflmin:
                self._notify("CFL = %g < %g after step: %d" % (
                    nCFL, self.cflmin, istep))
            if self.cflmax != None and xCFL >= self.cflmax:
                self._notify("CFL = %g >= %g after step: %d" % (
                    xCFL, self.cflmax, istep))
            # output information.
            if istep > 0 and istep%psteps == 0:
                info("CFL = %.2g/%.2g - %.2g/%.2g adjusted: %d/%d/%d\n" % (
                    nCFL, xCFL, self.hnCFL, self.hxCFL, nadj,
                    self.aadj, self.haadj))
                self.aadj = 0
    def postloop(self):
        self.info("Averaged maximum CFL = %g.\n" % self.mCFL)

################################################################################
# Convergence.
################################################################################

class ConvergeAnchor(Anchor):
    """
    Performs calculation for convergence on Solver.  Implements (i) preloop()
    and (ii) postfull() methods.

    @ivar rsteps: steps to run.
    @itype rsteps: int
    @ivar norm: container of calculated norm.
    @itype norm: dict
    """
    def __init__(self, svr, **kw):
        self.rsteps = kw.pop('rsteps')
        super(ConvergeAnchor, self).__init__(svr, **kw)
        self.norm = {}
    def preloop(self):
        from numpy import empty
        svr = self.svr
        der = svr.der
        der['diff'] = empty((svr.ngstcell+svr.ncell, svr.neq),
            dtype=svr.fpdtype)
    def postfull(self):
        from ctypes import byref, c_double
        svr = self.svr
        istep = svr.step_global
        rsteps = self.rsteps
        if istep > 0 and istep%rsteps == 0:
            diff = svr.der['diff']
            svr._clib_cuse_c.process_norm_diff(byref(svr.exd),
                diff[svr.ngstcell:].ctypes._as_parameter_)
            # Linf norm.
            Linf = []
            Linf.extend(diff.max(axis=0))
            self.norm['Linf'] = Linf
            # L1 norm.
            svr._clib_cuse_c.process_norm_L1.restype = c_double
            L1 = []
            for ieq in range(svr.neq):
                vals = svr._clib_cuse_c.process_norm_L1(byref(svr.exd),
                    diff[svr.ngstcell:].ctypes._as_parameter_, ieq)
                L1.append(vals)
            self.norm['L1'] = L1
            # L2 norm.
            svr._clib_cuse_c.process_norm_L2.restype = c_double
            L2 = []
            for ieq in range(svr.neq):
                vals = svr._clib_cuse_c.process_norm_L2(byref(svr.exd),
                    diff[svr.ngstcell:].ctypes._as_parameter_, ieq)
                L2.append(vals)
            self.norm['L2'] = L2

class ConvergeHook(BlockHook):
    """
    Initiates and controls the remote ConvergeAnchor.  Implements (i)
    drop_anchor() and (ii) postmarch() methods.

    @ivar name: name of the converge tool.
    @itype name: str
    @ivar keys: kinds of norms to output; Linf, L1, and L2.
    @itype keys: list
    @ivar eqs: indices of unknowns (associated with the equations).
    @itype eqs: list
    @ivar csteps: steps to collect; default to psteps.
    @itype csteps: int
    @ivar rsteps: steps to run (Anchor); default to csteps.
    @itype rsteps: int
    @ivar ankkw: hold the remaining keywords for Anchor.
    @itype ankkw: dict
    """
    def __init__(self, cse, **kw):
        self.name = kw.pop('name', 'converge')
        self.keys = kw.pop('keys', None)
        self.eqs = kw.pop('eqs', None)
        csteps = kw.pop('csteps', None)
        rsteps = kw.pop('rsteps', None)
        super(ConvergeHook, self).__init__(cse, **kw)
        self.csteps = self.psteps if csteps == None else csteps
        self.rsteps = self.csteps if rsteps == None else rsteps
        self.ankkw = kw
    def drop_anchor(self, svr):
        ankkw = self.ankkw.copy()
        ankkw['name'] = self.name
        ankkw['rsteps'] = self.rsteps
        self._deliver_anchor(svr, ConvergeAnchor, ankkw)
    def _collect(self):
        from numpy import sqrt
        cse = self.cse
        neq = cse.execution.neq
        if cse.is_parallel:
            dom = cse.solver.domainobj
            dealer = cse.solver.dealer
            allnorm = list()
            for iblk in range(dom.nblk):
                dealer[iblk].cmd.pullank(self.name, 'norm', with_worker=True)
                allnorm.append(dealer[iblk].recv())
            norm = {'Linf': [0.0]*neq, 'L1': [0.0]*neq, 'L2': [0.0]*neq}
            for ieq in range(neq):
                norm['Linf'][ieq] = max([nm['Linf'][ieq] for nm in allnorm])
                norm['L1'][ieq] = sum([nm['L1'][ieq] for nm in allnorm])
                norm['L2'][ieq] = sum([nm['L2'][ieq] for nm in allnorm])
        else:
            svr = self.cse.solver.solverobj
            norm = svr.runanchors[self.name].norm
        for ieq in range(neq):
            norm['L2'][ieq] = sqrt(norm['L2'][ieq])
        self.norm = norm
    def postmarch(self):
        info = self.info
        cse = self.cse
        istep = cse.execution.step_current
        csteps = self.csteps
        psteps = self.psteps
        if istep > 0 and istep%csteps == 0:
            self._collect()
        if istep > 0 and istep%psteps == 0:
            norm = self.norm
            keys = self.keys if self.keys != None else self.norm.keys()
            keys.sort()
            eqs = self.eqs if self.eqs != None else range(cse.execution.neq)
            for key in keys:
                info("Converge/%-4s [ %s ]:\n  %s\n" % (key,
                    ', '.join(['%d'%ieq for ieq in eqs]),
                    ' '.join(['%.4e'%norm[key][ieq] for ieq in eqs]),
                ))

################################################################################
# Probe.
################################################################################

class Probe(object):
    """
    Represent a point in the mesh.
    """
    def __init__(self, *args, **kw):
        from numpy import array
        self.speclst = kw.pop('speclst')
        self.name = kw.pop('name', None)
        self.crd = array(args, dtype='float64')
        self.pcl = -1
        self.vals = list()
    def __str__(self):
        crds = ','.join(['%g'%val for val in self.crd])
        return 'Pt/%s#%d(%s)%d' % (self.name, self.pcl, crds, len(self.vals))
    def locate_cell(self, svr):
        idx = svr.locate_point(*self.crd)
        self.pcl = idx[0]
    def __call__(self, svr, time):
        ngstcell = svr.ngstcell
        vlist = [time]
        for spec in self.speclst:
            arr = None
            if isinstance(spec, str):
                arr = svr.der[spec]
            elif isinstance(spec, int):
                if spec >= 0 and spec < svr.neq:
                    arr = svr.soln[:,spec]
                elif spec < 0 and -1-spec < svr.neq:
                    spec = -1-spec
                    arr = svr.sol[:,spec]
            if arr == None:
                raise IndexError, 'spec %s incorrect'%str(spec)
            vlist.append(arr[ngstcell+self.pcl])
        self.vals.append(vlist)

class ProbeAnchor(Anchor):
    """
    Anchor for probe.
    """
    def __init__(self, svr, **kw):
        speclst = kw.pop('speclst')
        self.points = list()
        for data in kw.pop('coords'):
            pkw = {'speclst': speclst, 'name': data[0]}
            self.points.append(Probe(*data[1:], **pkw))
        super(ProbeAnchor, self).__init__(svr, **kw)
    def preloop(self):
        for point in self.points: point.locate_cell(self.svr)
        for point in self.points: point(self.svr, self.svr.time)
    def postfull(self):
        for point in self.points: point(self.svr, self.svr.time)

class ProbeHook(BlockHook):
    """
    Point probe.
    """
    def __init__(self, cse, **kw):
        self.name = kw.pop('name', 'ppank')
        super(ProbeHook, self).__init__(cse, **kw)
        self.ankkw = kw
        self.points = None
    def drop_anchor(self, svr):
        ankkw = self.ankkw.copy()
        ankkw['name'] = self.name
        self._deliver_anchor(svr, ProbeAnchor, ankkw)
    def _collect(self):
        cse = self.cse
        if cse.is_parallel:
            dom = cse.solver.domainobj
            dealer = cse.solver.dealer
            allpoints = list()
            for iblk in range(dom.nblk):
                dealer[iblk].cmd.pullank(self.name, 'points', with_worker=True)
                allpoints.append(dealer[iblk].recv())
            npt = len(allpoints[0])
            points = [None]*npt
            for rpoints in allpoints:
                ipt = 0
                while ipt < npt:
                    if points[ipt] == None and rpoints[ipt].pcl >=0:
                        points[ipt] = rpoints[ipt]
                    ipt += 1
        else:
            svr = self.cse.solver.solverobj
            points = [pt for pt in svr.runanchors[self.name].points
                if pt.pcl >= 0]
        self.points = points
    def postmarch(self):
        psteps = self.psteps
        istep = self.cse.execution.step_current
        if istep%psteps != 0: return False
        self._collect()
        return True
    def postloop(self):
        import os
        from numpy import array, save
        for point in self.points:
            ptfn = '%s_pt_%s_%s.npy' % (
                self.cse.io.basefn, self.name, point.name)
            ptfn = os.path.join(self.cse.io.basedir, ptfn)
            save(ptfn, array(point.vals, dtype='float64'))
