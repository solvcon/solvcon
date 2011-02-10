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
Second-order CESE method with CUDA enabled.
"""

from ctypes import Structure
from solvcon.gendata import AttributeDict
from solvcon.solver import BlockSolver
from solvcon.case import BlockCase
from solvcon.boundcond import BC, periodic
from solvcon.anchor import Anchor
from solvcon.hook import Hook, BlockHook

class CudaDataManager(AttributeDict):
    """
    Manage solver data on GPU memory.
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

    def arr_to_gpu(self, name=None):
        scu = self.svr.scu
        names = self if name is None else [name]
        for name in names:
            scu.memcpy(self[name], getattr(self.svr, name))
    def arr_from_gpu(self, name=None):
        scu = self.svr.scu
        names = self if name is None else [name]
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
        # function pointer.
        ('jacofunc', c_void_p), ('taufunc', c_void_p), ('omegafunc', c_void_p),
        # scheme.
        ('alpha', c_int), ('taylor', c_double),
        ('cnbfac', c_double), ('sftfac', c_double),
        ('taumin', c_double), ('taumax', c_double), ('tauscale', c_double),
        ('omegamin', c_double), ('omegascale', c_double),
        # meta array.
        ('fctpn', c_void_p),
        ('cltpn', c_void_p), ('clgrp', c_void_p),
        ('grpda', c_void_p),
        # metric array.
        ('ndcrd', c_void_p),
        ('fccnd', c_void_p), ('fcnml', c_void_p),
        ('clcnd', c_void_p), ('clvol', c_void_p),
        ('cecnd', c_void_p), ('cevol', c_void_p),
        # connectivity array.
        ('fcnds', c_void_p), ('fccls', c_void_p),
        ('clnds', c_void_p), ('clfcs', c_void_p),
        # solution array.
        ('sol', c_void_p), ('dsol', c_void_p),
        ('solt', c_void_p),
        ('soln', c_void_p), ('dsoln', c_void_p),
        ('cfl', c_void_p), ('ocfl', c_void_p),
        ('amsca', c_void_p), ('amvec', c_void_p),
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
        # inherited.
        for key in ('ncore', 'neq', 'time', 'time_increment',):
            setattr(self, key, getattr(svr, key))
        # mesh shape.
        for key in ('ndim', 'nnode', 'nface', 'ncell', 'nbound',
                    'ngstnode', 'ngstface', 'ngstcell',):
            setattr(self, key, getattr(svr, key))
        # group shape.
        for key in ('ngroup', 'gdlen',):
            setattr(self, key, getattr(svr, key))
        # parameter shape.
        for key in ('nsca', 'nvec',):
            setattr(self, key, getattr(svr, key))
        # function pointer.
        self.jacofunc = cast(svr._jacofunc_, c_void_p).value
        self.taufunc = cast(getattr(svr._clib_cuse_c,
            'tau_'+svr.tauname), c_void_p).value
        self.omegafunc = cast(getattr(svr._clib_cuse_c,
            'omega_'+svr.omeganame), c_void_p).value
        # scheme. 
        for key in ('alpha', 'taylor', 'cnbfac', 'sftfac',
                    'taumin', 'taumax', 'tauscale', 'omegamin', 'omegascale',):
            setattr(self, key, getattr(svr, key))
        # meta array.
        for aname in ('fctpn',):
            self.__set_pointer(svr, aname, svr.ngstface)
        for aname in ('cltpn', 'clgrp',):
            self.__set_pointer(svr, aname, svr.ngstcell)
        for aname in ('grpda',):
            self.__set_pointer(svr, aname, 0)
        # metric array.
        for aname in ('ndcrd',):
            self.__set_pointer(svr, aname, svr.ngstnode)
        for aname in ('fccnd', 'fcnml',):
            self.__set_pointer(svr, aname, svr.ngstface)
        for aname in ('clcnd', 'clvol', 'cecnd', 'cevol',):
            self.__set_pointer(svr, aname, svr.ngstcell)
        # connectivity array.
        for aname in ('fcnds', 'fccls',):
            self.__set_pointer(svr, aname, svr.ngstface)
        for aname in ('clnds', 'clfcs',):
            self.__set_pointer(svr, aname, svr.ngstcell)
        # solution array.
        for aname in ('sol', 'dsol', 'solt', 'soln', 'dsoln',):
            self.__set_pointer(svr, aname, svr.ngstcell)
        for aname in ('cfl', 'ocfl'):
            self.__set_pointer(svr, aname, 0)
        for aname in ('amsca', 'amvec'):
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

    @ivar debug: flag for debugging.
    @itype debug: bool

    @ivar scu: CUDA wrapper.
    @itype scu: solvcon.scuda.Scuda
    @ivar ncuthread: number of thread per block for CUDA.
    @itype ncuthread: int

    @ivar diffname: name of gradient calculation function; tau is default,
        omega is selectable.
    @itype diffname: str
    @ivar tauname: name of tau function; default linear.
    @itype tauname: str
    @ivar omeganame: name of omega function; default scale.
    @itype omeganame: str
    @ivar alpha: parameter to the weighting function.
    @itype alpha: int
    @ivar taylor: factor for Taylor's expansion; 0 off, 1 on.
    @itype taylor: float
    @ivar cnbfac: factor to use BCE centroid, othersize midpoint; 0 off, 1 on.
    @itype cnbfac: float
    @ivar sftfac: factor to shift gradient shape; 0 off, 1 on.
    @itype sftfac: float
    @ivar taumin: the lower bound of tau.
    @itype taumin: float
    @ivar taumax: the upper bound of tau.
    @itype taumax: float
    @ivar tauscale: scaling of tau.
    @itype tauscale: float
    @ivar omegamin: the lower bound of omega.
    @itype omegamin: float
    @ivar omegascale: scaling of omega.
    @itype omegascale: float

    @ivar grpda: group data.
    @ivar cecnd: solution points for CCEs and BCEs.
    @ivar cevol: CCE and BCE volumes.
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

    def __init__(self, blk, *args, **kw):
        from numpy import empty
        from solvcon.conf import env
        self.debug = kw.pop('debug', False)
        # shape parameters.
        nsca = kw.pop('nsca', 0)
        nvec = kw.pop('nvec', 0)
        # CUDA parameters.
        self.ncuthread = kw.pop('ncuthread',  0)
        self.scu = scu = env.scu if self.ncuthread else None
        # scheme parameters.
        diffname = kw.pop('diffname', None)
        self.diffname = diffname if diffname != None else 'tau'
        tauname = kw.pop('tauname', None)
        self.tauname = tauname if tauname != None else 'linear'
        omeganame = kw.pop('omeganame', None)
        self.omeganame = omeganame if omeganame != None else 'scale'
        self.alpha = int(kw.pop('alpha', 0))
        self.taylor = float(kw.pop('taylor', 1.0))
        self.cnbfac = float(kw.pop('cnbfac', 1.0))
        self.sftfac = float(kw.pop('sftfac', 1.0))
        self.taumin = float(kw.pop('taumin', 0.0))
        self.taumax = float(kw.pop('taumax', 1.0))
        self.tauscale = float(kw.pop('tauscale', 0.0))
        self.omegamin = float(kw.pop('omegamin', 1.1))
        self.omegascale = float(kw.pop('omegascale', 0.0))
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
        for key in ('clfcs', 'clcnd', 'cltpn'):
            self.cuarr_map[key] = self.ngstcell
        for key in ('fcnds', 'fccls', 'fccnd', 'fcnml'):
            self.cuarr_map[key] = self.ngstface
        for key in ('ndcrd',):
            self.cuarr_map[key] = self.ngstnode
        # meta array.
        self.grpda = empty((ngroup, self._gdlen_), dtype=fpdtype)
        self.cuarr_map['grpda'] = 0
        # dual mesh.
        self.cecnd = empty((ngstcell+ncell, self.CLMFC+1, ndim), dtype=fpdtype)
        self.cevol = empty((ngstcell+ncell, self.CLMFC+1), dtype=fpdtype)
        self.cuarr_map['cecnd'] = self.cuarr_map['cevol'] = self.ngstcell
        # solutions.
        self.amsca = empty((ngstcell+ncell, nsca), dtype=fpdtype)
        self.amvec = empty((ngstcell+ncell, nvec, ndim), dtype=fpdtype)
        self.solt = empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.sol = empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.soln = empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.dsol = empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        self.dsoln = empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        for key in ('amsca', 'amvec', 'solt', 'sol', 'soln', 'dsol', 'dsoln'):
            self.cuarr_map[key] = self.ngstcell
        self.cfl = empty(ncell, dtype=fpdtype)
        self.ocfl = empty(ncell, dtype=fpdtype)
        self.cuarr_map['cfl'] = self.cuarr_map['ocfl'] = 0

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
        if self.scu:
            self.cumgr = CudaDataManager(svr=self)
        super(CuseSolver, self).bind()
    def unbind(self):
        if self.scu and self.cumgr:
            self.cumgr.free_all()
            self.cumgr = None
        super(CuseSolver, self).unbind()

    def init(self, **kw):
        self._tcall(self._clib_cuse_c.prepare_ce, 0, self.ncell)
        super(CuseSolver, self).init(**kw)
        if self.scu: self.cumgr.arr_to_gpu()

    def boundcond(self):
        if self.scu:
            self.cumgr.update_exd()
            for key in ('sol', 'soln', 'dsol', 'dsoln'):
                self.cumgr.arr_to_gpu(key)
            if self.nsca: self.cumgr.arr_to_gpu('amsca')
            if self.nvec: self.cumgr.arr_to_gpu('amvec')
        super(CuseSolver, self).boundcond()
        for bc in self.bclist: bc.soln()
        for bc in self.bclist: bc.dsoln()

    ###########################################################################
    # utility.
    ###########################################################################
    def locate_point(self, *args):
        """
        Locate the cell index where the input coordinate is.
        """
        from ctypes import byref
        from numpy import array
        crd = array(args, dtype=self.fpdtype)   # FIXME: could be type error.
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
        2: getcdll('cuse2d_c'),
        3: getcdll('cuse3d_c'),
    }
    __clib_cuse_cu = {
        2: getcdll('cuse2d_cu'),
        3: getcdll('cuse3d_cu'),
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
        # reset pointers in execution data.
        ngstcell = self.ngstcell
        self.exd.sol = self.sol[ngstcell:].ctypes._as_parameter_
        self.exd.soln = self.soln[ngstcell:].ctypes._as_parameter_
        self.exd.dsol = self.dsol[ngstcell:].ctypes._as_parameter_
        self.exd.dsoln = self.dsoln[ngstcell:].ctypes._as_parameter_
        # reset GPU execution data.
        if self.scu:
            cumgr = self.cumgr
            cumgr.sol, cumgr.soln = cumgr.soln, cumgr.sol
            cumgr.dsol, cumgr.dsoln = cumgr.dsoln, cumgr.dsol
            cumgr.update_exd()
        if self.debug: self.mesg(' done.\n')

    MMNAMES.append('ibcam')
    def ibcam(self, worker=None):
        if self.debug: self.mesg('ibcam')
        if worker:  # FIXME: not working with CUDA.
            if self.nsca: self.exchangeibc('amsca', worker=worker)
            if self.nvec: self.exchangeibc('amvec', worker=worker)
        if self.debug: self.mesg(' done.\n')

    MMNAMES.append('calcsolt')
    def calcsolt(self, worker=None):
        if self.debug: self.mesg('calcsolt')
        if self.scu:
            from ctypes import byref
            self._clib_cuse_cu.calc_solt(self.ncuthread,
                byref(self.cumgr.exd), self.cumgr.gexd.gptr)
        else:
            self._tcall(self._clib_cuse_c.calc_solt, -self.ngstcell, self.ncell,
                tickerkey='calcsolt')
        if self.debug: self.mesg(' done.\n')
    MMNAMES.append('calcsoln')
    def calcsoln(self, worker=None):
        if self.debug: self.mesg('calcsoln')
        if self.scu:
            from ctypes import byref
            self._clib_cuse_cu.calc_soln(self.ncuthread,
                byref(self.cumgr.exd), self.cumgr.gexd.gptr)
            self.cumgr.arr_from_gpu('soln')
        else:
            func = self._clib_cuse_c.calc_soln
            self._tcall(func, 0, self.ncell, tickerkey='calcsoln')
        if self.debug: self.mesg(' done.\n')

    MMNAMES.append('ibcsoln')
    def ibcsoln(self, worker=None):
        if self.debug: self.mesg('ibcsoln')
        if worker: self.exchangeibc('soln', worker=worker)
        if self.debug: self.mesg(' done.\n')
    MMNAMES.append('bcsoln')
    def bcsoln(self, worker=None):
        if self.debug: self.mesg('bcsoln')
        for bc in self.bclist: bc.soln()
        if self.debug: self.mesg(' done.\n')

    MMNAMES.append('calccfl')
    def calccfl(self, worker=None):
        """
        @return: mincfl, maxcfl, number of tuned CFL, accumulated number of
            tuned CFL.
        @rtype: tuple
        """
        raise NotImplementedError

    MMNAMES.append('calcdsoln')
    def calcdsoln(self, worker=None):
        if self.debug: self.mesg('calcdsoln')
        if self.scu:
            from ctypes import byref
            self._clib_cuse_cu.calc_dsoln(self.ncuthread,
                byref(self.cumgr.exd), self.cumgr.gexd.gptr)
            self.cumgr.arr_from_gpu('dsoln')
        else:
            func = self._clib_cuse_c.calc_dsoln
            self._tcall(func, 0, self.ncell, tickerkey='calcdsoln')
        if self.debug: self.mesg(' done.\n')

    MMNAMES.append('ibcdsoln')
    def ibcdsoln(self, worker=None):
        if self.debug: self.mesg('ibcdsoln')
        if worker: self.exchangeibc('dsoln', worker=worker)
        if self.debug: self.mesg(' done.\n')
    MMNAMES.append('bcdsoln')
    def bcdsoln(self, worker=None):
        if self.debug: self.mesg('bcdsoln')
        for bc in self.bclist: bc.dsoln()
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
        'solver.diffname': None,
        'solver.tauname': None,
        'solver.omeganame': None,
        'solver.alpha': 1,
        'solver.taylor': 1.0,
        'solver.cnbfac': 1.0,
        'solver.sftfac': 1.0,
        'solver.taumin': None,
        'solver.taumax': None,
        'solver.tauscale': None,
        'solver.omegamin': None,
        'solver.omegascale': None,
    }
    def make_solver_keywords(self):
        kw = super(CuseCase, self).make_solver_keywords()
        kw['debug'] = self.solver.debug_cese
        for key in 'diffname', 'tauname', 'omeganame':
            val = self.solver.get(key)
            if val != None: kw[key] = val
        kw['alpha'] = int(self.solver.alpha)
        for key in ('taylor', 'cnbfac', 'sftfac',
                    'taumin', 'taumax', 'tauscale', 'omegamin', 'omegascale',):
            val = self.solver.get(key)
            if val != None: kw[key] = float(val)
        return kw

###############################################################################
# Boundary conditions.
###############################################################################

class CuseBC(BC):
    """
    Basic BC class for the Euler equations.

    @cvar _ghostgeom_: indicate which ghost geometry processor to use.
    @ctype _ghostgeom_: str
    """

    _ghostgeom_ = None

    ###########################################################################
    # library.
    ###########################################################################
    from solvcon.dependency import getcdll
    __clib_cuseb_c = {
        2: getcdll('cuseb2d_c'),
        3: getcdll('cuseb3d_c'),
    }
    __clib_cuseb_cu = {
        2: getcdll('cuseb2d_cu'),
        3: getcdll('cuseb3d_cu'),
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
        if self.svr.scu:
            self._clib_cuseb_cu.bound_nonrefl_soln(svr.ncuthread,
                svr.cumgr.gexd.gptr, self.facn.shape[0], self.cufacn.gptr)
        else:
            self._clib_cuseb_c.bound_nonrefl_soln(byref(svr.exd),
                self.facn.shape[0], self.facn.ctypes._as_parameter_)

    def dsoln(self):
        from ctypes import byref
        svr = self.svr
        if self.svr.scu:
            self._clib_cuseb_cu.bound_nonrefl_dsoln(svr.ncuthread,
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
        svr.cumgr.arr_from_gpu('soln')
        svr.soln[slctm,:] = svr.soln[slctr,:]
        svr.cumgr.arr_to_gpu('soln')
    def dsoln(self):
        svr = self.svr
        slctm = self.rclp[:,0] + svr.ngstcell
        slctr = self.rclp[:,1] + svr.ngstcell
        svr.cumgr.arr_from_gpu('dsoln')
        svr.dsoln[slctm,:,:] = svr.dsoln[slctr,:,:]
        svr.cumgr.arr_to_gpu('dsoln')

################################################################################
# Anchors.
################################################################################

class ConvergeAnchor(Anchor):
    def __init__(self, svr, **kw):
        self.norm = {}
        super(ConvergeAnchor, self).__init__(svr, **kw)
    def preloop(self):
        from numpy import empty
        svr = self.svr
        der = svr.der
        der['diff'] = empty((svr.ngstcell+svr.ncell, svr.neq),
            dtype=svr.fpdtype)
    def postfull(self):
        from ctypes import c_int, c_double
        svr = self.svr
        diff = svr.der['diff']
        svr._tcall(svr._clib_cuse_c.process_norm_diff, -svr.ngstcell,
            svr.ncell, diff[svr.ngstcell:].ctypes._as_parameter_)
        # Linf norm.
        Linf = []
        Linf.extend(diff.max(axis=0))
        self.norm['Linf'] = Linf
        # L1 norm.
        svr._clib_cuse_c.process_norm_L1.restype = c_double
        L1 = []
        for ieq in range(svr.neq):
            vals = svr._tcall(svr._clib_cuse_c.process_norm_L1, 0, svr.ncell,
                diff[svr.ngstcell:].ctypes._as_parameter_, c_int(ieq))
            L1.append(sum(vals))
        self.norm['L1'] = L1
        # L2 norm.
        svr._clib_cuse_c.process_norm_L2.restype = c_double
        L2 = []
        for ieq in range(svr.neq):
            vals = svr._tcall(svr._clib_cuse_c.process_norm_L2, 0, svr.ncell,
                diff[svr.ngstcell:].ctypes._as_parameter_, c_int(ieq))
            L2.append(sum(vals))
        self.norm['L2'] = L2

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

################################################################################
# Hooks.
################################################################################

class CflHook(Hook):
    """
    Make sure is CFL number is bounded and print averaged CFL number over time.

    @ivar cflmin: CFL number should be greater than or equal to the value.
    @itype cflmin: float
    @ivar cflmax: CFL number should be less than the value.
    @itype cflmax: float
    @ivar fullstop: flag to stop when CFL is out of bound.  Default True.
    @itype fullstop: bool
    """
    def __init__(self, cse, **kw):
        self.cflmin = kw.pop('cflmin', 0.0)
        self.cflmax = kw.pop('cflmax', 1.0)
        self.fullstop = kw.pop('fullstop', True)
        self.aCFL = 0.0
        self.mCFL = 0.0
        self.hnCFL = 1.0
        self.hxCFL = 0.0
        self.aadj = 0
        self.haadj = 0
        super(CflHook, self).__init__(cse, **kw)
    def _notify(self, msg):
        from warnings import warn
        if self.fullstop:
            raise RuntimeError, msg
        else:
            warn(msg)
    def postmarch(self):
        from numpy import isnan
        info = self.info
        steps_stride = self.cse.execution.steps_stride
        istep = self.cse.execution.step_current
        marchret = self.cse.execution.marchret
        is_parallel = self.cse.is_parallel
        psteps = self.psteps
        # collect CFL.
        nCFL = max([m[0] for m in marchret]) if is_parallel else marchret[0]
        xCFL = max([m[1] for m in marchret]) if is_parallel else marchret[1]
        nadj = sum([m[2] for m in marchret]) if is_parallel else marchret[2]
        aadj = sum([m[3] for m in marchret]) if is_parallel else marchret[2]
        hnCFL = min([nCFL, self.hnCFL])
        self.hnCFL = hnCFL if not isnan(hnCFL) else self.hnCFL
        hxCFL = max([xCFL, self.hxCFL])
        self.hxCFL = hxCFL if not isnan(hxCFL) else self.hxCFL
        self.aCFL += xCFL*steps_stride
        self.mCFL = self.aCFL/(istep+steps_stride)
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
            info("CFL = %.2f/%.2f - %.2f/%.2f adjusted: %d/%d/%d\n" % (
                nCFL, xCFL, self.hnCFL, self.hxCFL, nadj,
                self.aadj, self.haadj))
            self.aadj = 0

    def postloop(self):
        self.info("Averaged maximum CFL = %g.\n" % self.mCFL)

class ConvergeHook(BlockHook):
    def __init__(self, cse, **kw):
        self.name = kw.pop('name', 'converge')
        self.keys = kw.pop('keys', None)
        self.eqs = kw.pop('eqs', None)
        csteps = kw.pop('csteps', None)
        super(ConvergeHook, self).__init__(cse, **kw)
        self.csteps = self.psteps if csteps == None else cstep
        self.ankkw = kw
    def drop_anchor(self, svr):
        ankkw = self.ankkw.copy()
        ankkw['name'] = self.name
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
