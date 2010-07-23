# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Provide functionalities for unittests.
"""

import os
from unittest import TestCase
from .solver import BlockSolver

def loadfile(filename):
    """
    Load file with requested file name.  The file name contains relative path
    to 'data' directory in this directory, and uses forward slash as delimiter 
    of directory components.

    @param filename: path of file relative to 'data' directory in this directory.
    @type filename: str
    @return: loaded data.
    @rtype: str
    """
    import os
    from .conf import env
    path = [env.datadir] + filename.split('/')
    path = os.path.join(*path)
    return open(path).read()

def get_blk_from_sample_neu(fpdtype=None):
    """
    Read data from sample.neu file and convert it into Block.
    """
    from .io.gambit import GambitNeutral
    from .boundcond import bctregy
    return GambitNeutral(loadfile('sample.neu')).toblock(fpdtype=fpdtype)

def get_blk_from_oblique_neu(fpdtype=None):
    """
    Read data from oblique.neu file and convert it into Block.
    """
    from .io.gambit import GambitNeutral
    from .boundcond import bctregy
    bcname_mapper = {
        'inlet': (bctregy.unspecified, {}),
        'outlet': (bctregy.unspecified, {}),
        'wall': (bctregy.unspecified, {}),
        'farfield': (bctregy.unspecified, {}),
    }
    return GambitNeutral(loadfile('oblique.neu')
        ).toblock(bcname_mapper=bcname_mapper, fpdtype=fpdtype)

class TestingSolver(BlockSolver):
    _pointers_ = ['msh']

    _interface_init_ = ['cecnd', 'cevol']

    def __init__(self, blk, *args, **kw):
        """
        @keyword neq: number of equations (variables).
        @type neq: int
        """
        from numpy import empty
        super(TestingSolver, self).__init__(blk, *args, **kw)
        # data structure for C/FORTRAN.
        self.msh = None
        # arrays.
        ndim = self.ndim
        ncell = self.ncell
        ngstcell = self.ngstcell
        ## solutions.
        neq = self.neq
        self.sol = empty((ngstcell+ncell, neq), dtype=self.fpdtype)
        self.soln = empty((ngstcell+ncell, neq), dtype=self.fpdtype)
        self.dsol = empty((ngstcell+ncell, neq, ndim), dtype=self.fpdtype)
        self.dsoln = empty((ngstcell+ncell, neq, ndim), dtype=self.fpdtype)
        ## metrics.
        self.cecnd = empty(
            (ngstcell+ncell, self.CLMFC+1, ndim), dtype=self.fpdtype)
        self.cevol = empty((ngstcell+ncell, self.CLMFC+1), dtype=self.fpdtype)

    def bind(self):
        """
        Bind all the boundary condition objects.

        @note: BC must be bound AFTER solver "pointers".  Overridders to the
            method should firstly bind all pointers, secondly super binder, and 
            then methods/subroutines.
        """
        from .block import BlockShape, MeshData
        super(TestingSolver, self).bind()
        # structures.
        self.msd = MeshData(blk=self)
        self.msh = BlockShape(
            ndim=self.ndim,
            fcmnd=self.FCMND, clmnd=self.CLMND, clmfc=self.CLMFC,
            nnode=self.nnode, nface=self.nface, ncell=self.ncell,
            nbound=self.nbound,
            ngstnode=self.ngstnode, ngstface=self.ngstface,
            ngstcell=self.ngstcell,
        )

    ##################################################
    # marching algorithm.
    ##################################################
    MMNAMES = list()
    MMNAMES.append('update')
    def update(self, worker=None):
        self.sol[:,:] = self.soln[:,:]
        self.dsol[:,:,:] = self.dsoln[:,:,:]

    MMNAMES.append('calcsoln')
    def calcsoln(self, worker=None):
        from ctypes import byref
        fpptr = self.fpptr
        self._clib_solvcon.calc_soln(
            byref(self.msd),
            byref(self.exd),
            self.clvol.ctypes.data_as(fpptr),
            self.sol.ctypes.data_as(fpptr),
            self.soln.ctypes.data_as(fpptr),
        )
        #self._clib_solvcon.calc_soln_(
        #    byref(self.msh),
        #    byref(self.exd),
        #    self.clvol.ctypes.data_as(fpptr),
        #    self.sol.ctypes.data_as(fpptr),
        #    self.soln.ctypes.data_as(fpptr),
        #)

    MMNAMES.append('ibcsol')
    def ibcsol(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)
    MMNAMES.append('bcsol')
    def bcsol(self, worker=None):
        for bc in self.bclist: bc.sol()

    MMNAMES.append('calccfl')
    def calccfl(self, worker=None):
        self.marchret = -2.0

    MMNAMES.append('calcdsoln')
    def calcdsoln(self, worker=None):
        from ctypes import byref
        fpptr = self.fpptr
        self._clib_solvcon.calc_dsoln(
            byref(self.msd),
            byref(self.exd),
            self.clcnd.ctypes.data_as(fpptr),
            self.dsol.ctypes.data_as(fpptr),
            self.dsoln.ctypes.data_as(fpptr),
        )
        #self._clib_solvcon.calc_dsoln_(
        #    byref(self.msh),
        #    byref(self.exd),
        #    self.clcnd.ctypes.data_as(fpptr),
        #    self.dsol.ctypes.data_as(fpptr),
        #    self.dsoln.ctypes.data_as(fpptr),
        #)

    MMNAMES.append('ibcdsol')
    def ibcdsol(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)
    MMNAMES.append('bcdsol')
    def bcdsol(self, worker=None):
        for bc in self.bclist: bc.dsol()
