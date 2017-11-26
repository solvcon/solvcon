# -*- coding: UTF-8 -*-


from __future__ import absolute_import, division, print_function


import os
from unittest import TestCase

import numpy as np

import solvcon as sc
from .. import solver
from .. import testing
from ..parcel.fake._algorithm import FakeAlgorithm


class CustomBaseSolver(solver.BaseSolver):
    def __init__(self, **kw):
        kw['neq'] = 1
        super(CustomBaseSolver, self).__init__(**kw)
    def bind(self):
        super(CustomBaseSolver, self).bind()
        self.val = 'bind'

class CustomBlockSolver(solver.BlockSolver):
    MESG_FILENAME_DEFAULT = os.devnull

    _interface_init_ = ['cecnd', 'cevol']

    def __init__(self, blk, *args, **kw):
        """
        @keyword neq: number of equations (variables).
        @type neq: int
        """
        from numpy import empty
        super(CustomBlockSolver, self).__init__(blk, *args, **kw)
        # data structure for C/FORTRAN.
        self.blk = blk
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

    def create_alg(self):
        alg = FakeAlgorithm()
        alg.setup_mesh(self.blk)
        alg.setup_algorithm(self)
        return alg

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
        self.create_alg().calc_soln()

    MMNAMES.append('ibcsoln')
    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    MMNAMES.append('calccfl')
    def calccfl(self, worker=None):
        self.marchret = -2.0

    MMNAMES.append('calcdsoln')
    def calcdsoln(self, worker=None):
        self.create_alg().calc_dsoln()

    MMNAMES.append('ibcdsoln')
    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)

class TestBase(TestCase):
    def test_base(self):
        self.assertRaises(KeyError, solver.BaseSolver)
        bsvr = solver.BaseSolver(neq=1)
        self.assertEqual(getattr(bsvr, 'val', None), None)
        bsvr.bind()
        self.assertEqual(getattr(bsvr, 'val', None), None)

    def test_inheritance(self):
        svr = CustomBaseSolver()
        self.assertEqual(getattr(svr, 'val', None), None)
        svr.bind()
        self.assertEqual(svr.val, 'bind')

class TestFpdtype(TestCase):
    def test_fp(self):
        from ..dependency import str_of
        from ..conf import env
        bsvr = solver.BaseSolver(neq=1)
        self.assertEqual(bsvr.fpdtype, env.fpdtype)
        self.assertEqual(bsvr.fpdtypestr, str_of(env.fpdtype))

class TestBlock(TestCase):
    neq = 1

    def test_simplex(self):
        svr = CustomBlockSolver(testing.get_blk_from_oblique_neu(),
            neq=self.neq, enable_mesg=True)
        self.assertTrue(svr.all_simplex)
        svr = CustomBlockSolver(testing.get_blk_from_sample_neu(),
            neq=self.neq, enable_mesg=True)
        self.assertFalse(svr.all_simplex)

    def test_incenter(self):
        svr = CustomBlockSolver(testing.get_blk_from_oblique_neu(use_incenter=True),
            neq=self.neq, enable_mesg=True)
        self.assertTrue(svr.use_incenter)
        svr = CustomBlockSolver(testing.get_blk_from_sample_neu(use_incenter=False),
            neq=self.neq, enable_mesg=True)
        self.assertFalse(svr.use_incenter)

    @staticmethod
    def _get_block():
        return testing.get_blk_from_sample_neu()

    @classmethod
    def _get_solver(cls, init=True):
        import warnings
        svr = CustomBlockSolver(cls._get_block(), neq=cls.neq, enable_mesg=True)
        if init:
            warnings.simplefilter("ignore")
            svr.bind()
            svr.init()
            warnings.resetwarnings()
        return svr

    def test_debug(self):
        import sys
        from io import StringIO
        CustomBlockSolver.MESG_FILENAME_DEFAULT = 'sys.stdout'
        stdout = sys.stdout
        sys.stdout = StringIO()
        svr = self._get_solver(init=True)
        svr.mesg('test message')
        self.assertEqual(sys.stdout.getvalue(), 'test message')
        sys.stdout = stdout
        CustomBlockSolver.MESG_FILENAME_DEFAULT = os.devnull

    def test_create(self):
        svr = self._get_solver()
        self.assertTrue(svr)

    def test_bound_full(self):
        svr = self._get_solver()
        svr.bind()
        self.assertTrue(svr.is_bound)
        self.assertFalse(svr.is_unbound)

    def test_unbound_full(self):
        svr = self._get_solver()
        svr.unbind()
        self.assertFalse(svr.is_bound)
        self.assertTrue(svr.is_unbound)

    def test_neq(self):
        svr = self._get_solver()
        self.assertEqual(svr.neq, self.neq)

    def test_blkn(self):
        svr = self._get_solver()
        self.assertEqual(svr.svrn, None)


class CustomMeshSolver(solver.MeshSolver):

    MESG_FILENAME_DEFAULT = os.devnull

    _interface_init_ = ['cecnd', 'cevol']

    def __init__(self, blk, *args, **kw):
        neq = kw.pop("neq")
        super(CustomMeshSolver, self).__init__(blk, *args, **kw)
        # data structure for C/FORTRAN.
        self.blk = blk
        # arrays.
        ndim = self.ndim
        ncell = self.ncell
        ngstcell = self.ngstcell
        ## solutions.
        fpdtype = "float64"
        self.neq = neq
        self.sol = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.soln = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        self.dsol = np.empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        self.dsoln = np.empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        ## metrics.
        self.cecnd = np.empty(
            (ngstcell+ncell, sc.Block.CLMFC+1, ndim), dtype=fpdtype)
        self.cevol = np.empty(
            (ngstcell+ncell, sc.Block.CLMFC+1), dtype=fpdtype)

    def create_alg(self):
        alg = FakeAlgorithm()
        alg.setup_mesh(self.blk)
        alg.setup_algorithm(self)
        return alg

    ##################################################
    # Time marchers
    ##################################################
    @solver.MeshSolver.register_marcher
    def update(self, worker=None):
        self.sol[:,:] = self.soln[:,:]
        self.dsol[:,:,:] = self.dsoln[:,:,:]

    @solver.MeshSolver.register_marcher
    def calcsoln(self, worker=None):
        self.create_alg().calc_soln()

    @solver.MeshSolver.register_marcher
    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    @solver.MeshSolver.register_marcher
    def calccfl(self, worker=None):
        self.marchret = -2.0

    @solver.MeshSolver.register_marcher
    def calcdsoln(self, worker=None):
        self.create_alg().calc_dsoln()

    @solver.MeshSolver.register_marcher
    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)


class TestMeshSolverBlock(TestCase):
    neq = 1

    def test_MMNAMES(self):
        svr = CustomMeshSolver(
            testing.get_blk_from_oblique_neu(use_incenter=True),
            neq=self.neq, enable_mesg=True)
        self.assertEqual(
            ('update', 'calcsoln', 'ibcsoln', 'calccfl', 'calcdsoln',
             'ibcdsoln'),
            svr._MMNAMES)

    def test_mmnames(self):
        svr = CustomMeshSolver(
            testing.get_blk_from_oblique_neu(use_incenter=True),
            neq=self.neq, enable_mesg=True)
        self.assertEqual(
            ('update', 'calcsoln', 'ibcsoln', 'calccfl', 'calcdsoln',
             'ibcdsoln'),
            tuple(svr.mmnames))
