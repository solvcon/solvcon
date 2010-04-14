import os
from unittest import TestCase
from ..testing import get_blk_from_sample_neu
from ..solver import BaseSolver, BlockSolver

class CustomBaseSolver(BaseSolver):
    def __init__(self, **kw):
        kw['neq'] = 1
        super(CustomBaseSolver, self).__init__(**kw)
    def bind(self):
        super(CustomBaseSolver, self).bind()
        self.val = 'bind'

class CustomBlockSolver(BlockSolver):
    MESG_FILENAME_DEFAULT = os.devnull

class TestBase(TestCase):
    def test_base(self):
        self.assertRaises(KeyError, BaseSolver)
        bsvr = BaseSolver(neq=1)
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
        from ..dependency import pointer_of, str_of
        from ..conf import env
        bsvr = BaseSolver(neq=1)
        self.assertEqual(bsvr.fpdtype, env.fpdtype)
        self.assertEqual(bsvr.fpdtypestr, str_of(env.fpdtype))
        self.assertEqual(bsvr.fpptr, pointer_of(env.fpdtype))

class TestBlock(TestCase):
    neq = 1

    @staticmethod
    def _get_block():
        return get_blk_from_sample_neu()

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
        from cStringIO import StringIO
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

    def test_metric(self):
        svr = self._get_solver()
        self.assertEqual(len(svr.cecnd.shape), 3)
        self.assertEqual(svr.cecnd.shape[0], svr.ncell+svr.ngstcell)
        self.assertEqual(svr.cecnd.shape[1], svr.CLMFC+1)
        self.assertEqual(svr.cecnd.shape[2], svr.ndim)
        self.assertEqual(len(svr.cevol.shape), 2)
        self.assertEqual(svr.cevol.shape[0], svr.ncell+svr.ngstcell)
        self.assertEqual(svr.cevol.shape[1], svr.CLMFC+1)

    def test_solution(self):
        svr = self._get_solver()
        self.assertEqual(len(svr.sol.shape), 2)
        self.assertEqual(svr.sol.shape[0], svr.soln.shape[0])
        self.assertEqual(svr.sol.shape[1], svr.soln.shape[1])
        self.assertEqual(svr.sol.shape[0], svr.ncell+svr.ngstcell)
        self.assertEqual(svr.sol.shape[1], svr.neq)
        self.assertEqual(len(svr.dsol.shape), 3)
        self.assertEqual(svr.dsol.shape[0], svr.dsoln.shape[0])
        self.assertEqual(svr.dsol.shape[1], svr.dsoln.shape[1])
        self.assertEqual(svr.dsol.shape[2], svr.dsoln.shape[2])
        self.assertEqual(svr.dsol.shape[0], svr.ncell+svr.ngstcell)
        self.assertEqual(svr.dsol.shape[1], svr.neq)
        self.assertEqual(svr.dsol.shape[2], svr.ndim)

    time = 0.0
    time_increment = 1.0
    nsteps = 10

    def _run_solver(self, time, time_increment, nsteps):
        # initialize.
        svr = self._get_solver()
        svr.soln.fill(0.0)
        svr.dsoln.fill(0.0)
        # run.
        svr.march(time, time_increment, nsteps)
        return svr

    def test_soln(self):
        from numpy import zeros
        # run.
        svr = self._run_solver(self.time, self.time_increment, self.nsteps)
        ngstcell = svr.ngstcell
        # get result.
        soln = svr.soln[ngstcell:,0]
        # calculate reference
        clvol = zeros(soln.shape, dtype=soln.dtype)
        for iistep in range(self.nsteps*2):
            clvol += svr.clvol[ngstcell:]*self.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())

    def test_dsoln(self):
        from numpy import zeros
        # run.
        svr = self._run_solver(self.time, self.time_increment, self.nsteps)
        ngstcell = svr.ngstcell
        # get result.
        dsoln = svr.dsoln[ngstcell:,0,:]
        # calculate reference
        clcnd = zeros(dsoln.shape, dtype=dsoln.dtype)
        for iistep in range(self.nsteps*2):
            clcnd += svr.clcnd[ngstcell:]*self.time_increment/2
        # compare.
        self.assertTrue((dsoln==clcnd).all())
