from unittest import TestCase

class TestMultidim(TestCase):
    def test_import(self):
        from ..multidim import BlockCase
        case = BlockCase()

    def test_init(self):
        from ..multidim import BlockCase
        case = BlockCase()
        self.assertFalse(case._have_init)
        self.assertRaises(TypeError, case.init)

class TestHook(TestCase):
    def test_existence(self):
        from .. import multidim
        self.assertTrue(multidim.BlockHook)
        self.assertTrue(multidim.BlockInfoHook)
        self.assertTrue(multidim.Initializer)
        self.assertTrue(multidim.Calculator)
        self.assertTrue(multidim.VtkSave)
        self.assertTrue(multidim.SplitSave)
        self.assertTrue(multidim.MarchSave)

    def test_type(self):
        from .. import multidim
        self.assertRaises(AssertionError, multidim.BlockHook, None)

    def test_blockhook(self):
        from ..core import Hook
        from ..multidim import BlockHook
        self.assertEqual(BlockHook.preloop, Hook.preloop)
        self.assertEqual(BlockHook.premarch, Hook.premarch)
        self.assertEqual(BlockHook.postmarch, Hook.postmarch)
        self.assertEqual(BlockHook.postloop, Hook.postloop)

    def test_blockinfohook(self):
        from ..core import Hook
        from ..multidim import BlockInfoHook
        self.assertNotEqual(BlockInfoHook.preloop, Hook.preloop)
        self.assertEqual(BlockInfoHook.premarch, Hook.premarch)
        self.assertEqual(BlockInfoHook.postmarch, Hook.postmarch)
        self.assertNotEqual(BlockInfoHook.postloop, Hook.postloop)

    def test_vtksave(self):
        from ..core import Hook
        from ..multidim import BlockCase, VtkSave
        case = BlockCase()
        hook = VtkSave(case)
        self.assertEqual(hook.binary, False)
        self.assertEqual(hook.cache_grid, True)
        self.assertEqual(VtkSave.preloop, Hook.preloop)
        self.assertEqual(VtkSave.premarch, Hook.premarch)
        self.assertEqual(VtkSave.postmarch, Hook.postmarch)
        self.assertEqual(VtkSave.postloop, Hook.postloop)

    def test_splitsave(self):
        from ..core import Hook
        from ..multidim import SplitSave
        self.assertNotEqual(SplitSave.preloop, Hook.preloop)
        self.assertEqual(SplitSave.premarch, Hook.premarch)
        self.assertEqual(SplitSave.postmarch, Hook.postmarch)
        self.assertEqual(SplitSave.postloop, Hook.postloop)

class BlockHookTest(TestCase):
    def setUp(self):
        self._msg = ''

    def info(self, msg):
        self._msg += msg

    def assertInfo(self, msg):
        self.assertEqual(self._msg, msg)

class TestInitializer(BlockHookTest):
    def test_property(self):
        from ..multidim import Initializer
        self.assertEqual(Initializer._varnames_, tuple())

    def test_methods(self):
        from ..core import Hook
        from ..multidim import Initializer
        self.assertTrue(callable(Initializer._take_data))
        self.assertTrue(callable(Initializer._set_data))
        self.assertTrue(callable(Initializer._put_data))
        self.assertNotEqual(Initializer.preloop, Hook.preloop)
        self.assertEqual(Initializer.premarch, Hook.premarch)
        self.assertEqual(Initializer.postmarch, Hook.postmarch)
        self.assertEqual(Initializer.postloop, Hook.postloop)

    def test_set_data(self):
        from ..multidim import BlockCase, Initializer
        case = BlockCase()
        hook = Initializer(case)
        self.assertRaises(NotImplementedError, hook._set_data)

    def test_preloop(self):
        from ..multidim import BlockCase, Initializer
        case = BlockCase()
        hook = Initializer(case)
        self.assertRaises(TypeError, hook.preloop)

class TestCalculator(BlockHookTest):
    def test_methods(self):
        from ..core import Hook
        from ..multidim import Calculator
        self.assertTrue(callable(Calculator._collect_solutions))
        self.assertTrue(callable(Calculator._calculate))
        self.assertEqual(Calculator.preloop, Hook.preloop)
        self.assertEqual(Calculator.premarch, Hook.premarch)
        self.assertEqual(Calculator.postmarch, Hook.postmarch)
        self.assertEqual(Calculator.postloop, Hook.postloop)

    def test_calculate(self):
        from ..multidim import BlockCase, Calculator
        case = BlockCase()
        hook = Calculator(case)
        self.assertRaises(NotImplementedError, hook._calculate)

class TestMarchSave(BlockHookTest):
    def test_methods(self):
        from ..core import Hook
        from ..multidim import MarchSave
        self.assertTrue(isinstance(MarchSave.data, property))
        self.assertTrue(callable(MarchSave._write))
        self.assertNotEqual(MarchSave.preloop, Hook.preloop)
        self.assertEqual(MarchSave.premarch, Hook.premarch)
        self.assertNotEqual(MarchSave.postmarch, Hook.postmarch)
        self.assertEqual(MarchSave.postloop, Hook.postloop)

from ..multidim import Initializer, Calculator
class CaseInitSet(Initializer):
    _varnames_ = (
        # key, putback.
        ('soln', True,),
        ('dsoln', True,),
    )
    def _set_data(self, **kw):
        soln = kw['soln']
        dsoln = kw['dsoln']
        # solutions.
        soln.fill(0.0)
        dsoln.fill(0.0)
class CaseInitCollect(Initializer):
    def preloop(self):
        # super preloop.
        soln = self._collect_interior('soln')
        dsoln = self._collect_interior('dsoln')
        soln.fill(0.0)
        dsoln.fill(0.0)
        self._spread_interior(soln, 'soln')
        self._spread_interior(dsoln, 'dsoln')
class CaseCalc(Calculator):
    def postloop(self):
        self._collect_solutions()
del Initializer, Calculator

class TestRun(TestCase):
    time = 0.0
    time_increment = 1.0
    nsteps = 10

    def _get_case(self, CaseInit, **kw):
        from ...solver.multidim import BlockSolver
        from ...testing import get_blk_from_sample_neu
        from ..multidim import BlockCase
        case = BlockCase(basedir='.', basefn='blockcase',
            solvertype=BlockSolver, neq=1,
            steps_run=self.nsteps, time_increment=self.time_increment,
            **kw
        )
        case.info = lambda *a: None
        case.load_block = get_blk_from_sample_neu
        case.execution.runhooks.append(CaseInit(case))
        case.execution.runhooks.append(CaseCalc(case))
        case.init()
        return case

class TestSequential(TestRun):
    def test_soln_set(self):
        from numpy import zeros
        from ...domain import Domain
        case = self._get_case(CaseInitSet, domaintype=Domain)
        svr = case.solver.solverobj
        case.run()
        ngstcell = svr.ngstcell
        # get result.
        soln = svr.soln[ngstcell:,0]
        # calculate reference
        clvol = zeros(soln.shape, dtype=soln.dtype)
        for iistep in range(self.nsteps*2):
            clvol += svr.clvol[ngstcell:]*self.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())
    def test_dsoln_set(self):
        from numpy import zeros
        from ...domain import Domain
        case = self._get_case(CaseInitSet, domaintype=Domain)
        svr = case.solver.solverobj
        case.run()
        ngstcell = svr.ngstcell
        # get result.
        dsoln = svr.dsoln[ngstcell:,0,:]
        # calculate reference
        clcnd = zeros(dsoln.shape, dtype=dsoln.dtype)
        for iistep in range(self.nsteps*2):
            clcnd += svr.clcnd[ngstcell:]*self.time_increment/2
        # compare.
        self.assertTrue((dsoln==clcnd).all())

    def test_soln_collect(self):
        from numpy import zeros
        from ...domain import Domain
        case = self._get_case(CaseInitCollect, domaintype=Domain)
        svr = case.solver.solverobj
        case.run()
        ngstcell = svr.ngstcell
        # get result.
        soln = svr.soln[ngstcell:,0]
        # calculate reference
        clvol = zeros(soln.shape, dtype=soln.dtype)
        for iistep in range(self.nsteps*2):
            clvol += svr.clvol[ngstcell:]*self.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())
    def test_dsoln_collect(self):
        from numpy import zeros
        from ...domain import Domain
        case = self._get_case(CaseInitCollect, domaintype=Domain)
        svr = case.solver.solverobj
        case.run()
        ngstcell = svr.ngstcell
        # get result.
        dsoln = svr.dsoln[ngstcell:,0,:]
        # calculate reference
        clcnd = zeros(dsoln.shape, dtype=dsoln.dtype)
        for iistep in range(self.nsteps*2):
            clcnd += svr.clcnd[ngstcell:]*self.time_increment/2
        # compare.
        self.assertTrue((dsoln==clcnd).all())

class TestParallel(TestRun):
    npart = 3

    def test_soln_set(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from ...domain import Collective
        case = self._get_case(CaseInitSet,
            npart=self.npart, domaintype=Collective)
        case.run()
        # get result.
        soln = case.execution.var['soln'][:,0]
        # calculate reference
        blk = case.solver.domainobj.blk
        clvol = zeros(soln.shape, dtype=soln.dtype)
        for iistep in range(self.nsteps*2):
            clvol += blk.clvol*self.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())
    def test_dsoln_set(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from ...domain import Collective
        case = self._get_case(CaseInitSet,
            npart=self.npart, domaintype=Collective)
        case.run()
        # get result.
        dsoln = case.execution.var['dsoln'][:,0,:]
        # calculate reference
        blk = case.solver.domainobj.blk
        clcnd = zeros(dsoln.shape, dtype=dsoln.dtype)
        for iistep in range(self.nsteps*2):
            clcnd += blk.clcncrd*self.time_increment/2
        # compare.
        self.assertTrue((dsoln==clcnd).all())

    def test_soln_collect(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from ...domain import Collective
        case = self._get_case(CaseInitCollect,
            npart=self.npart, domaintype=Collective)
        case.run()
        # get result.
        soln = case.execution.var['soln'][:,0]
        # calculate reference
        blk = case.solver.domainobj.blk
        clvol = zeros(soln.shape, dtype=soln.dtype)
        for iistep in range(self.nsteps*2):
            clvol += blk.clvol*self.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())
    def test_dsoln_collect(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from ...domain import Collective
        case = self._get_case(CaseInitCollect,
            npart=self.npart, domaintype=Collective)
        case.run()
        # get result.
        dsoln = case.execution.var['dsoln'][:,0,:]
        # calculate reference
        blk = case.solver.domainobj.blk
        clcnd = zeros(dsoln.shape, dtype=dsoln.dtype)
        for iistep in range(self.nsteps*2):
            clcnd += blk.clcncrd*self.time_increment/2
        # compare.
        self.assertTrue((dsoln==clcnd).all())
