from unittest import TestCase

class TestCaseInfo(TestCase):
    def test_CaseInfo(self):
        from ..core import CaseInfo
        tdic = CaseInfo(_defdict={
            'lay1_1': '1_1',
            'lay1_2.lay2_1': '1_2/2_1',
            'lay1_2.lay2_2': '1_2/2_2',
            'lay1_3.': '1_3/',
        })
        self.assertEqual(tdic.lay1_1, '1_1')
        self.assertEqual(tdic.lay1_2.lay2_1, '1_2/2_1')
        self.assertEqual(tdic.lay1_2.lay2_2, '1_2/2_2')
        self.assertEqual(len(tdic.lay1_3), 0)

class TestBaseCase(TestCase):
    def test_empty_fields(self):
        from ..core import BaseCase
        case = BaseCase()
        # execution related.
        self.assertTrue(isinstance(case.execution.runhooks, list))
        self.assertEqual(case.execution.run_inner, False)
        self.assertEqual(case.execution.time, 0.0)
        self.assertEqual(case.execution.time_increment, 0.0)
        self.assertEqual(case.execution.step_init, 0)
        self.assertEqual(case.execution.step_current, None)
        self.assertEqual(case.execution.steps_run, None)
        self.assertEqual(case.execution.cCFL, 0.0)
        self.assertEqual(case.execution.aCFL, 0.0)
        self.assertEqual(case.execution.mCFL, 0.0)
        self.assertEqual(case.execution.neq, 0)
        self.assertEqual(case.execution.var, dict())
        self.assertEqual(case.execution.varstep, None)
        # io related.
        self.assertEqual(case.io.abspath, False)
        self.assertEqual(case.io.basedir, None)
        self.assertEqual(case.io.basefn, None)
        # condition related.
        self.assertEqual(case.condition.mtrllist, list())
        # solver related.
        self.assertEqual(case.solver.solvertype, None)
        self.assertEqual(case.solver.solverobj, None)
        # logging.
        self.assertEqual(case.log.time, dict())

    def test_abspath(self):
        import os
        from ..core import BaseCase
        case = BaseCase(basedir='.', abspath=True)
        path = os.path.abspath('.')
        self.assertEqual(case.io.basedir, path)

    def test_init(self):
        from ..core import BaseCase
        case = BaseCase()
        self.assertFalse(case._have_init)
        case.init()
        self.assertTrue(case._have_init)

    def test_run(self):
        from ..core import BaseCase
        case = BaseCase()
        self.assertRaises(NotImplementedError, case.run)

class TestHook(TestCase):
    def test_existence(self):
        from .. import core
        self.assertTrue(core.Hook)
        self.assertTrue(core.ProgressHook)
        self.assertTrue(core.CflHook)

    def test_type(self):
        from .. import core
        self.assertRaises(AssertionError, core.Hook, None)

    def test_hookmethods(self):
        from ..core import Hook
        self.assertTrue(getattr(Hook, 'preloop', False))
        self.assertTrue(getattr(Hook, 'premarch', False))
        self.assertTrue(getattr(Hook, 'postmarch', False))
        self.assertTrue(getattr(Hook, 'postloop', False))

    def test_progress(self):
        from ..core import Hook, ProgressHook
        self.assertNotEqual(ProgressHook.preloop, Hook.preloop)
        self.assertEqual(ProgressHook.premarch, Hook.premarch)
        self.assertNotEqual(ProgressHook.postmarch, Hook.postmarch)
        self.assertEqual(ProgressHook.postloop, Hook.postloop)

    def test_cfl(self):
        from ..core import Hook, CflHook
        self.assertEqual(CflHook.preloop, Hook.preloop)
        self.assertEqual(CflHook.premarch, Hook.premarch)
        self.assertNotEqual(CflHook.postmarch, Hook.postmarch)
        self.assertNotEqual(CflHook.postloop, Hook.postloop)
