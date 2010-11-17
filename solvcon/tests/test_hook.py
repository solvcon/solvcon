from unittest import TestCase

class TestHook(TestCase):
    def test_existence(self):
        from .. import hook
        self.assertTrue(hook.Hook)
        self.assertTrue(hook.ProgressHook)

    def test_type(self):
        from .. import hook
        self.assertRaises(AssertionError, hook.Hook, None)

    def test_hookmethods(self):
        from ..hook import Hook
        self.assertTrue(getattr(Hook, 'preloop', False))
        self.assertTrue(getattr(Hook, 'premarch', False))
        self.assertTrue(getattr(Hook, 'postmarch', False))
        self.assertTrue(getattr(Hook, 'postloop', False))

    def test_progress(self):
        from ..hook import Hook, ProgressHook
        self.assertNotEqual(ProgressHook.preloop, Hook.preloop)
        self.assertEqual(ProgressHook.premarch, Hook.premarch)
        self.assertNotEqual(ProgressHook.postmarch, Hook.postmarch)
        self.assertEqual(ProgressHook.postloop, Hook.postloop)

class TestBlockHook(TestCase):
    def test_type(self):
        from .. import hook
        self.assertRaises(AssertionError, hook.BlockHook, None)

    def test_blockhook(self):
        from ..hook import Hook, BlockHook
        self.assertEqual(BlockHook.preloop, Hook.preloop)
        self.assertEqual(BlockHook.premarch, Hook.premarch)
        self.assertEqual(BlockHook.postmarch, Hook.postmarch)
        self.assertEqual(BlockHook.postloop, Hook.postloop)

    def test_blockinfohook(self):
        from ..hook import Hook, BlockInfoHook
        self.assertNotEqual(BlockInfoHook.preloop, Hook.preloop)
        self.assertEqual(BlockInfoHook.premarch, Hook.premarch)
        self.assertNotEqual(BlockInfoHook.postmarch, Hook.postmarch)
        self.assertNotEqual(BlockInfoHook.postloop, Hook.postloop)

    def test_vtksave(self):
        from ..case import BlockCase
        from ..hook import Hook, VtkSave
        cse = BlockCase()
        hok = VtkSave(cse)
        self.assertEqual(hok.binary, False)
        self.assertEqual(hok.cache_grid, True)
        self.assertEqual(VtkSave.preloop, Hook.preloop)
        self.assertEqual(VtkSave.premarch, Hook.premarch)
        self.assertEqual(VtkSave.postmarch, Hook.postmarch)
        self.assertEqual(VtkSave.postloop, Hook.postloop)

    def test_splitsave(self):
        from ..hook import Hook, SplitSave
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

class TestMarchSave(BlockHookTest):
    def test_methods(self):
        from ..hook import Hook, MarchSave
        self.assertTrue(isinstance(MarchSave.data, property))
        self.assertTrue(callable(MarchSave._write))
        self.assertNotEqual(MarchSave.preloop, Hook.preloop)
        self.assertEqual(MarchSave.premarch, Hook.premarch)
        self.assertNotEqual(MarchSave.postmarch, Hook.postmarch)
        self.assertEqual(MarchSave.postloop, Hook.postloop)
