from unittest import TestCase

class Test(TestCase):
    def test_arrangement_registry(self):
        from ..core import BaseCase
        from ..onedim import OnedimCase
        from ..multidim import BlockCase
        self.assertNotEqual(
            id(BaseCase.arrangements),
            id(OnedimCase.arrangements)
        )
        self.assertNotEqual(
            id(BaseCase.arrangements),
            id(BlockCase.arrangements)
        )
