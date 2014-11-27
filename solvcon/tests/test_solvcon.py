from unittest import TestCase

import solvcon as sc


class TestNamespace(TestCase):

    def test_hasattr(self):
        for name in sc.__all__:
            self.assertTrue(hasattr(sc, name))
