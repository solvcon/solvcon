# -*- coding: UTF-8 -*-

from unittest import TestCase

class TestClibrary(TestCase):
    def test_solvcon(self):
        from ctypes import CDLL
        from .. import dependency as dep
        self.assertTrue(isinstance(dep._clib_solvcon_d, CDLL))
        self.assertTrue(isinstance(dep._clib_solvcon_s, CDLL))
    def test_metis(self):
        from ctypes import CDLL
        from .. import dependency as dep
        self.assertTrue(isinstance(dep._clib_metis, CDLL))
