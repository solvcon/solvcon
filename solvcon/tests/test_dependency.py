# -*- coding: UTF-8 -*-

from unittest import TestCase

class TestUtility(TestCase):
    def test_guess_dllname(self):
        import sys
        from ..dependency import guess_dllname
        if sys.platform.startswith('win'):
            self.assertEqual('name.dll', guess_dllname('name'))
        elif sys.platform == 'darwin':
            self.assertEqual('libname.dylib', guess_dllname('name'))
        else:
            self.assertEqual('libname.so', guess_dllname('name'))

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
