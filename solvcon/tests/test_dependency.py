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
