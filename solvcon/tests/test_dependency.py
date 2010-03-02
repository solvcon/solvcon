# -*- coding: UTF-8 -*-

from unittest import TestCase

class TestFortranType(TestCase):
    def test_to_fortran_type(self):
        from ..dependency import FortranType
        from ctypes import c_int, c_double
        class TestType(FortranType):
            _fields_ = [
                ('field1', c_int),
                ('field2', c_double),
            ]
            _fortran_name_ = 'testtype'
        testdata = TestType()
        self.assertEqual(testdata.to_text(),
            """type testtype
    integer*4 :: field1
    real*8 :: field2
end type testtype""")

    def test_str(self):
        from ..dependency import FortranType
        from ctypes import c_int, c_double
        class TestType(FortranType):
            _fields_ = [
                ('field1', c_int),
                ('field2', c_double),
            ]
            _fortran_name_ = 'testtype'
        testdata = TestType(field1=1, field2=1.0)
        self.assertEqual(str(testdata),
            """type testtype
    integer*4 :: field1 = 1
    real*8 :: field2 = 1.0
end type testtype""")
