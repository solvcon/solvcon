# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

import solvcon as sc


class ComplexTB(sc.testing.TestBase):

    def test_construct_default(self):
        cplx = self.sc_complex()
        self.assert_allclose(cplx.real, 0.0)
        self.assert_allclose(cplx.imag, 0.0)

    def test_construct_random(self):
        cplx = self.sc_complex(self.real1, self.imag1)
        self.assert_allclose(cplx.real, self.real1)
        self.assert_allclose(cplx.imag, self.imag1)

    def test_operator_add(self):
        cplx = self.sc_complex(self.real1, self.imag1)
        realv = self.np_float(2.0)

        result = cplx + realv

        self.assert_allclose(result.real, self.real1 + realv)
        self.assert_allclose(result.imag, self.imag1)

        result = realv + cplx

        self.assert_allclose(result.real, realv + cplx.real)
        self.assert_allclose(result.imag, cplx.imag)

        cplx1 = self.sc_complex(self.real1, self.imag1)
        cplx2 = self.sc_complex(self.real2, self.imag2)

        result = cplx1 + cplx2

        expected_real = self.real1 + self.real2
        expected_imag = self.imag1 + self.imag2

        self.assert_allclose(result.real, expected_real)
        self.assert_allclose(result.imag, expected_imag)

    def test_operator_sub(self):
        cplx = self.sc_complex(self.real1, self.imag1)
        realv = self.np_float(2.0)

        result = cplx - realv

        self.assert_allclose(result.real, self.real1 - realv)
        self.assert_allclose(result.imag, self.imag1)

        result = realv - cplx

        self.assert_allclose(result.real, realv - cplx.real)
        self.assert_allclose(result.imag, -cplx.imag)

        cplx1 = self.sc_complex(self.real1, self.imag1)
        cplx2 = self.sc_complex(self.real2, self.imag2)

        result = cplx1 - cplx2

        expected_real = self.real1 - self.real2
        expected_imag = self.imag1 - self.imag2

        self.assert_allclose(result.real, expected_real)
        self.assert_allclose(result.imag, expected_imag)

    def test_operator_mul(self):
        cplx = self.sc_complex(self.real1, self.imag1)
        realv = self.np_float(2.0)

        result = cplx * realv

        self.assert_allclose(result.real, self.real1 * realv)
        self.assert_allclose(result.imag, self.imag1 * realv)

        result = realv * cplx
        golden = self.sc_complex(realv, 0.0) * cplx

        self.assert_allclose(result.real, golden.real)
        self.assert_allclose(result.imag, golden.imag)

        cplx1 = self.sc_complex(self.real1, self.imag1)
        cplx2 = self.sc_complex(self.real2, self.imag2)

        result = cplx1 * cplx2

        expected_real = (self.real1 * self.real2
                         - self.imag1 * self.imag2)
        expected_imag = (self.real1 * self.imag2
                         + self.imag1 * self.real2)

        self.assert_allclose(result.real, expected_real)
        self.assert_allclose(result.imag, expected_imag)

    def test_operator_div(self):
        cplx = self.sc_complex(self.real1, self.imag1)
        realv = self.np_float(2.0)

        result = cplx / realv

        self.assert_allclose(result.real, self.real1 / realv)
        self.assert_allclose(result.imag, self.imag1 / realv)

        result = realv / cplx

        denominator = (self.real1 * self.real1 + self.imag1 *
                       self.imag1)
        expected_real = (self.real1 * realv) / denominator
        expected_imag = (-self.imag1 * realv) / denominator

        self.assert_allclose(result.real, expected_real)
        self.assert_allclose(result.imag, expected_imag)

        cplx1 = self.sc_complex(self.real1, self.imag1)
        cplx2 = self.sc_complex(self.real2, self.imag2)

        result = cplx1 / cplx2

        denominator = (self.real2 * self.real2 + self.imag2 *
                       self.imag2)
        expected_real = (self.real1 * self.real2 + self.imag1 *
                         self.imag2) / denominator
        expected_imag = (self.imag1 * self.real2 - self.real1 *
                         self.imag2) / denominator

        self.assert_allclose(result.real, expected_real)
        self.assert_allclose(result.imag, expected_imag)

    def test_operator_comparison(self):
        cplx1 = self.sc_complex(self.real1, self.imag1)
        cplx2 = self.sc_complex(self.real2, self.imag2)

        # Numpy uses lexicographic ordering to compare complex numbers, as
        # documented in
        # https://numpy.org/devdocs/reference/generated/numpy.sort.html .
        # We match the behaviors. The following discussions include more
        # details:
        # 1. https://github.com/numpy/numpy/issues/12943
        # 2. https://stackoverflow.com/questions/52481376
        def compare():
            if self.real1 == self.real2:
                return self.imag1 <= self.imag2
            return self.real1 <= self.real2
        self.assertEqual(cplx1 < cplx2, compare())

    def test_norm(self):
        cplx = self.sc_complex(self.real1, self.imag1)

        result = cplx.norm()

        expected_val = self.real1 ** 2 + self.imag1 ** 2

        self.assert_allclose(result, expected_val)

    def test_dtype_verification(self):
        self.assertEqual(self.dtype, self.expected_dtype)

    def test_complex_array(self):
        cplx = self.sc_complex(self.real1, self.imag1)
        sarr = self.sc_simplearraycomplex(10)
        sarr.fill(cplx)
        ndarr = np.array(sarr, copy=False, dtype=self.dtype)

        for i in range(10):
            self.assertEqual(ndarr[i].real, self.real1)
            self.assertEqual(ndarr[i].imag, self.imag1)

        self.assertEqual(ndarr.dtype, self.dtype)

        sarr = self.sc_simplearraycomplex(array=ndarr)

        for i in range(10):
            self.assertEqual(sarr[i].real, self.real1)
            self.assertEqual(sarr[i].imag, self.imag1)

        self.assertEqual(sarr.ndarray.dtype, ndarr.dtype)
        self.assertEqual(10 * self.esize, sarr.nbytes)

    def test_complex_conj(self):
        cplx = self.sc_complex(self.real1, self.imag1)
        cplx_conj = self.sc_complex(self.real1, -self.imag1)

        self.assertEqual(cplx.conj().real, cplx_conj.real)
        self.assertEqual(cplx.conj().imag, cplx_conj.imag)


class ComplexFp32TC(ComplexTB, unittest.TestCase):

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def sc_complex(self, real=None, imag=None):
        if real is not None and imag is not None:
            return sc.complex64(real, imag)
        else:
            return sc.complex64()

    def sc_simplearraycomplex(self, size=None, array=None):
        if size is not None:
            return sc.SimpleArrayComplex64(size)
        if array is not None:
            return sc.SimpleArrayComplex64(array=array)
        return sc.SimpleArrayComplex64(0)

    def np_float(self, val):
        return np.float32(val)

    def setUp(self):
        self.real1 = np.float32(0.7)
        self.imag1 = np.float32(1.6)
        self.real2 = np.float32(2.5)
        self.imag2 = np.float32(3.4)
        self.realv = np.float32(2.0)
        self.dtype = sc.complex64.dtype()
        self.expected_dtype = np.dtype('complex64')
        self.esize = 4 * 2


class ComplexFp64TC(ComplexTB, unittest.TestCase):

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def sc_complex(self, real=None, imag=None):
        if real is not None and imag is not None:
            return sc.complex128(real, imag)
        else:
            return sc.complex128()

    def sc_simplearraycomplex(self, size=None, array=None):
        if size is not None:
            return sc.SimpleArrayComplex128(size)
        if array is not None:
            return sc.SimpleArrayComplex128(array=array)
        return sc.SimpleArrayComplex128(0)

    def np_float(self, val):
        return np.float64(val)

    def setUp(self):
        self.real1 = np.float64(4.3)
        self.imag1 = np.float64(5.2)
        self.real2 = np.float64(6.1)
        self.imag2 = np.float64(7.0)
        self.realv = np.float64(2.0)
        self.dtype = sc.complex128.dtype()
        self.expected_dtype = np.dtype('complex128')
        self.esize = 8 * 2
