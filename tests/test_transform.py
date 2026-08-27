# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

import solvcon as sc


class FourierTransformTB(sc.testing.TestBase):

    def setUp(self):
        pass

    def real_rng(self):
        pass

    def imag_rng(self):
        pass

    def test_numpy_dft_comparison(self):
        input_size = 100

        sc_input = self.SimpleArray(input_size)
        for i in range(input_size):
            sc_input[i] = self.complex(self.real_rng(), self.imag_rng())

        np_input = np.array(sc_input, copy=False)

        sc_output = self.SimpleArray(input_size, self.complex())
        sc.FourierTransform.dft(sc_input, sc_output)

        np_output = np.fft.fft(np_input)

        for i in range(input_size):
            self.assert_allclose(sc_output[i].real, np_output[i].real)
            self.assert_allclose(sc_output[i].imag, np_output[i].imag)

    def test_numpy_duplicate_dft_comparison(self):
        input_size = 100

        sc_input = self.SimpleArray(input_size)
        for i in range(input_size):
            sc_input[i] = self.complex(self.real_rng(), self.imag_rng())

        np_input = np.array(sc_input, copy=False)

        sc_output = self.SimpleArray(input_size, self.complex())
        sc.FourierTransform.dft(sc_input, sc_output)
        sc.FourierTransform.dft(sc_input, sc_output)

        np_output = np.fft.fft(np_input)

        for i in range(input_size):
            self.assert_allclose(sc_output[i].real, np_output[i].real)
            self.assert_allclose(sc_output[i].imag, np_output[i].imag)

    def check_fft_against_numpy(self, **kw):
        input_size = 100

        sc_input = self.SimpleArray(input_size)
        for i in range(input_size):
            sc_input[i] = self.complex(self.real_rng(), self.imag_rng())

        np_input = np.array(sc_input, copy=False)

        sc_output = self.SimpleArray(input_size, self.complex())
        sc.FourierTransform.fft(sc_input, sc_output, **kw)

        np_output = np.fft.fft(np_input)

        for i in range(input_size):
            self.assert_allclose(sc_output[i].real, np_output[i].real)
            self.assert_allclose(sc_output[i].imag, np_output[i].imag)

    def test_numpy_fft_comparison(self):
        self.check_fft_against_numpy()

    def test_numpy_ifft_comparison(self):
        input_size = 100

        sc_input = self.SimpleArray(input_size)
        for i in range(input_size):
            sc_input[i] = self.complex(self.real_rng(), self.imag_rng())

        np_input = np.array(sc_input, copy=False)

        sc_output = self.SimpleArray(input_size, self.complex())
        sc.FourierTransform.ifft(sc_input, sc_output)

        np_output = np.fft.ifft(np_input)

        for i in range(input_size):
            self.assert_allclose(sc_output[i].real, np_output[i].real)
            self.assert_allclose(sc_output[i].imag, np_output[i].imag)

    def test_fft_cpu_backend_argument(self):
        self.check_fft_against_numpy(backend=sc.FourierBackend.cpu)

    def test_fft_cuda_backend(self):
        if not sc.FourierTransform.cuda_available():
            self.skipTest("CUDA is not available")
        self.check_fft_against_numpy(backend=sc.FourierBackend.cuda)

    def test_fft_cuda_backend_unavailable(self):
        if sc.FourierTransform.cuda_available():
            self.skipTest("CUDA is available")
        input_size = 8

        sc_input = self.SimpleArray(input_size, self.complex())
        sc_output = self.SimpleArray(input_size, self.complex())

        with self.assertRaisesRegex(RuntimeError,
                                    "CUDA FFT is not available"):
            sc.FourierTransform.fft(sc_input, sc_output,
                                    backend=sc.FourierBackend.cuda)


class TransformFp32TC(FourierTransformTB, unittest.TestCase):

    def assert_allclose(self, *args, **kw):
        if 'atol' not in kw:
            kw['atol'] = 1.e-2
        return super().assert_allclose(*args, **kw)

    def real_rng(self):
        return np.float32(np.random.uniform(-1.0, 1.0))

    def imag_rng(self):
        return np.float32(np.random.uniform(-1.0, 1.0))

    def setUp(self):
        np.random.seed()
        self.complex = sc.complex64
        self.SimpleArray = sc.SimpleArrayComplex64


class TransformFp64TC(FourierTransformTB, unittest.TestCase):

    def assert_allclose(self, *args, **kw):
        if 'atol' not in kw:
            kw['atol'] = 1.e-10
        return super().assert_allclose(*args, **kw)

    def real_rng(self):
        return np.random.uniform(-1.0, 1.0)

    def imag_rng(self):
        return np.random.uniform(-1.0, 1.0)

    def setUp(self):
        np.random.seed()
        self.complex = sc.complex128
        self.SimpleArray = sc.SimpleArrayComplex128

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
