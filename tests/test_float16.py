# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

import solvcon as sc


assert_array = np.testing.assert_array_equal
# 2**-25 puts binary64 above the Float16 midpoint but vanishes in Float32.
ABOVE_HALF_MIDPOINT = 1.0 + 2.0**-11 + 2.0**-25


class Float16PythonIntegrationTC(unittest.TestCase):

    def test_numpy_storage_is_zero_copy(self):
        source = np.arange(6, dtype='float16').reshape(2, 3)
        array = sc.SimpleArrayFloat16(array=source)

        self.assertEqual('e', memoryview(array).format)
        self.assertTrue(np.shares_memory(source, array.ndarray))

        source[0, 0] = -2
        self.assertEqual(-2, array[0, 0])

        with self.assertRaisesRegex(RuntimeError, 'dtype mismatch'):
            sc.SimpleArrayFloat16(array=source.astype('float32'))

    def test_python_scalars_and_collector(self):
        array = sc.SimpleArrayFloat16((3,))
        for index, value in enumerate((1, 2.25, np.float16(-3.5))):
            array[index] = value
        assert_array(np.array([1, 2.25, -3.5], dtype='float16'), array.ndarray)

        scalar = sc.SimpleArrayFloat16((1,))
        scalar[0] = ABOVE_HALF_MIDPOINT
        self.assertEqual(0x3c01, scalar.ndarray.view('uint16')[0])

        signaling_nan = np.array([0x7c01], dtype='uint16').view('float16')[0]
        scalar[0] = signaling_nan
        self.assertEqual(0x7c01, scalar.ndarray.view('uint16')[0])

        with self.assertRaises(RuntimeError):
            scalar[0] = np.complex64(1 + 2j)

        collector = sc.SimpleCollectorFloat16()
        for value in (1, 2.25, np.float16(-3.5)):
            collector.push_back(value)
        assert_array(array.ndarray, collector.as_array().ndarray)
        collector.push_back(signaling_nan)
        self.assertEqual(
            0x7c01, collector.as_array().ndarray.view('uint16')[-1])

        view = memoryview(signaling_nan)
        with self.assertRaises(TypeError):
            scalar.fill(view)
        with self.assertRaises(TypeError):
            collector.push_back(view)

        calculation_error = r'Float16.*not supported'
        with self.assertRaisesRegex(RuntimeError, calculation_error):
            _ = scalar == scalar
        with self.assertRaisesRegex(RuntimeError, calculation_error):
            _ = scalar != 1

    def test_typed_numpy_conversion_rules(self):
        target = sc.SimpleArrayFloat16((2, 3))
        source = np.arange(6, dtype='float64').reshape(2, 3) / 3
        target[...] = source
        assert_array(source.astype('float16'), target.ndarray)
        wider = sc.SimpleArrayFloat64((2, 3))
        wider[...] = target.ndarray
        assert_array(target.ndarray.astype('float64'), wider.ndarray)

        # Preserve a NaN payload and signed zero exactly.
        bits = np.array([0x7c01, 0x8000], dtype='uint16')
        copied = sc.SimpleArrayFloat16((2,))
        copied[...] = bits.view('float16')
        assert_array(bits, copied.ndarray.view('uint16'))

        for dtype, expected in (('float64', 0x3c01), ('float32', 0x3c00)):
            with self.subTest(dtype=dtype):
                converted = sc.SimpleArrayFloat16((1,))
                converted[...] = np.array([ABOVE_HALF_MIDPOINT], dtype=dtype)
                self.assertEqual(expected, converted.ndarray.view('uint16')[0])

        conversion_error = 'Cannot convert between complex and non-complex'
        for array_type, dtype in ((sc.SimpleArrayFloat16, 'complex64'),
                                  (sc.SimpleArrayComplex64, 'float16')):
            with self.subTest(array_type=array_type, dtype=dtype), \
                    self.assertRaisesRegex(RuntimeError, conversion_error):
                array_type((1,))[...] = np.ones(1, dtype=dtype)

    def test_float16_to_integer_conversion(self):
        for array_type, dtype, values in (
            (sc.SimpleArrayInt8, 'int8', [-128.75, -0.75, 0.75, 127.75]),
            (sc.SimpleArrayUint8, 'uint8', [-0.75, 0.75, 255.75]),
        ):
            source = np.array(values, dtype='float16')
            destination = array_type(source.shape)
            destination[...] = source
            assert_array(source.astype(dtype), destination.ndarray)

    def test_plex_float16_dispatch(self):
        array = sc.SimpleArray(
            shape=(2,), value=np.float16(1.25), dtype='float16')
        self.assertIs(sc.SimpleArray.typed_class('float16'),
                      sc.SimpleArrayFloat16)
        self.assertIsInstance(array.typed, sc.SimpleArrayFloat16)
        assert_array(np.full(2, 1.25, dtype='float16'), np.asarray(array))

        signaling_nan = np.array([0x7c01], dtype='uint16').view('float16')[0]
        array.fill(signaling_nan)
        assert_array(
            np.full(2, 0x7c01, dtype='uint16'),
            np.asarray(array).view('uint16'))

        array.fill(ABOVE_HALF_MIDPOINT)
        assert_array(
            np.full(2, 0x3c01, dtype='uint16'),
            np.asarray(array).view('uint16'))

        with self.assertRaisesRegex(TypeError, 'expected real number'):
            array.fill(np.complex128(1 + 2j))

        source = np.array([1.5, -2.25], dtype='float16')
        shared = sc.SimpleArray(array=source)
        self.assertTrue(np.shares_memory(source, np.asarray(shared)))

        with self.assertRaisesRegex(RuntimeError, r'Float16.*not supported'):
            shared.min()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
