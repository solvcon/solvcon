# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import numpy as np

import solvcon
from solvcon import timeseries as ts


def arr(dtype, values):
    cls = getattr(solvcon, 'SimpleArray' + dtype.capitalize())
    return cls(array=np.array(values, dtype=dtype))


def u64(values):
    return arr('uint64', values)


def f64(values):
    return arr('float64', values)


INT_DTYPES = ('int8', 'int16', 'int32', 'int64',
              'uint8', 'uint16', 'uint32', 'uint64')
ALL_DTYPES = ('bool',) + INT_DTYPES + ('float32', 'float64',
                                       'complex64', 'complex128')


class MergeSortedUniqueTC(unittest.TestCase):

    def test_union_grid(self):
        grid = ts.merge_sorted_unique(u64([0, 10, 20, 30]), u64([5, 10, 25]))
        self.assertIs(type(grid), solvcon.SimpleArrayUint64)
        self.assertEqual(grid.ndarray.tolist(), [0, 5, 10, 20, 25, 30])

    def test_repeat_within_and_across_arrays_appears_once(self):
        grid = ts.merge_sorted_unique(u64([1, 1, 2]), u64([2, 3, 3]),
                                      u64([3]))
        self.assertEqual(grid.ndarray.tolist(), [1, 2, 3])

    def test_single_array_drops_its_repeats(self):
        grid = ts.merge_sorted_unique(u64([4, 4, 4, 9]))
        self.assertEqual(grid.ndarray.tolist(), [4, 9])

    def test_no_or_empty_input_gives_empty_grid(self):
        self.assertEqual(ts.merge_sorted_unique().shape, (0,))
        self.assertEqual(ts.merge_sorted_unique(u64([]), u64([])).shape,
                         (0,))
        grid = ts.merge_sorted_unique(u64([]), u64([7]))
        self.assertEqual(grid.ndarray.tolist(), [7])

    def test_against_numpy_union(self):
        rng = np.random.default_rng(20260817)
        for _ in range(20):
            ndatas = [np.sort(rng.integers(0, 50, rng.integers(0, 30),
                                           dtype='uint64'))
                      for _ in range(4)]
            grid = ts.merge_sorted_unique(*(u64(nd) for nd in ndatas))
            np.testing.assert_array_equal(grid.ndarray,
                                          np.unique(np.concatenate(ndatas)))

    def test_strided_view_is_read_in_array_order(self):
        ndata = np.arange(12, dtype='uint64')[::4]
        grid = ts.merge_sorted_unique(solvcon.SimpleArrayUint64(array=ndata),
                                      u64([2]))
        self.assertEqual(grid.ndarray.tolist(), [0, 2, 4, 8])

    def test_zero_stride_view_counts_its_elements(self):
        # A broadcast view repeats one element with stride 0; the merge
        # walks it by count, so the repeated timestamp still appears once.
        base = np.array([7], dtype='uint64')
        view = np.lib.stride_tricks.as_strided(base, shape=(3,), strides=(0,))
        grid = ts.merge_sorted_unique(solvcon.SimpleArrayUint64(array=view),
                                      u64([5]))
        self.assertEqual(grid.ndarray.tolist(), [5, 7])

    def test_unsorted_array_raises(self):
        msg = ("array 1 must be non-decreasing but element 2 = 2 is less "
               "than element 1 = 3")
        with self.assertRaisesRegex(ValueError, msg):
            ts.merge_sorted_unique(u64([1, 2]), u64([1, 3, 2]))

    def test_rejects_other_dtype_and_dimension(self):
        with self.assertRaisesRegex(TypeError, "SimpleArrayInt64"):
            ts.merge_sorted_unique(solvcon.SimpleArrayInt64(
                array=np.array([1], dtype='int64')))
        with self.assertRaisesRegex(ValueError, "currently only support 1D "
                                    "array but the array 0 is 2 dimension"):
            ts.merge_sorted_unique(solvcon.SimpleArrayUint64((2, 2), 0))


class DedupLastTC(unittest.TestCase):

    def test_keeps_last_of_each_group(self):
        times = u64([0, 1, 1, 1, 2, 3, 3])
        values = f64([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        otimes, ovalues = ts.dedup_last(times, values)
        self.assertEqual(otimes.ndarray.tolist(), [0, 1, 2, 3])
        self.assertEqual(ovalues.ndarray.tolist(), [0.5, 2.0, 2.5, 3.5])

    def test_series_without_repeat_comes_back_as_a_copy(self):
        times, values = u64([2, 5, 9]), f64([1.0, 2.0, 3.0])
        otimes, ovalues = ts.dedup_last(times, values)
        self.assertEqual(otimes.ndarray.tolist(), [2, 5, 9])
        self.assertEqual(ovalues.ndarray.tolist(), [1.0, 2.0, 3.0])
        otimes[0], ovalues[0] = 99, -1.0
        self.assertEqual((times[0], values[0]), (2, 1.0))

    def test_strided_view_is_read_in_array_order(self):
        times = solvcon.SimpleArrayUint64(
            array=np.arange(12, dtype='uint64')[::4])
        values = solvcon.SimpleArrayFloat64(
            array=np.arange(6, dtype='float64')[::2])
        otimes, ovalues = ts.dedup_last(times, values)
        self.assertEqual(otimes.ndarray.tolist(), [0, 4, 8])
        self.assertEqual(ovalues.ndarray.tolist(), [0.0, 2.0, 4.0])
        otimes, ovalues = ts.dedup_last(u64([0, 0, 5]), values)
        self.assertEqual(otimes.ndarray.tolist(), [0, 5])
        self.assertEqual(ovalues.ndarray.tolist(), [2.0, 4.0])

    def test_repeat_at_head_and_all_equal_keep_the_last(self):
        otimes, ovalues = ts.dedup_last(u64([0, 0, 1]), f64([1.0, 2.0, 3.0]))
        self.assertEqual(otimes.ndarray.tolist(), [0, 1])
        self.assertEqual(ovalues.ndarray.tolist(), [2.0, 3.0])
        otimes, ovalues = ts.dedup_last(u64([7, 7, 7]), f64([1.0, 2.0, 3.0]))
        self.assertEqual(otimes.ndarray.tolist(), [7])
        self.assertEqual(ovalues.ndarray.tolist(), [3.0])

    def test_empty_and_single_pass_through(self):
        otimes, ovalues = ts.dedup_last(u64([]), f64([]))
        self.assertEqual((otimes.shape, ovalues.shape), ((0,), (0,)))
        otimes, ovalues = ts.dedup_last(u64([4]), f64([1.5]))
        self.assertEqual((otimes.ndarray.tolist(), ovalues.ndarray.tolist()),
                         ([4], [1.5]))

    def test_output_keeps_the_value_dtype(self):
        times = u64([1, 1, 2])
        for dtype in ALL_DTYPES:
            values = arr(dtype, [1, 0, 1])
            otimes, ovalues = ts.dedup_last(times, values)
            self.assertIs(type(ovalues), type(values), dtype)
            self.assertEqual(otimes.ndarray.tolist(), [1, 2], dtype)
            self.assertEqual(ovalues.ndarray.tolist(), [0, 1], dtype)

    def test_invalid_input_raises(self):
        with self.assertRaisesRegex(ValueError, "dedup_last.*non-decreasing"):
            ts.dedup_last(u64([2, 1]), f64([0.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "dedup_last.*2 samples but "
                                    "values has 3"):
            ts.dedup_last(u64([1, 2]), f64([0.0, 0.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "currently only support 1D "
                                    "array but the values is 2 dimension"):
            ts.dedup_last(u64([1, 2]), solvcon.SimpleArrayFloat64((1, 2), 0.0))
        ghost_msg = "ghosted time series are not supported"
        gtimes = solvcon.SimpleArrayUint64((2,), 0)
        gtimes.nghost = 1
        with self.assertRaisesRegex(ValueError, ghost_msg):
            ts.dedup_last(gtimes, f64([0.0, 0.0]))
        gvalues = solvcon.SimpleArrayFloat64((2,), 0.0)
        gvalues.nghost = 1
        with self.assertRaisesRegex(ValueError, ghost_msg):
            ts.dedup_last(u64([1, 2]), gvalues)


class DerivTC(unittest.TestCase):

    def test_backward_difference_drops_first_point(self):
        times = u64([0, 10, 30, 60])
        values = f64([1.0, 3.0, 2.0, 5.0])
        otimes, oderiv = ts.deriv(times, values)
        self.assertIs(type(oderiv), solvcon.SimpleArrayFloat64)
        self.assertEqual(otimes.ndarray.tolist(), [10, 30, 60])
        self.assertEqual(oderiv.ndarray.tolist(), [0.2, -0.05, 0.1])

    def test_against_numpy_diff_ratio(self):
        rng = np.random.default_rng(20260817)
        ntimes = np.cumsum(rng.integers(1, 100, 200, dtype='uint64'))
        nvalues = rng.standard_normal(200, dtype='float64')
        otimes, oderiv = ts.deriv(u64(ntimes), f64(nvalues))
        np.testing.assert_array_equal(otimes.ndarray, ntimes[1:])
        np.testing.assert_allclose(
            oderiv.ndarray,
            np.diff(nvalues) / np.diff(ntimes).astype('float64'))

    def test_fewer_than_two_samples_give_empty_result(self):
        for data in ([], [3.0]):
            otimes, oderiv = ts.deriv(u64(range(len(data))), f64(data))
            self.assertEqual((otimes.shape, oderiv.shape), ((0,), (0,)))

    def test_float32_keeps_float32_and_integer_gives_float64(self):
        times = u64([0, 4])
        _, oderiv = ts.deriv(times, arr('float32', [1.0, 3.0]))
        self.assertIs(type(oderiv), solvcon.SimpleArrayFloat32)
        self.assertEqual(oderiv.ndarray.tolist(), [0.5])
        for dtype in INT_DTYPES:
            _, oderiv = ts.deriv(times, arr(dtype, [5, 3]))
            self.assertIs(type(oderiv), solvcon.SimpleArrayFloat64, dtype)
            self.assertEqual(oderiv.ndarray.tolist(), [-0.5], dtype)

    def test_integer_difference_keeps_sign_and_full_width(self):
        # SimpleArrayUint64.diff() wraps a fall to 2**64 - 2; deriv() takes
        # the smaller integer from the larger in uint64 and negates, so a
        # fall keeps its sign and a difference wider than T keeps its
        # magnitude.
        times = u64([0, 1, 2])
        _, oderiv = ts.deriv(times, arr('uint64', [5, 3, 2**63 + 1]))
        self.assertEqual(oderiv.ndarray.tolist(), [-2.0, float(2**63 - 2)])
        expected = [float(2**64 - 1), -float(2**64 - 1)]
        _, oderiv = ts.deriv(times, arr('uint64', [0, 2**64 - 1, 0]))
        self.assertEqual(oderiv.ndarray.tolist(), expected)
        _, oderiv = ts.deriv(times, arr('int64', [-2**63, 2**63 - 1, -2**63]))
        self.assertEqual(oderiv.ndarray.tolist(), expected)
        _, oderiv = ts.deriv(u64([0, 1]), arr('int8', [-128, 127]))
        self.assertEqual(oderiv.ndarray.tolist(), [255.0])

    def test_invalid_input_raises(self):
        msg = "strictly increasing"
        with self.assertRaisesRegex(ValueError, msg):
            ts.deriv(u64([0, 1, 1, 2]), f64([0.0, 1.0, 2.0, 3.0]))
        with self.assertRaisesRegex(ValueError, msg):
            ts.deriv(u64([0, 0, 1]), f64([0.0, 1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "element 2 = 1 does not "
                                    "exceed element 1 = 2"):
            ts.deriv(u64([0, 2, 1]), f64([0.0, 1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "deriv.*3 samples but values "
                                    "has 2"):
            ts.deriv(u64([0, 1, 2]), f64([0.0, 1.0]))

    def test_dedup_last_makes_a_repeated_log_differentiable(self):
        cases = (
            ([0, 1, 1, 2], [0.0, 1.0, 2.0, 3.0], [1, 2], [2.0, 1.0]),
            ([0, 0, 1], [0.0, 1.0, 2.0], [1], [1.0]),
            ([3, 3, 3], [0.0, 1.0, 2.0], [], []),
        )
        for times, values, etimes, ederiv in cases:
            otimes, oderiv = ts.deriv(*ts.dedup_last(u64(times), f64(values)))
            self.assertEqual(otimes.ndarray.tolist(), etimes)
            self.assertEqual(oderiv.ndarray.tolist(), ederiv)

    def test_strided_view_is_read_in_array_order(self):
        times = solvcon.SimpleArrayUint64(
            array=np.arange(12, dtype='uint64')[::4])
        values = solvcon.SimpleArrayFloat64(
            array=np.arange(6, dtype='float64')[::2])
        otimes, oderiv = ts.deriv(times, values)
        self.assertEqual(otimes.ndarray.tolist(), [4, 8])
        self.assertEqual(oderiv.ndarray.tolist(), [0.5, 0.5])

    def test_chains_into_a_second_derivative(self):
        times = u64([0, 1, 2, 3])
        t_first, first = ts.deriv(times, f64([0.0, 1.0, 4.0, 9.0]))
        t_second, second = ts.deriv(t_first, first)
        self.assertEqual(t_second.ndarray.tolist(), [2, 3])
        self.assertEqual(second.ndarray.tolist(), [2.0, 2.0])

    def test_rejects_bool_and_complex_values(self):
        msg = "incompatible function arguments"
        for values in (arr('bool', [True, False]),
                       arr('complex128', [1 + 1j, 2j])):
            with self.assertRaisesRegex(TypeError, msg):
                ts.deriv(u64([0, 1]), values)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
