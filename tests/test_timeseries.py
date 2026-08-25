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


def bl(values):
    return arr('bool', values)


INT_DTYPES = ('int8', 'int16', 'int32', 'int64',
              'uint8', 'uint16', 'uint32', 'uint64')
ALL_DTYPES = ('bool',) + INT_DTYPES + ('float32', 'float64',
                                       'complex64', 'complex128')


class NumpyInputTC(unittest.TestCase):

    def test_every_kernel_accepts_ndarrays(self):
        times = np.array([0, 10, 20], dtype='uint64')
        values = np.array([1.0, 3.0, 5.0], dtype='float64')
        flags = np.array([False, True, True], dtype='bool')

        self.assertEqual(
            ts.merge_sorted_unique(times).ndarray.tolist(), [0, 10, 20])
        self.assertEqual(
            ts.dedup_last(times, values)[1].ndarray.tolist(), [1.0, 3.0, 5.0])
        self.assertEqual(
            ts.deriv(times, values)[1].ndarray.tolist(), [0.2, 0.2])
        self.assertEqual(
            ts.movavg(times, values, span=10)[1].ndarray.tolist(),
            [1.0, 3.0, 5.0])
        self.assertEqual(
            ts.held(times, flags, span=10)[1].ndarray.tolist(),
            [False, False, True])
        self.assertEqual(
            ts.true_intervals(times, flags).ndarray.tolist(), [[10, 20, 10]])

    def test_strided_ndarrays_keep_their_logical_order(self):
        times = np.arange(0, 60, 10, dtype='uint64')[::2]
        values = np.arange(6, dtype='float64')[::2]

        output_times, output_values = ts.deriv(times, values)

        self.assertEqual(output_times.ndarray.tolist(), [20, 40])
        self.assertEqual(output_values.ndarray.tolist(), [0.1, 0.1])

    def test_arrayplex_inputs_alias_their_ndarray(self):
        ntimes = np.array([0, 10, 20], dtype='uint64')
        nvalues = np.array([1.0, 3.0, 5.0], dtype='float64')
        times = solvcon.SimpleArray(array=ntimes)
        values = solvcon.SimpleArray(array=nvalues)
        nvalues[1] = 5.0

        output_times, output_values = ts.deriv(times, values)

        self.assertIs(type(output_values), solvcon.SimpleArrayFloat64)
        self.assertEqual(output_times.ndarray.tolist(), [10, 20])
        self.assertEqual(output_values.ndarray.tolist(), [0.4, 0.0])

    def test_arrayplex_inputs_reach_merge_sorted_unique(self):
        times = solvcon.SimpleArray(array=np.array([0, 10, 20],
                                                   dtype='uint64'))

        merged = ts.merge_sorted_unique(times)

        self.assertEqual(merged.ndarray.tolist(), [0, 10, 20])

    def test_strided_arrayplex_inputs_keep_their_logical_order(self):
        ntimes = np.arange(0, 60, 10, dtype='uint64')[::2]
        nvalues = np.arange(6, dtype='float64')[::2]
        times = solvcon.SimpleArray(array=ntimes)
        values = solvcon.SimpleArray(array=nvalues)

        nvalues[1] = 6.0
        output_times, output_values = ts.deriv(times, values)

        self.assertEqual(list(times), [0, 20, 40])
        self.assertEqual(list(values), [0.0, 6.0, 4.0])
        self.assertEqual(output_times.ndarray.tolist(), [20, 40])
        self.assertEqual(output_values.ndarray.tolist(), [0.3, -0.1])


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
            values = np.array([1, 0, 1], dtype=dtype)
            otimes, ovalues = ts.dedup_last(times, values)
            self.assertIs(type(ovalues), type(arr(dtype, [])), dtype)
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


def naive_movavg(times, values, span):
    out = []
    for t in times:
        window = [v for tj, v in zip(times, values)
                  if tj <= t and tj + span > t]
        out.append(sum(window) / len(window))
    return out


class MovavgTC(unittest.TestCase):

    def test_window_is_trailing_and_half_open(self):
        times = u64([0, 10, 20, 30])
        values = f64([1.0, 2.0, 3.0, 4.0])
        otimes, omean = ts.movavg(times, values, span=20)
        self.assertIs(type(omean), solvcon.SimpleArrayFloat64)
        self.assertEqual(otimes.ndarray.tolist(), [0, 10, 20, 30])
        # At t=20 the window is (0, 20]: the sample at 0 is exactly span
        # behind and out, so the mean is over 2.0 and 3.0. The two head
        # windows are partial and never empty.
        self.assertEqual(omean.ndarray.tolist(), [1.0, 1.5, 2.5, 3.5])

    def test_span_wider_than_the_log_does_not_underflow(self):
        # t - span wraps in uint64 near the head; every sample must stay in
        # the window instead of the window going empty.
        times = u64([0, 1, 2, 3])
        values = f64([1.0, 3.0, 5.0, 7.0])
        _, omean = ts.movavg(times, values, span=2**64 - 1)
        self.assertEqual(omean.ndarray.tolist(), [1.0, 2.0, 3.0, 4.0])
        big = u64([2**63, 2**63 + 1])
        _, omean = ts.movavg(big, f64([2.0, 4.0]), span=2**63 + 5)
        self.assertEqual(omean.ndarray.tolist(), [2.0, 3.0])

    def test_a_bad_sample_outlives_its_window(self):
        # The running sum subtracts a sample on the way out (see the TODO
        # on movavg()). Subtracting a non-finite sample back leaves NaN in
        # every later mean, and subtracting the 1e17 takes the absorbed
        # 1.0 with it: the exact trailing means are [1e17, 1.0, 1.0].
        times = u64([0, 1, 2, 3])
        for bad in (float('inf'), float('-inf'), np.nan):
            _, omean = ts.movavg(times, f64([bad, 1.0, 1.0, 1.0]), span=1)
            np.testing.assert_array_equal(omean.ndarray[0], bad)
            self.assertTrue(np.all(np.isnan(omean.ndarray[1:])))
        _, omean = ts.movavg(u64([0, 1, 2]), f64([1e17, 1.0, 1.0]), span=1)
        self.assertEqual(omean.ndarray.tolist(), [1e17, 0.0, 0.0])

    def test_float32_stays_accurate_over_a_long_series(self):
        # The running sum is kept in double, so a float32 series with a
        # large offset stays close to the exact window mean.
        rng = np.random.default_rng(20260819)
        nvalues = (1e5 + rng.standard_normal(5000)).astype('float32')
        _, omean = ts.movavg(u64(np.arange(5000, dtype='uint64')),
                             arr('float32', nvalues), span=10)
        expected = [nvalues[max(0, i - 9):i + 1].astype('float64').mean()
                    for i in range(5000)]
        np.testing.assert_allclose(omean.ndarray, expected, atol=0.05)

    def test_repeated_timestamp_weighs_the_window_by_its_count(self):
        # Every sample of a repeated timestamp sees the same window, and
        # the three samples at 10 all weigh in it; a repeated group at the
        # head shares the head window the same way.
        times = u64([0, 10, 10, 10, 20])
        values = f64([0.0, 1.0, 2.0, 3.0, 4.0])
        otimes, omean = ts.movavg(times, values, span=15)
        self.assertEqual(otimes.ndarray.tolist(), [0, 10, 10, 10, 20])
        self.assertEqual(omean.ndarray.tolist(), [0.0, 1.5, 1.5, 1.5, 2.5])
        _, omean = ts.movavg(u64([10, 10, 20]), f64([1.0, 3.0, 5.0]), span=5)
        self.assertEqual(omean.ndarray.tolist(), [2.0, 2.0, 5.0])

    def test_against_a_naive_sweep(self):
        rng = np.random.default_rng(20260819)
        for _ in range(20):
            ntimes = np.sort(rng.integers(0, 200, 60, dtype='uint64'))
            nvalues = rng.standard_normal(60, dtype='float64')
            times, values = u64(ntimes), f64(nvalues)
            ltimes, lvalues = ntimes.tolist(), nvalues.tolist()
            for span in (1, 7, 50, 500):
                _, omean = ts.movavg(times, values, span)
                np.testing.assert_allclose(
                    omean.ndarray, naive_movavg(ltimes, lvalues, span))

    def test_empty_and_single_series(self):
        otimes, omean = ts.movavg(u64([]), f64([]), span=5)
        self.assertEqual((otimes.shape, omean.shape), ((0,), (0,)))
        otimes, omean = ts.movavg(u64([9]), f64([2.5]), span=5)
        self.assertEqual((otimes.ndarray.tolist(), omean.ndarray.tolist()),
                         ([9], [2.5]))

    def test_float32_keeps_float32_and_integer_gives_float64(self):
        times = u64([0, 1])
        _, omean = ts.movavg(times, arr('float32', [1.0, 2.0]), span=5)
        self.assertIs(type(omean), solvcon.SimpleArrayFloat32)
        self.assertEqual(omean.ndarray.tolist(), [1.0, 1.5])
        for dtype in INT_DTYPES:
            _, omean = ts.movavg(times, arr(dtype, [3, 4]), span=5)
            self.assertIs(type(omean), solvcon.SimpleArrayFloat64, dtype)
            self.assertEqual(omean.ndarray.tolist(), [3.0, 3.5], dtype)

    def test_strided_view_is_read_in_array_order(self):
        times = solvcon.SimpleArrayUint64(
            array=np.arange(12, dtype='uint64')[::4])
        values = solvcon.SimpleArrayFloat64(
            array=np.arange(6, dtype='float64')[::2])
        otimes, omean = ts.movavg(times, values, span=5)
        self.assertEqual(otimes.ndarray.tolist(), [0, 4, 8])
        self.assertEqual(omean.ndarray.tolist(), [0.0, 1.0, 3.0])

    def test_smooths_a_derivative(self):
        t_acc, acc = ts.deriv(u64([0, 1, 2, 3, 4]),
                              f64([0.0, 1.0, 0.0, 1.0, 0.0]))
        t_smooth, smooth = ts.movavg(t_acc, acc, span=2)
        self.assertEqual(t_smooth.ndarray.tolist(), [1, 2, 3, 4])
        self.assertEqual(smooth.ndarray.tolist(), [1.0, 0.0, 0.0, 0.0])

    def test_invalid_input_raises(self):
        with self.assertRaisesRegex(ValueError,
                                    "movavg.*span must be positive but is 0"):
            ts.movavg(u64([0, 1]), f64([0.0, 1.0]), span=0)
        with self.assertRaisesRegex(ValueError, "movavg.*non-decreasing"):
            ts.movavg(u64([2, 1]), f64([0.0, 0.0]), span=5)
        with self.assertRaisesRegex(ValueError, "movavg.*2 samples but "
                                    "values has 3"):
            ts.movavg(u64([1, 2]), f64([0.0, 0.0, 0.0]), span=5)
        with self.assertRaisesRegex(TypeError, "incompatible function "
                                    "arguments"):
            ts.movavg(u64([0, 1]), bl([True, False]), span=5)


class HeldTC(unittest.TestCase):

    def test_true_only_after_a_full_window_of_true(self):
        times = u64([0, 10, 20, 30, 40])
        values = bl([True, True, True, True, True])
        otimes, oheld = ts.held(times, values, span=20)
        self.assertIs(type(oheld), solvcon.SimpleArrayBool)
        self.assertEqual(otimes.ndarray.tolist(), [0, 10, 20, 30, 40])
        # No sample at or before t - 20 until t = 20.
        self.assertEqual(oheld.ndarray.tolist(),
                         [False, False, True, True, True])

    def test_a_false_inside_the_window_clears_it(self):
        times = u64([0, 10, 20, 30, 40, 50])
        values = bl([True, False, True, True, True, True])
        _, oheld = ts.held(times, values, span=20)
        # The false at 10 is in the window until t = 30, and the window at
        # t = 30 starts from the sample at 10, which is false as well.
        self.assertEqual(oheld.ndarray.tolist(),
                         [False, False, False, False, True, True])

    def test_the_boundary_sample_carries_the_claim(self):
        # The boundary sample is the last one stamped at or before
        # t - span. The window (10, 30] holds only true samples, but the
        # state over (10, 20) comes from the boundary sample at 10.
        times = u64([0, 10, 30])
        _, oheld = ts.held(times, bl([True, False, True]), span=20)
        self.assertEqual(oheld.ndarray.tolist(), [False, False, False])
        _, oheld = ts.held(times, bl([True, True, True]), span=20)
        self.assertEqual(oheld.ndarray.tolist(), [False, False, True])
        # A sample exactly span behind t is the boundary, not a member.
        _, oheld = ts.held(u64([0, 20]), bl([True, True]), span=20)
        self.assertEqual(oheld.ndarray.tolist(), [False, True])
        _, oheld = ts.held(u64([0, 20]), bl([False, True]), span=20)
        self.assertEqual(oheld.ndarray.tolist(), [False, False])

    def test_span_wider_than_the_log_gives_false(self):
        times = u64([0, 1, 2])
        _, oheld = ts.held(times, bl([True, True, True]), span=2**64 - 1)
        self.assertEqual(oheld.ndarray.tolist(), [False, False, False])
        big = u64([2**63, 2**63 + 1, 2**63 + 2])
        _, oheld = ts.held(big, bl([True, True, True]), span=2**63 + 5)
        self.assertEqual(oheld.ndarray.tolist(), [False, False, False])

    def test_repeated_timestamp_needs_every_sample_of_the_group(self):
        # A group on the boundary resolves to its last sample; a false
        # anywhere in a group inside the window clears it.
        times = u64([0, 10, 10, 20])
        _, oheld = ts.held(times, bl([True, True, False, True]), span=10)
        self.assertEqual(oheld.ndarray.tolist(),
                         [False, False, False, False])
        _, oheld = ts.held(times, bl([True, False, True, True]), span=10)
        self.assertEqual(oheld.ndarray.tolist(), [False, False, False, True])
        _, oheld = ts.held(u64([0, 10, 10, 15]),
                           bl([True, True, False, True]), span=10)
        self.assertEqual(oheld.ndarray.tolist(),
                         [False, False, False, False])

    def test_empty_and_single_series(self):
        otimes, oheld = ts.held(u64([]), bl([]), span=5)
        self.assertEqual((otimes.shape, oheld.shape), ((0,), (0,)))
        otimes, oheld = ts.held(u64([9]), bl([True]), span=5)
        self.assertEqual((otimes.ndarray.tolist(), oheld.ndarray.tolist()),
                         ([9], [False]))

    def test_strided_view_is_read_in_array_order(self):
        times = solvcon.SimpleArrayUint64(
            array=np.arange(12, dtype='uint64')[::4])
        values = solvcon.SimpleArrayBool(
            array=np.array([True, False, True, False, True, False],
                           dtype='bool')[::2])
        otimes, oheld = ts.held(times, values, span=4)
        self.assertEqual(otimes.ndarray.tolist(), [0, 4, 8])
        self.assertEqual(oheld.ndarray.tolist(), [False, True, True])

    def test_holds_a_thresholded_moving_average(self):
        times = u64([0, 1, 2, 3, 4, 5])
        _, smooth = ts.movavg(times, f64([9.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                              span=2)
        under = bl([v < 2.0 for v in smooth.ndarray.tolist()])
        _, oheld = ts.held(times, under, span=2)
        self.assertEqual(smooth.ndarray.tolist(),
                         [9.0, 5.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(oheld.ndarray.tolist(),
                         [False, False, False, False, True, True])

    def test_invalid_input_raises(self):
        with self.assertRaisesRegex(ValueError,
                                    "held.*span must be positive but is 0"):
            ts.held(u64([0, 1]), bl([True, True]), span=0)
        with self.assertRaisesRegex(ValueError, "held.*non-decreasing"):
            ts.held(u64([2, 1]), bl([True, True]), span=5)
        with self.assertRaisesRegex(ValueError, "held.*2 samples but values "
                                    "has 3"):
            ts.held(u64([1, 2]), bl([True, True, True]), span=5)
        with self.assertRaisesRegex(TypeError, "incompatible function "
                                    "arguments"):
            ts.held(u64([0, 1]), f64([1.0, 0.0]), span=5)


class TrueIntervalsTC(unittest.TestCase):

    def test_runs_are_half_open_rows(self):
        times = u64([0, 10, 20, 30, 40, 50])
        values = bl([False, True, True, False, True, False])
        runs = ts.true_intervals(times, values)
        self.assertIs(type(runs), solvcon.SimpleArrayUint64)
        self.assertEqual(runs.shape, (2, 3))
        self.assertEqual(runs.ndarray.tolist(),
                         [[10, 30, 20], [40, 50, 10]])

    def test_run_open_at_log_end_closes_at_the_last_sample(self):
        # The log cannot say when the run ended, so the run ends with the
        # log; a run that starts at the last sample is kept with duration 0.
        runs = ts.true_intervals(u64([0, 10, 20]), bl([False, True, True]))
        self.assertEqual(runs.ndarray.tolist(), [[10, 20, 10]])
        runs = ts.true_intervals(u64([0, 10, 20]), bl([True, False, True]))
        self.assertEqual(runs.ndarray.tolist(), [[0, 10, 10], [20, 20, 0]])
        runs = ts.true_intervals(u64([7]), bl([True]))
        self.assertEqual(runs.ndarray.tolist(), [[7, 7, 0]])

    def test_repeated_timestamp_resolves_to_its_last_sample(self):
        # Only the last sample of a group counts, both for starting a run
        # and for ending one.
        runs = ts.true_intervals(u64([0, 10, 10, 20]),
                                 bl([False, True, False, False]))
        self.assertEqual(runs.shape, (0, 3))
        runs = ts.true_intervals(u64([0, 10, 10, 20]),
                                 bl([True, False, True, False]))
        self.assertEqual(runs.ndarray.tolist(), [[0, 20, 20]])
        runs = ts.true_intervals(u64([5, 5]), bl([False, True]))
        self.assertEqual(runs.ndarray.tolist(), [[5, 5, 0]])

    def test_never_true_and_empty_give_no_rows(self):
        runs = ts.true_intervals(u64([0, 1, 2]), bl([False, False, False]))
        self.assertEqual(runs.shape, (0, 3))
        self.assertEqual(ts.true_intervals(u64([]), bl([])).shape, (0, 3))

    def test_against_a_naive_sweep(self):
        rng = np.random.default_rng(20260821)
        for _ in range(20):
            n = int(rng.integers(0, 40))
            ntimes = np.sort(rng.integers(0, 30, n, dtype='uint64'))
            nvalues = rng.random(n) < 0.5
            expected, open_start = [], None
            for i in range(n):
                if i + 1 < n and ntimes[i + 1] == ntimes[i]:
                    continue
                if nvalues[i] and open_start is None:
                    open_start = int(ntimes[i])
                elif not nvalues[i] and open_start is not None:
                    expected.append([open_start, int(ntimes[i]),
                                     int(ntimes[i]) - open_start])
                    open_start = None
            if open_start is not None:
                expected.append([open_start, int(ntimes[-1]),
                                 int(ntimes[-1]) - open_start])
            runs = ts.true_intervals(u64(ntimes), bl(nvalues))
            self.assertEqual(runs.ndarray.tolist(), expected)

    def test_strided_view_is_read_in_array_order(self):
        times = solvcon.SimpleArrayUint64(
            array=np.arange(12, dtype='uint64')[::4])
        values = solvcon.SimpleArrayBool(
            array=np.array([True, False, False, True, True, False],
                           dtype='bool')[::2])
        runs = ts.true_intervals(times, values)
        self.assertEqual(runs.ndarray.tolist(), [[0, 4, 4], [8, 8, 0]])

    def test_extracts_where_a_smoothed_signal_exceeds_a_limit(self):
        times = u64([0, 1, 2, 3, 4, 5, 6])
        _, smooth = ts.movavg(times, f64([0.0, 0.0, 6.0, 6.0, 0.0, 0.0,
                                          0.0]), span=2)
        over = bl([v > 2.0 for v in smooth.ndarray.tolist()])
        runs = ts.true_intervals(times, over)
        self.assertEqual(smooth.ndarray.tolist(),
                         [0.0, 0.0, 3.0, 6.0, 3.0, 0.0, 0.0])
        self.assertEqual(runs.ndarray.tolist(), [[2, 5, 3]])
        self.assertEqual(int(runs.ndarray[:, 2].sum()), 3)

    def test_invalid_input_raises(self):
        with self.assertRaisesRegex(ValueError,
                                    "true_intervals.*non-decreasing"):
            ts.true_intervals(u64([2, 1]), bl([True, True]))
        with self.assertRaisesRegex(ValueError, "true_intervals.*2 samples "
                                    "but values has 3"):
            ts.true_intervals(u64([1, 2]), bl([True, True, True]))
        with self.assertRaisesRegex(TypeError, "incompatible function "
                                    "arguments"):
            ts.true_intervals(u64([0, 1]), f64([1.0, 0.0]))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
