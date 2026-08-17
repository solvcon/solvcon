# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import numpy as np

import solvcon
from solvcon import timeseries as ts


def u64(values):
    return solvcon.SimpleArrayUint64(array=np.array(values, dtype='uint64'))


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


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
