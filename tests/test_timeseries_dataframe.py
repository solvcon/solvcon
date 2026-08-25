# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import io
import os
import pathlib
import re
import tempfile
import unittest

import numpy as np

import solvcon as sc
from solvcon import timeseries
from solvcon.track import dataframe


class TimeSeriesDataFrameTC(unittest.TestCase):

    col_sol = ['DELTA_VEL[1]', 'DELTA_VEL[2]', 'DELTA_VEL[3]']
    col_sol2 = ['DELTA_VEL[1]', 'DELTA_VEL[2]', 'EPOCH', 'DELTA_VEL[3]']

    dlc_data = """EPOCH ,DELTA_VEL[1] ,DELTA_VEL[2] ,DELTA_VEL[3]
1.6025960102293e+18,-0.18792724609375,-0.00048828125,-0.0478515625
1.60259601024931e+18,-0.1903076171875,-0.0009765625,-0.0489501953125
1.60259601026931e+18,-0.18743896484375,0.0006103515625,-0.0498046875
1.60259601028932e+18,-0.18927001953125,-0.0009765625,-0.04840087890625
1.60259601030931e+18,-0.188720703125,-0.00103759765625,-0.0504150390625
1.60259601032931e+18,-0.18951416015625,-0.000732421875,-0.0489501953125
1.60259601034931e+18,-0.18902587890625,-0.000732421875,-0.0489501953125
1.6025960103693e+18,-0.1895751953125,-0.00128173828125,-0.04925537109375
1.60259601038931e+18,-0.18841552734375,6.103515625e-05,-0.0489501953125
1.60259601040931e+18,-0.1884765625,-0.00042724609375,-0.04840087890625
"""
    modified_dlc_data = """DELTA_VEL[1] ,DELTA_VEL[2] ,EPOCH ,DELTA_VEL[3]
-0.18792724609375,-0.00048828125,1602596010229299968,-0.0478515625
-0.1903076171875,-0.0009765625,1602596010249309952,-0.0489501953125
-0.18743896484375,0.0006103515625,1602596010269309952,-0.0498046875
-0.18927001953125,-0.0009765625,1602596010289319936,-0.04840087890625
-0.188720703125,-0.00103759765625,1602596010309309952,-0.0504150390625
-0.18951416015625,-0.000732421875,1602596010329309952,-0.0489501953125
-0.18902587890625,-0.000732421875,1602596010349309952,-0.0489501953125
-0.1895751953125,-0.00128173828125,1602596010369299968,-0.04925537109375
-0.18841552734375,6.103515625e-05,1602596010389309952,-0.0489501953125
-0.1884765625,-0.00042724609375,1602596010409309952,-0.04840087890625
"""
    unsorted_dlc_data = """EPOCH ,DELTA_VEL[1] ,DELTA_VEL[2] ,DELTA_VEL[3]
1.60259601024931e+18,-0.1903076171875,-0.0009765625,-0.0489501953125
1.60259601034931e+18,-0.18902587890625,-0.000732421875,-0.0489501953125
1.60259601040931e+18,-0.1884765625,-0.00042724609375,-0.04840087890625
1.60259601032931e+18,-0.18951416015625,-0.000732421875,-0.0489501953125
1.6025960102293e+18,-0.18792724609375,-0.00048828125,-0.0478515625
1.6025960103693e+18,-0.1895751953125,-0.00128173828125,-0.04925537109375
1.60259601030931e+18,-0.188720703125,-0.00103759765625,-0.0504150390625
1.60259601026931e+18,-0.18743896484375,0.0006103515625,-0.0498046875
1.60259601028932e+18,-0.18927001953125,-0.0009765625,-0.04840087890625
1.60259601038931e+18,-0.18841552734375,6.103515625e-05,-0.0489501953125
"""

    @staticmethod
    def _u64(values):
        return sc.SimpleArrayUint64(
            array=np.array(values, dtype='uint64'))

    @staticmethod
    def _f64(values):
        return sc.SimpleArrayFloat64(
            array=np.array(values, dtype='float64'))

    def test_read_from_text_file_basic(self):
        tsdf = dataframe.DataFrame()

        tsdf.read_from_text_file(io.StringIO(self.dlc_data))
        self.assertEqual(tsdf._columns, self.col_sol)
        self.assertEqual(len(tsdf._columns), 3)
        for i in range(len(tsdf._columns)):
            self.assertEqual(tsdf._data[i].ndarray.shape[0], 10)
        self.assertEqual(tsdf._index_name, 'EPOCH')

        tsdf.read_from_text_file(
            io.StringIO(self.modified_dlc_data),
            delimiter=',',
            timestamp_column='EPOCH'
        )

        self.assertEqual(tsdf._columns, self.col_sol)
        self.assertEqual(len(tsdf._columns), 3)
        for i in range(len(tsdf._columns)):
            self.assertEqual(tsdf._data[i].ndarray.shape[0], 10)
        self.assertEqual(tsdf._index_name, 'EPOCH')

        tsdf.read_from_text_file(
            io.StringIO(self.modified_dlc_data),
            delimiter=',',
            timestamp_in_file=False
        )
        self.assertEqual(tsdf._columns, self.col_sol2)
        self.assertEqual(len(tsdf._columns), 4)
        for i in range(len(tsdf._columns)):
            self.assertEqual(tsdf._data[i].ndarray.shape[0], 10)
        self.assertEqual(tsdf._index_name, 'Index')

    def test_dataframe_attribute_columns(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))
        self.assertEqual(tsdf.columns, self.col_sol)

    def test_dataframe_attribute_shape(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))
        self.assertEqual(tsdf.shape, (10, 3))

    def test_dataframe_attribute_index(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))

        nd_arr = np.genfromtxt(io.StringIO(self.dlc_data), delimiter=',')[1:]

        self.assertEqual(
            list(tsdf.index), list(nd_arr[:, 0].astype(np.uint64))
        )
        self.assertIsInstance(tsdf.index, np.ndarray)

    def test_dataframe_get_column(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))

        col_data = tsdf['DELTA_VEL[1]']

        nd_arr = np.genfromtxt(io.StringIO(self.dlc_data), delimiter=',')[1:]

        self.assertEqual(list(col_data), list(nd_arr[:, 1]))
        self.assertIsInstance(col_data, np.ndarray)

    def test_dataframe_sort(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.unsorted_dlc_data))

        # Test out-of-place sort
        reordered_tsdf = tsdf.sort(tsdf.columns, index_column=None,
                                   inplace=False)
        col_data = reordered_tsdf['DELTA_VEL[1]']
        nd_arr = np.genfromtxt(io.StringIO(self.dlc_data), delimiter=',')[1:]
        self.assertEqual(list(col_data), list(nd_arr[:, 1]))

        # Test inplace sort_by_index
        tsdf.sort_by_index()
        col_data = tsdf['DELTA_VEL[1]']
        nd_arr = np.genfromtxt(io.StringIO(self.dlc_data), delimiter=',')[1:]
        self.assertEqual(list(col_data), list(nd_arr[:, 1]))

        # Test out-of-place sort with index_column
        tsdf.read_from_text_file(io.StringIO(self.unsorted_dlc_data),
                                 timestamp_in_file=False)

        reordered_tsdf = tsdf.sort(['EPOCH', 'DELTA_VEL[1]'],
                                   index_column='EPOCH', inplace=False)
        col_data = reordered_tsdf['DELTA_VEL[1]']
        nd_arr = np.genfromtxt(io.StringIO(self.dlc_data), delimiter=',')[1:]
        self.assertEqual(list(col_data), list(nd_arr[:, 1]))

    def test_read_from_text_file_accepts_str_path(self):
        tsdf = dataframe.DataFrame()
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False,
        ) as fh:
            fh.write(self.dlc_data)
            path = fh.name
        try:
            tsdf.read_from_text_file(path)
            self.assertEqual(tsdf._columns, self.col_sol)
            self.assertEqual(tsdf._index_name, 'EPOCH')
        finally:
            os.unlink(path)

    def test_read_from_text_file_accepts_pathlib_path(self):
        tsdf = dataframe.DataFrame()
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False,
        ) as fh:
            fh.write(self.dlc_data)
            path = pathlib.Path(fh.name)
        try:
            tsdf.read_from_text_file(path)
            self.assertEqual(tsdf._columns, self.col_sol)
            self.assertEqual(tsdf._index_name, 'EPOCH')
        finally:
            path.unlink()

    def test_read_from_text_file_missing_raises_filenotfound(self):
        tsdf = dataframe.DataFrame()
        missing = pathlib.Path(tempfile.gettempdir()) / 'no_such_file.csv'
        expected_pattern = (
            r"^Text file '" + re.escape(str(missing)) + r"' does not exist$"
        )
        with self.assertRaisesRegex(FileNotFoundError, expected_pattern):
            tsdf.read_from_text_file(missing)

    def test_from_columns_accepts_arrayplex(self):
        nindex = np.array([10, 20, 30], dtype='uint64')
        nspeed = np.array([1.0, 2.0, 3.0], dtype='float64')
        index = sc.SimpleArray(array=nindex)
        speed = sc.SimpleArray(array=nspeed)

        frame = dataframe.DataFrame.from_columns(index=index, speed=speed)

        self.assertTrue(np.shares_memory(frame.index, nindex))
        self.assertTrue(np.shares_memory(frame['speed'], nspeed))
        nindex[0] = 5
        nspeed[1] = 9.0
        self.assertEqual(frame.index.tolist(), [5, 20, 30])
        self.assertEqual(frame['speed'].tolist(), [1.0, 9.0, 3.0])

    def test_from_columns_rejects_incompatible_arrays(self):
        index = self._u64([10, 20, 30])
        with self.assertRaisesRegex(
                ValueError, "speed length 2 does not match index length 3"):
            dataframe.DataFrame.from_columns(
                index=index, speed=self._f64([1.0, 2.0]))
        with self.assertRaisesRegex(
                TypeError, "index must have dtype uint64"):
            dataframe.DataFrame.from_columns(
                index=self._f64([10.0]), speed=self._f64([1.0]))
        with self.assertRaisesRegex(
                TypeError, "index must be a numpy.ndarray or SimpleArray"):
            dataframe.DataFrame.from_columns(index=[10, 20])
        with self.assertRaisesRegex(
                ValueError, "index must be one-dimensional"):
            dataframe.DataFrame.from_columns(
                index=np.zeros((2, 2), dtype='uint64'))
        with self.assertRaisesRegex(
                TypeError, "index has unsupported dtype"):
            dataframe.DataFrame.from_columns(
                index=np.array(['a'], dtype='U1'))

    def test_setitem_replaces_an_existing_column(self):
        frame = dataframe.DataFrame.from_columns(
            index=self._u64([10, 20]), speed=self._f64([1.0, 2.0]))

        frame['speed'] = self._f64([3.0, 4.0])

        self.assertEqual(frame.columns, ['speed'])
        self.assertEqual(frame['speed'].tolist(), [3.0, 4.0])

    def test_operations_require_an_index(self):
        frame = dataframe.DataFrame()
        message = "data frame has no index"

        with self.assertRaisesRegex(ValueError, message):
            frame['speed'] = self._f64([])
        with self.assertRaisesRegex(ValueError, message):
            frame.asof(self._u64([]), self._f64([]))
        with self.assertRaisesRegex(ValueError, message):
            frame.window(0, 1)

    def test_asof_is_causal_and_uses_last_duplicate(self):
        frame = dataframe.DataFrame.from_columns(
            index=self._u64([5, 10, 15, 20, 25, 30, 35]))
        times = self._u64([10, 20, 20, 30])
        values = self._f64([1.0, 2.0, 3.0, 4.0])

        aligned, valid = frame.asof(times, values)

        self.assertEqual(valid.ndarray.tolist(),
                         [False, True, True, True, True, True, True])
        self.assertEqual(aligned.ndarray[valid.ndarray].tolist(),
                         [1.0, 1.0, 3.0, 3.0, 4.0, 4.0])
        values[3] = 99.0
        realigned, _ = frame.asof(times, values)
        self.assertEqual(realigned.ndarray[5], 99.0)

    def test_asof_empty_source_has_no_valid_values(self):
        frame = dataframe.DataFrame.from_columns(
            index=self._u64([10, 20, 30]))

        aligned, valid = frame.asof(self._u64([]), self._f64([]))

        self.assertIs(type(aligned), sc.SimpleArrayFloat64)
        self.assertEqual(aligned.ndarray.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(valid.ndarray.tolist(), [False, False, False])

    def test_asof_requires_sorted_indices(self):
        frame = dataframe.DataFrame.from_columns(
            index=self._u64([10, 30, 20]))
        with self.assertRaisesRegex(ValueError, "frame index must be sorted"):
            frame.asof(self._u64([10, 20]), self._f64([1.0, 2.0]))

        frame = dataframe.DataFrame.from_columns(
            index=self._u64([10, 20, 30]))
        with self.assertRaisesRegex(ValueError, "times must be sorted"):
            frame.asof(self._u64([20, 10]), self._f64([2.0, 1.0]))

    def test_window_is_half_open_and_zero_copy(self):
        frame = dataframe.DataFrame.from_columns(
            index=self._u64([0, 10, 20, 20, 30, 40]),
            value=self._f64([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))

        window = frame.window(10, 30)

        self.assertEqual(window.index.tolist(), [10, 20, 20])
        self.assertEqual(window['value'].tolist(), [1.0, 2.0, 3.0])
        self.assertTrue(np.shares_memory(window.index, frame.index))
        self.assertTrue(np.shares_memory(window['value'], frame['value']))
        window['value'][0] = 99.0
        self.assertEqual(frame['value'][1], 99.0)

    def test_window_cuts_on_epoch_nanosecond_bounds(self):
        # A uint64 index and a plain int bound have no common integer type,
        # so an unguarded search compares them as float64, which cannot tell
        # neighbouring nanosecond timestamps apart.
        base = 1602596010389310000
        frame = dataframe.DataFrame.from_columns(
            index=self._u64([base, base + 1, base + 2, base + 3]),
            value=self._f64([0.0, 1.0, 2.0, 3.0]))

        window = frame.window(base + 1, base + 3)

        self.assertEqual(window.index.tolist(), [base + 1, base + 2])
        self.assertEqual(window['value'].tolist(), [1.0, 2.0])

    def test_window_keeps_a_column_named_index(self):
        frame = dataframe.DataFrame.from_columns(index=self._u64([10, 20, 30]))
        frame['index'] = self._f64([4.0, 5.0, 6.0])

        window = frame.window(10, 30)

        self.assertEqual(window['index'].tolist(), [4.0, 5.0])

    def test_an_index_less_frame_is_empty(self):
        self.assertTrue(dataframe.DataFrame().empty)

    def test_window_is_empty_when_no_sample_falls_inside(self):
        frame = dataframe.DataFrame.from_columns(
            index=self._u64([10, 20]), value=self._f64([1.0, 2.0]))

        window = frame.window(20, 20)

        self.assertTrue(window.empty)
        self.assertEqual(window.columns, ['value'])

    def test_window_rejects_a_reversed_interval(self):
        frame = dataframe.DataFrame.from_columns(
            index=self._u64([10, 20]), value=self._f64([1.0, 2.0]))

        with self.assertRaisesRegex(
                ValueError, "window start must not exceed end"):
            frame.window(30, 20)

    def test_window_keeps_the_index_name(self):
        frame = dataframe.DataFrame.from_columns(index=self._u64([10, 20]))
        frame._index_name = 'Timestamp'

        self.assertEqual(frame.window(10, 20)._index_name, 'Timestamp')

    def test_sort_keeps_equal_timestamps_in_recorded_order(self):
        # asof and dedup_last both let the last row of a duplicate group win,
        # so an unstable sort would make them answer differently every run.
        frame = dataframe.DataFrame.from_columns(
            index=self._u64([1, 0] * 20),
            value=self._f64(list(range(40))))

        frame.sort()

        self.assertEqual(frame['value'].tolist()[:3], [1.0, 3.0, 5.0])

    def test_timeseries_pipeline_composes_with_frame(self):
        source = dataframe.DataFrame.from_columns(
            index=np.array([0, 10, 20, 30], dtype='uint64'),
            speed=np.array([0.0, 10.0, 30.0, 60.0], dtype='float64'),
            brake=np.array([False, True, True, False], dtype='bool'))

        output_times, acceleration = timeseries.deriv(
            source.index, source['speed'])
        frame = dataframe.DataFrame.from_columns(
            index=output_times, acceleration=acceleration)
        brake, valid = frame.asof(source.index, source['brake'])
        frame['brake'] = brake
        window = frame.window(10, 30)

        self.assertTrue(valid.ndarray.all())
        self.assertEqual(window.index.tolist(), [10, 20])
        self.assertEqual(window['acceleration'].tolist(), [1.0, 2.0])
        self.assertEqual(window['brake'].tolist(), [True, True])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
