# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import io
import os
import pathlib
import re
import tempfile
import unittest

import numpy as np

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

    def test_dataframe_get_column(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))

        col_data = tsdf['DELTA_VEL[1]']

        nd_arr = np.genfromtxt(io.StringIO(self.dlc_data), delimiter=',')[1:]

        self.assertEqual(list(col_data), list(nd_arr[:, 1]))

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

    def test_dataframe_reorder_columns_inplace(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))

        col2 = list(tsdf['DELTA_VEL[2]'])
        new_order = ['DELTA_VEL[3]', 'DELTA_VEL[1]', 'DELTA_VEL[2]']

        ret = tsdf.reorder_columns(new_order)

        self.assertIs(ret, tsdf)
        self.assertEqual(tsdf.columns, new_order)
        self.assertEqual(list(tsdf['DELTA_VEL[2]']), col2)

    def test_dataframe_reorder_columns_out_of_place(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))

        original_order = list(tsdf.columns)
        new_order = ['DELTA_VEL[3]', 'DELTA_VEL[1]', 'DELTA_VEL[2]']

        reordered = tsdf.reorder_columns(new_order, inplace=False)

        self.assertIsNot(reordered, tsdf)
        self.assertEqual(reordered.columns, new_order)
        self.assertEqual(tsdf.columns, original_order)

    def test_dataframe_reorder_columns_does_not_alias_input_list(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))

        new_order = ['DELTA_VEL[3]', 'DELTA_VEL[1]', 'DELTA_VEL[2]']
        tsdf.reorder_columns(new_order)
        new_order.append('EXTRA')

        self.assertEqual(
            tsdf.columns, ['DELTA_VEL[3]', 'DELTA_VEL[1]', 'DELTA_VEL[2]']
        )

    def test_dataframe_reorder_columns_requires_permutation(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))

        missing_pattern = re.escape(
            "DataFrame: columns missing ['DELTA_VEL[3]'], unknown []"
        )
        with self.assertRaisesRegex(ValueError, missing_pattern):
            tsdf.reorder_columns(['DELTA_VEL[1]', 'DELTA_VEL[2]'])

        unknown_pattern = re.escape(
            "DataFrame: columns missing [], unknown ['EXTRA']"
        )
        with self.assertRaisesRegex(ValueError, unknown_pattern):
            tsdf.reorder_columns(
                ['DELTA_VEL[1]', 'DELTA_VEL[2]', 'DELTA_VEL[3]', 'EXTRA']
            )

    def test_dataframe_reorder_columns_rejects_duplicates(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))

        duplicate_pattern = re.escape("DataFrame: columns has duplicate names")
        with self.assertRaisesRegex(ValueError, duplicate_pattern):
            tsdf.reorder_columns(
                ['DELTA_VEL[1]', 'DELTA_VEL[1]', 'DELTA_VEL[2]',
                 'DELTA_VEL[3]']
            )

    def test_dataframe_reorder_columns_requires_columns(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))

        empty_pattern = re.escape("DataFrame: provide columns to reorder")
        with self.assertRaisesRegex(ValueError, empty_pattern):
            tsdf.reorder_columns()

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

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
