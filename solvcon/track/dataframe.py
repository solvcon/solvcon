# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import contextlib
import numpy as np

from solvcon import SimpleArrayUint64, SimpleArrayFloat64

all = ['DataFrame']


class DataFrame(object):

    def __init__(self):
        self._init_members()

    def _init_members(self):
        self._columns = list()
        self._index_data = None
        self._index_name = None
        self._data = list()

    def read_from_text_file(
        self,
        fname,
        delimiter=',',
        timestamp_in_file=True,
        timestamp_column=None
    ):
        """
        Generate dataframe from a text file.

        :param fname: path to the text file.
        :type fname: str | Iterable[str] | io.StringIO
        :param delimiter: delimiter.
        :type delimiter: str
        :param timestamp_in_file: If the text file containing index column,
                   data in this column expected to be integer.
        :type timestamp_in_file: bool
        :prarm timestamp_column: Column which stores timestamp data.
        :type timestamp_column: str
        :return: None
        """

        if isinstance(fname, (str, os.PathLike)):
            if not os.path.exists(fname):
                raise FileNotFoundError(
                    "Text file '{}' does not exist".format(fname)
                )
            fid = open(fname, 'rt')
            fid_ctx = contextlib.closing(fid)
        else:
            fid = fname
            fid_ctx = contextlib.nullcontext(fid)

        with fid_ctx:
            fhd = iter(fid)

            idx_col_num = 0 if timestamp_in_file else None

            table_header = [
                x.strip() for x in next(fhd).strip().split(delimiter)
            ]
            nd_arr = np.genfromtxt(fhd, delimiter=delimiter)

            self._init_members()

            if timestamp_in_file:
                if timestamp_column in table_header:
                    idx_col_num = table_header.index(timestamp_column)
                self._index_data = SimpleArrayUint64(
                    array=nd_arr[:, idx_col_num].astype(np.uint64)
                )
                self._index_name = table_header[idx_col_num]
            else:
                self._index_data = SimpleArrayUint64(
                    array=np.arange(nd_arr.shape[0]).astype(np.uint64)
                )
                self._index_name = "Index"

            self._columns = table_header
            if idx_col_num is not None:
                self._columns.pop(idx_col_num)

            for i in range(nd_arr.shape[1]):
                if i != idx_col_num:
                    self._data.append(
                        SimpleArrayFloat64(array=nd_arr[:, i].copy())
                    )

    def __getitem__(self, name):
        if name not in self._columns:
            raise Exception("Column '{}' does not exist".format(name))
        return self._data[self._columns.index(name)].ndarray

    @property
    def columns(self):
        return self._columns

    @property
    def shape(self):
        return (self._index_data.ndarray.shape[0], len(self._data))

    @property
    def index(self):
        return self._index_data.ndarray

    def sort(self, columns=None, index_column=None, inplace=True):
        """
        Sort the dataframe along the given index column

        :param columns: column names required in reordered DataFrame
        :type columns: Option[List[str]]
        :param index_column: column name treated as the index, if None is
                                given, sort along the index
        :type index_column: Option[str]
        :param inplace: flag indicates whether to sort inplace or out-of-place
        :type inplace: bool
        :return: sorted DataFrame (return self if inplace is set to
                 True)
        """

        if index_column is None and self._index_data is None:
            raise ValueError("DataFrame: data frame has no index, "
                             "please provide index column")

        index_data = self._index_data if (
            index_column is None
            ) else self._data[self._columns.index(index_column)]
        indices = index_data.argsort()

        if inplace:
            for i, col in enumerate(self._data):
                self._data[i] = col.take_along_axis(indices)

            if self._index_data is not None:
                self._index_data = self._index_data.take_along_axis(indices)

            return self
        else:
            if columns is None:
                columns = []

            ret = DataFrame()
            ret._index_name = self._index_name
            ret._index_data = self._index_data.take_along_axis(indices)

            for name in columns:
                if name not in self._columns:
                    raise ValueError("Column '{}' does not exist".format(name))
                idx = self._columns.index(name)
                new_col = self._data[idx].take_along_axis(indices)
                ret._columns.append(name)
                ret._data.append(new_col)

            return ret

    def sort_by_index(self, columns=None, inplace=True):
        """
        Sort the dataframe along the index column

        :param columns: column names required in reordered DataFrame
        :type columns: List[str]
        :param inplace: flag indicates whether to sort inplace or out-of-place
        :type inplace: bool
        :return: sorted DataFrame (return self if inplace is set to
                True)
        """
        return self.sort(columns=columns, index_column=None, inplace=inplace)

    def reorder_columns(self, columns=None, inplace=True):
        """
        Reorder the dataframe's columns

        :param columns: full permutation of the existing column names
        :type columns: List[str]
        :param inplace: flag indicates whether to reorder inplace or
                        out-of-place
        :type inplace: bool
        :return: reordered DataFrame (return self if inplace is set to
                True)
        """
        if columns is None:
            raise ValueError("DataFrame: provide columns to reorder")

        if sorted(columns) != sorted(self._columns):
            missing = sorted(set(self._columns) - set(columns))
            unknown = sorted(set(columns) - set(self._columns))
            if missing or unknown:
                raise ValueError(
                    "DataFrame: columns missing {}, unknown {}".format(
                        missing, unknown))
            raise ValueError("DataFrame: columns has duplicate names")

        new_data = [self._data[self._columns.index(name)]
                    for name in columns]

        if inplace:
            self._data = new_data
            self._columns = list(columns)
            return self
        else:
            ret = DataFrame()
            ret._index_name = self._index_name
            ret._index_data = self._index_data
            ret._columns = list(columns)
            ret._data = new_data
            return ret

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
