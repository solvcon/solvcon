# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import contextlib
import numpy as np

import solvcon as sc

all = ['DataFrame']

_SIMPLE_ARRAY_BY_DTYPE = {
    np.dtype('bool'): sc.SimpleArrayBool,
    np.dtype('int8'): sc.SimpleArrayInt8,
    np.dtype('int16'): sc.SimpleArrayInt16,
    np.dtype('int32'): sc.SimpleArrayInt32,
    np.dtype('int64'): sc.SimpleArrayInt64,
    np.dtype('uint8'): sc.SimpleArrayUint8,
    np.dtype('uint16'): sc.SimpleArrayUint16,
    np.dtype('uint32'): sc.SimpleArrayUint32,
    np.dtype('uint64'): sc.SimpleArrayUint64,
    np.dtype('float32'): sc.SimpleArrayFloat32,
    np.dtype('float64'): sc.SimpleArrayFloat64,
    np.dtype('complex64'): sc.SimpleArrayComplex64,
    np.dtype('complex128'): sc.SimpleArrayComplex128,
}
_ARRAY_TYPES = (sc.SimpleArray,) + tuple(_SIMPLE_ARRAY_BY_DTYPE.values())


class DataFrame(object):

    def __init__(self):
        self._init_members()

    def _init_members(self):
        self._columns = list()
        self._index_data = None
        self._index_name = None
        self._data = list()

    def _require_index(self):
        if self._index_data is None:
            raise ValueError("data frame has no index")

    @classmethod
    def _normalize_array(cls, name, array, size=None, size_name='index'):
        if isinstance(array, _ARRAY_TYPES):
            array = np.asarray(array)
        if not isinstance(array, np.ndarray):
            raise TypeError(
                "{} must be a numpy.ndarray or SimpleArray".format(name))

        if array.ndim != 1:
            raise ValueError("{} must be one-dimensional".format(name))
        if size is not None and array.shape[0] != size:
            raise ValueError(
                "{} length {} does not match {} length {}".format(
                    name, array.shape[0], size_name, size))

        return cls._typed_from_ndarray(name, array)

    @staticmethod
    def _typed_from_ndarray(name, ndarray):
        typ = _SIMPLE_ARRAY_BY_DTYPE.get(ndarray.dtype)
        if typ is None:
            raise TypeError(
                "{} has unsupported dtype {}".format(name, ndarray.dtype))
        return typ(array=ndarray)

    @staticmethod
    def _require_sorted(name, array):
        ndarray = np.asarray(array)
        if ndarray.shape[0] > 1 and np.any(ndarray[1:] < ndarray[:-1]):
            raise ValueError("{} must be sorted".format(name))

    @staticmethod
    def _slice(array, begin, end):
        return type(array)(array=np.asarray(array)[begin:end])

    @classmethod
    def _from_parts(cls, index, index_name, names=(), data=()):
        """Build a frame from parts that are already normalized."""
        ret = cls()
        ret._index_data = index
        ret._index_name = index_name
        ret._columns = list(names)
        ret._data = list(data)
        return ret

    @classmethod
    def from_columns(cls, index, **columns):
        """Construct a data frame from existing array columns."""
        index = cls._normalize_array('index', index)
        if index.ndarray.dtype != np.dtype('uint64'):
            raise TypeError("index must have dtype uint64")

        ret = cls._from_parts(index, 'Index')
        for name, column in columns.items():
            ret[name] = column
        return ret

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
                self._index_data = sc.SimpleArrayUint64(
                    array=nd_arr[:, idx_col_num].astype(np.uint64)
                )
                self._index_name = table_header[idx_col_num]
            else:
                self._index_data = sc.SimpleArrayUint64(
                    array=np.arange(nd_arr.shape[0]).astype(np.uint64)
                )
                self._index_name = "Index"

            self._columns = table_header
            if idx_col_num is not None:
                self._columns.pop(idx_col_num)

            for i in range(nd_arr.shape[1]):
                if i != idx_col_num:
                    self._data.append(
                        sc.SimpleArrayFloat64(array=nd_arr[:, i].copy())
                    )

    def __getitem__(self, name):
        if name not in self._columns:
            raise Exception("Column '{}' does not exist".format(name))
        return np.asarray(self._data[self._columns.index(name)])

    def __setitem__(self, name, column):
        self._require_index()
        column = self._normalize_array(name, column, self.shape[0])
        if name in self._columns:
            self._data[self._columns.index(name)] = column
        else:
            self._columns.append(name)
            self._data.append(column)

    @property
    def columns(self):
        return self._columns

    @property
    def shape(self):
        rows = 0 if self._index_data is None else self.index.shape[0]
        return (rows, len(self._data))

    @property
    def index(self):
        return np.asarray(self._index_data)

    @property
    def empty(self):
        return self.shape[0] == 0

    def asof(self, times, values):
        """Align values causally onto this frame's index.

        The value at each index timestamp is the last source sample at or
        before it.  The returned mask is false where the timestamp precedes
        the first source sample; what the value reads there is unspecified.
        The values are a copy, unlike the views window() returns.
        """
        self._require_index()
        times = self._normalize_array('times', times)
        if times.ndarray.dtype != np.dtype('uint64'):
            raise TypeError("times must have dtype uint64")
        values = self._normalize_array(
            'values', values, times.shape[0], 'times')
        self._require_sorted('frame index', self._index_data)
        self._require_sorted('times', times)

        found = times.searchsorted(self._index_data, side='right').ndarray
        valid = sc.SimpleArrayBool(array=found > 0)
        source = values.ndarray
        if source.shape[0] == 0:
            source = np.zeros(1, dtype=source.dtype)

        output = source[np.maximum(found - 1, 0)]
        return self._typed_from_ndarray('values', output), valid

    def window(self, start, end):
        """Return a zero-copy view over the half-open interval [start, end)."""
        self._require_index()
        if start > end:
            raise ValueError("window start must not exceed end")
        self._require_sorted('frame index', self._index_data)

        # SimpleArray compares the bound in the index type.  numpy has no
        # integer type holding both uint64 and int64, so it would search
        # through float64, which cannot tell nanosecond timestamps apart.
        begin = self._index_data.searchsorted(start, side='left')
        finish = self._index_data.searchsorted(end, side='left')
        return self._from_parts(
            self._slice(self._index_data, begin, finish), self._index_name,
            self._columns,
            [self._slice(column, begin, finish) for column in self._data])

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

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
