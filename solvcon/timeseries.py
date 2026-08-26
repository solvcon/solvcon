# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Kernels over recorded time series: sorted integer-nanosecond timestamps
paired with the values sampled at them.
"""


import numpy as np

from . import core

try:
    from _solvcon import timeseries as _impl
except ImportError:
    from ._solvcon import timeseries as _impl


__all__ = [
    'merge_sorted_unique',
    'dedup_last',
    'deriv',
    'movavg',
    'held',
    'true_intervals',
]


def _as_arrays(*arrays):
    return tuple(core.SimpleArray(array=array)
                 if isinstance(array, np.ndarray) else array
                 for array in arrays)


def merge_sorted_unique(*arrays):
    """Merge sorted timestamp arrays into one sorted array of the distinct
    timestamps.

    Every input must already be sorted.
    """
    return _impl.merge_sorted_unique(*_as_arrays(*arrays))


def dedup_last(times, values):
    """Keep the last sample of every group of equal timestamps.

    Returns (times, values).
    """
    return _impl.dedup_last(*_as_arrays(times, values))


def deriv(times, values):
    """Differentiate a series by the backward difference.

    Returns (times[1:], derivatives).
    """
    return _impl.deriv(*_as_arrays(times, values))


def movavg(times, values, span):
    """Average a series over the trailing half-open window (t - span, t].

    Returns (times, means).
    """
    return _impl.movavg(*_as_arrays(times, values), span)


def held(times, values, span):
    """Report whether a boolean series was true over the trailing half-open
    window (t - span, t].

    The last sample at or before t - span must be true as well.  Returns
    (times, answers).
    """
    return _impl.held(*_as_arrays(times, values), span)


def true_intervals(times, values):
    """Run-length encode the true stretches of a boolean series into rows of
    (start, end, duration).

    A run still open at the last sample ends there.
    """
    return _impl.true_intervals(*_as_arrays(times, values))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
