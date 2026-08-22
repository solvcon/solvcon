# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Kernels over recorded time series: sorted integer-nanosecond timestamps
paired with the values sampled at them.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _solvcon.timeseries import (  # noqa: F401
        dedup_last,
        deriv,
        held,
        merge_sorted_unique,
        movavg,
        true_intervals,
    )
else:
    try:
        from _solvcon import timeseries as _impl
    except ImportError:
        from ._solvcon import timeseries as _impl

    merge_sorted_unique = _impl.merge_sorted_unique
    dedup_last = _impl.dedup_last
    deriv = _impl.deriv
    movavg = _impl.movavg
    held = _impl.held
    true_intervals = _impl.true_intervals


__all__ = [
    'merge_sorted_unique',
    'dedup_last',
    'deriv',
    'movavg',
    'held',
    'true_intervals',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
