# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Kernels over recorded time series: sorted integer-nanosecond timestamps
paired with the values sampled at them.
"""


try:
    from _solvcon import timeseries as _impl  # noqa: F401
except ImportError:
    from ._solvcon import timeseries as _impl  # noqa: F401

_toload = [
    'merge_sorted_unique',
]


def _load():
    for name in _toload:  # noqa: F821
        globals()[name] = getattr(_impl, name)


__all__ = _toload


_load()
del _load
del _toload

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
