# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Support for the multi-dimensional Euler-equation solver.

The package stays free of any particular problem; a problem lives in its own
application package under :mod:`solvcon.pilot.apps`.
"""

from . import field

__all__ = [
    'field',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
