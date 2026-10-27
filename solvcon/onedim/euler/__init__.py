# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The one-dimensional Euler-equation solver.

The package stays free of any particular problem; a problem is solved by an
application, such as the shock tube in :mod:`solvcon.apps.shocktube`.
"""

from ._solver import Euler1DCore, Euler1DSolver

__all__ = [
    'Euler1DCore',
    'Euler1DSolver',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
