# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
One-dimensional solvers for conservation laws, grouped by equation set;
currently the Euler equations.
"""

from . import euler
from . import draw  # noqa: F401

__all__ = [
    'euler',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
