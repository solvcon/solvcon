# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The oblique-shock reflection solved with the multi-dimensional Euler solver.

The computing domain is a plain rectangle.  A uniform supersonic stream
enters horizontally from the left, the top boundary holds the state behind
an incident oblique shock, the bottom slip wall reflects that shock, and the
flow leaves through the non-reflective outflow on the right.
"""

from ._driver import ObliqueShock, ObliqueShockMesher, ObliqueShockRelation

__all__ = [
    'ObliqueShock',
    'ObliqueShockMesher',
    'ObliqueShockRelation',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
