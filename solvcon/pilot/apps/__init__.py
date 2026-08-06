# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Applications built on the solvcon solvers.

An application owns everything specific to one problem: the non-GUI driver
and analysis next to the pilot GUI that presents them.  The solver packages
stay problem-free, and each application package keeps its Qt modules behind
the pilot toggle so the problem still runs headlessly.
"""

from . import obsrefl

__all__ = [
    'obsrefl',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
