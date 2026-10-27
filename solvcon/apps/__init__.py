# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The built-in applications of solvcon.

An application owns everything specific to solving one problem set.  See
doc/source/application/index.md for the definition.

An application with a GUI registers itself with the pilot when imported, so
the pilot never imports an application.  The solver packages stay
problem-free, and an application with a GUI keeps its Qt modules behind the
pilot toggle so the problem still runs headlessly.
"""

from . import obsrefl
from . import shocktube

__all__ = [
    'obsrefl',
    'shocktube',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
