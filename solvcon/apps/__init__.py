# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Applications built on the solvcon code base.

An app is a package that owns everything specific to solving a problem set.
It can be headless, GUI-heavy, or both, and it can use everything the solvcon
code base provides.  An app is not a library: no other code builds on it, and
code that others need is factored out of it.  An app with a GUI registers
itself with the pilot when imported, so the pilot never imports an app.
The solver packages stay problem-free, and an app with a GUI keeps its Qt
modules behind the pilot toggle so the problem still runs headlessly.
"""

from . import obsrefl

__all__ = [
    'obsrefl',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
