# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The oblique-shock reflection solved with the multi-dimensional Euler solver.

The computing domain is a plain rectangle.  A uniform supersonic stream
enters horizontally from the left, the top boundary holds the state behind
an incident oblique shock, the bottom slip wall reflects that shock, and the
flow leaves through the non-reflective outflow on the right.

The driver, the analysis, and the session carry the problem itself and import
without the pilot, so a script or a no-Qt build runs and judges the
reflection without the GUI that rides behind the pilot toggle.
"""

from ... import pilot

__all__, __getattr__, __dir__ = pilot.lazy_exports(__name__, {
    '_analytic': ['Reflection'],
    '_app': ['ObsreflApp'],
    '_driver': ['ObliqueShock', 'ObliqueShockRelation'],
    '_panel': ['SolutionPanel'],
    '_session': ['ReflectionSession'],
}, ungated=('_analytic', '_driver', '_session'))

if pilot.enable:
    from . import _app
    pilot.register_app("obsrefl", _app.build_app)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
